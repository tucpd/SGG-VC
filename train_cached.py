"""
Training script using pre-extracted cached features.
This eliminates VideoMAE + SGG computation during training.

Usage:
    python train_cached.py --feature_dir data/features --batch_size 8 --epochs 50

Prerequisites:
    Run scripts/extract_features.py first to generate cached features.
"""

import argparse
import os
import torch
import torch.nn as nn
import random
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import pandas as pd

from models.temporal_encoder import TemporalSGEncoder
from models.qformer import QFormer
from models.decoder import CaptionHead
from dataset.dataset import CachedVideoDataset, cached_collate_fn
from utils.loss import total_loss
from validate import validate_epoch_cached, plot_training_curves


class CachedSGGClassCap(nn.Module):
    """
    Model that uses pre-extracted features (no VideoMAE/SGG).
    Only trains: TemporalEncoder + QFormer + CaptionHead projection
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temporal_encoder = TemporalSGEncoder(config)
        self.qformer = QFormer(config)
        self.caption_head = CaptionHead(config["decoder_config"])
        self.num_clips = config.get("temporal_encoder_config", {}).get("num_clips", 15)

    def forward(self, scene_graphs_batch, motion_feats_batch=None, caption_tokens_batch=None, mode='training'):
        """
        scene_graphs_batch: list[B] of list[num_clips] of scene graph triples
        motion_feats_batch: list[B] of tensor (num_clips, 768) - optional, for future use
        caption_tokens_batch: list[B] of caption strings
        """
        # Temporal encoding from scene graphs
        temp_emb = self.temporal_encoder(scene_graphs_batch)
        
        # Q-Former fusion
        visual_prompts = self.qformer(temp_emb)

        if mode == "training":
            loss, logits, _ = self.caption_head(visual_prompts, truth_caption=caption_tokens_batch, mode=mode)
            return loss, logits, temp_emb
        else:
            generated = self.caption_head(visual_prompts, mode=mode)
            return generated


def parse_args():
    parser = argparse.ArgumentParser(description="SGG-ClassCap Training with Cached Features")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='checkpoints/cached_train', help='Where to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--feature_dir', type=str, default='data/features', help='Directory containing cached features')
    parser.add_argument('--annotation_file', type=str, default='data/annotations/train_val_videodatainfo.json', help='Annotation file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--val_split', type=str, default='val', help='Validation split name')
    parser.add_argument('--num_clips', type=int, default=15, help='Number of clips per video')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of sample captions to print after each epoch')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_sample_captions(model, val_loader, device, num_samples=3):
    """Print sample generated captions vs ground truth after each epoch"""
    model.eval()
    from utils.caption_utils import get_tokenizer
    tokenizer = get_tokenizer()
    
    print("\n" + "-"*60)
    print("SAMPLE CAPTIONS:")
    print("-"*60)
    
    with torch.no_grad():
        batch = next(iter(val_loader))
        scene_graphs, motion_feats, caption_lists = batch
        
        # Only take num_samples
        scene_graphs = scene_graphs[:num_samples]
        motion_feats = motion_feats[:num_samples]
        caption_lists = caption_lists[:num_samples]
        
        generated_ids = model(scene_graphs, motion_feats, mode='inference')
        generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for i, (gen_cap, refs) in enumerate(zip(generated_captions, caption_lists)):
            ref = refs[0] if isinstance(refs, list) else refs
            print(f"[{i+1}] Generated: {gen_cap[:100]}")
            print(f"    Reference: {ref[:100]}")
            print()
    
    print("-"*60 + "\n")


def main(args):
    config = load_config(args.config)
    
    # Override config
    config['training']['batch_size'] = args.batch_size
    config['training']['num_epochs'] = args.epochs
    config['training']['lr'] = args.lr
    config['training']['checkpoint_dir'] = args.output_dir
    config['training']['num_workers'] = args.num_workers
    config['temporal_encoder_config']['num_clips'] = args.num_clips
    config['decoder_config']['num_query_tokens'] = args.num_clips
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(f'Config: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}')

    os.makedirs(args.output_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    # Initialize model (no VideoMAE/SGG)
    model = CachedSGGClassCap(config).to(device)
    
    # Freeze DeepSeek LM decoder - only train projection layer
    for param in model.caption_head.lm_decoder.parameters():
        param.requires_grad = False
    
    # Print parameter counts
    print("\n" + "="*60)
    print("MODEL PARAMETER SUMMARY (Cached Mode)")
    print("="*60)
    
    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable
    
    te_total, te_train = count_params(model.temporal_encoder)
    print(f"TemporalEncoder:     {te_total/1e6:>8.2f}M total, {te_train/1e6:>8.2f}M trainable")
    
    qf_total, qf_train = count_params(model.qformer)
    print(f"QFormer:             {qf_total/1e6:>8.2f}M total, {qf_train/1e6:>8.2f}M trainable")
    
    ch_total, ch_train = count_params(model.caption_head)
    print(f"CaptionHead:         {ch_total/1e6:>8.2f}M total, {ch_train/1e6:>8.2f}M trainable")
    
    lm_total, lm_train = count_params(model.caption_head.lm_decoder)
    print(f"  - DeepSeek-VL2:    {lm_total/1e6:>8.2f}M total, {lm_train/1e6:>8.2f}M trainable")
    
    total_params, trainable_params = count_params(model)
    print("-"*60)
    print(f"TOTAL:               {total_params/1e6:>8.2f}M total, {trainable_params/1e6:>8.2f}M trainable")
    print(f"Trainable ratio:     {trainable_params/total_params*100:.2f}%")
    print("="*60 + "\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-2
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    
    start_epoch = 0
    best_cider = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        # Load with strict=False since we only save trainable params (exclude lm_decoder)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_cider = checkpoint.get('best_cider', 0.0)
        print(f"Resume training from epoch {start_epoch}")
    
    # Load datasets with cached features
    train_dataset = CachedVideoDataset(
        annotation_file=args.annotation_file,
        feature_dir=args.feature_dir,
        split='train',
        num_clips=args.num_clips,
        is_train=True
    )

    val_dataset = CachedVideoDataset(
        annotation_file=args.annotation_file,
        feature_dir=args.feature_dir,
        split=args.val_split,
        num_clips=args.num_clips,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cached_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cached_collate_fn,
        pin_memory=True
    )

    # Results logging
    results_path = os.path.join(args.output_dir, 'results.csv')
    history = []

    # Early stopping setup
    patience_counter = 0
    
    # Training loop
    print(f"\nStarting training with {len(train_loader)} batches per epoch")
    print(f"Estimated time per epoch: {len(train_loader) * 0.1:.1f}s (vs ~{len(train_loader) * 2.6:.1f}s without caching)")
    print(f"Early stopping patience: {args.patience} epochs")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (scene_graphs, motion_feats, captions) in enumerate(progress_bar):
            # Forward
            loss_ce, logits, temp_emb = model(
                scene_graphs, motion_feats, captions, mode='training'
            )
            
            # Total loss
            loss = total_loss(
                caption_logits=logits,
                caption_targets=captions,
                temporal_emb_seq=temp_emb,
                lambda_temporal=0.1,
                caption_loss_value=loss_ce
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        # Validate
        _, metrics = validate_epoch_cached(model, val_loader, device)

        print(f"Epoch {epoch+1} | Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.4f} | "
              f"BLEU-4: {metrics['bleu4']:.4f} | CIDEr: {metrics['cider']:.4f}")
        
        # Print sample captions
        print_sample_captions(model, val_loader, device, num_samples=args.num_samples)

        scheduler.step()

        # Save results
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'epoch_time': epoch_time,
            **metrics
        }
        history.append(epoch_results)
        pd.DataFrame(history).to_csv(results_path, index=False)

        # Plot curves
        plot_training_curves(results_path, args.output_dir)

        # Get only trainable parameters state dict (exclude frozen DeepSeek-VL2)
        trainable_state_dict = {
            k: v for k, v in model.state_dict().items() 
            if not k.startswith('caption_head.lm_decoder.')
        }

        # Save best checkpoint and check early stopping
        current_cider = metrics.get('cider', 0.0)
        if current_cider > best_cider:
            best_cider = current_cider
            patience_counter = 0  # Reset patience
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainable_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cider': best_cider,
                'metrics': metrics
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved new best model (CIDEr: {current_cider:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")

        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainable_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_cider': best_cider,
            'metrics': metrics
        }, os.path.join(args.output_dir, 'last_model.pt'))
        
        # Early stopping check
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            print(f"Best CIDEr: {best_cider:.4f}")
            break
    
    print("Training finished!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
