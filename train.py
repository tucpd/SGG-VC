import argparse
import os
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
import pandas as pd

from models.model import SGGClassCap
from dataset.dataset import VideoDataset, collate_fn
from utils.loss import total_loss
from validate import validate_epoch, plot_training_curves

def parse_args():
    parser = argparse.ArgumentParser(description="SGG-ClassCap Training Script")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, help='Where to save checkpoints (overrides config)')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--freeze_sgg', action='store_true', default=True, help='Freeze SGG module (YOLO + REACT)')
    parser.add_argument('--video_dir', type=str, default=None, help='Path to video directory (overrides config)')
    parser.add_argument('--annotation_file', type=str, default=None, help='Path to annotation JSON file (overrides config)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of dataloader workers (overrides config)')
    parser.add_argument('--val_split', type=str, default='validate', help='Validation split name (validate/val/test)')
    parser.add_argument('--num_clips', type=int, default=None, help='Number of clips per video (default 15, use 8 for faster training)')
    return parser.parse_args()

def load_config(config_path):
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        from config import config
    return config

def main(args):
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.output_dir is not None:
        config['training']['checkpoint_dir'] = args.output_dir
    if args.num_workers is not None:
        config['training']['num_workers'] = args.num_workers
    if args.video_dir is not None:
        config['dataset']['video_dir'] = args.video_dir
    if args.annotation_file is not None:
        config['dataset']['annotation_file'] = args.annotation_file
    if args.num_clips is not None:
        config['dataset']['num_clips'] = args.num_clips
        config['temporal_encoder_config']['num_clips'] = args.num_clips
        config['decoder_config']['num_query_tokens'] = args.num_clips
    
    # Get final values from config
    output_dir = config['training'].get('checkpoint_dir', './checkpoints')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(f'Config: batch_size={config["training"]["batch_size"]}, epochs={config["training"]["num_epochs"]}, lr={config["training"]["lr"]}')

    os.makedirs(output_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    # 1. Inti model
    model = SGGClassCap(config).to(device)
    
    # Freeze SGG module if specified
    if args.freeze_sgg:
        # Freeze SGG module (YOLO + REACT) - da dong bang san trong SGGWrapper
        for param in model.sgg.parameters():
            param.requires_grad = False
    
    # Freeze DeepSeek LM decoder - chi train projection layer
    for param in model.caption_head.lm_decoder.parameters():
        param.requires_grad = False
        
    # Optimizer chỉ optimize các param requires_grad=True
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['lr'],
        weight_decay=1e-2
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
    )
    
    start_epoch = 0
    best_cider = 0.0  # Dung CIDEr score de save best model (cao hon = tot hon)
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_cider = checkpoint.get('best_cider', 0.0)
        print(f"Resume training from epoch {start_epoch}")
        
    # 2. Load dataset
    train_dataset = VideoDataset(
        annotation_file=config['dataset']['annotation_file'],
        video_dir=config['dataset']['video_dir'],
        split='train',
        num_clips=config.get('temporal_encoder_config', {}).get('num_clips', 15),
        is_train=True
    )

    val_dataset = VideoDataset(
        annotation_file=config['dataset']['annotation_file'],
        video_dir=config['dataset']['video_dir'],
        split=args.val_split,  # Default 'validate' for MSR-VTT, can override via --val_split
        num_clips=config.get('temporal_encoder_config', {}).get('num_clips', 15),
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True 
    )

    # Results logging
    results_path = os.path.join(output_dir, 'results.csv')
    history = []

    # 3. Training loop
    num_epochs = config['training']['num_epochs']
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (clips, keyframes, captions) in enumerate(progress_bar):
            # clips: list[B] of (15, C, T=16, H, W)
            # keyframes: list[B] of (15, C, H, W)
            # captions: list[B] of caption strings
            
            # Forward
            loss_ce, logits, temporal_emb_seq = model(
                clips, keyframes, captions, mode='training'
            )
            
            # total loss - use loss_ce directly from CaptionHead
            loss = total_loss(
                caption_logits=logits,
                caption_targets=captions,
                temporal_emb_seq=temporal_emb_seq,
                lambda_temporal=0.1,
                caption_loss_value=loss_ce  # Pre-computed loss from CaptionHead
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # validate - chi tinh metrics, khong tinh val_loss
        _, metrics = validate_epoch(model, val_loader, device)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
              f"BLEU-4: {metrics['bleu4']:.4f} | CIDEr: {metrics['cider']:.4f}")

        scheduler.step()

        # Save results to CSV
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            **metrics
        }
        history.append(epoch_results)
        pd.DataFrame(history).to_csv(results_path, index=False)

        # Plot training curves
        plot_training_curves(results_path, output_dir)

        # Save best checkpoint - dung CIDEr score thay vi val_loss
        current_cider = metrics.get('cider', 0.0)
        if current_cider > best_cider:
            best_cider = current_cider
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cider': best_cider,
                'metrics': metrics
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"Saved new best model (CIDEr: {current_cider:.4f})")

        # Save last checkpoint (for resume)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_cider': best_cider,
            'metrics': metrics
        }, os.path.join(output_dir, 'last_model.pt'))
    
    print("Training finished!")

if __name__ == "__main__":
    args = parse_args()
    main(args)