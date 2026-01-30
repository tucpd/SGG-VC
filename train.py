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
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Where to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--freeze_sgg', action='store_true', default=True, help='Freeze SGG module (YOLO + REACT)')
    parser.add_argument('--dataset', type=str, default='msrvtt', help='Dataset name: msrvtt or classroom')
    parser.add_argument('--video_dir', type=str, default=None, help='Path to video directory')
    parser.add_argument('--annotation_file', type=str, default=None, help='Path to annotation JSON file')
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
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    # Override dataset paths from arguments if provided
    if args.video_dir is not None:
        config['dataset']['video_dir'] = args.video_dir
    if args.annotation_file is not None:
        config['dataset']['annotation_file'] = args.annotation_file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    # 1. Inti model
    model = SGGClassCap(config).to(device)
    
    # Freeze SGG module if specified
    if args.freeze_sgg:
        # Freezing SGG module (YOLO + REACT)
        for param in model.feature_extractor.yolo.parameters():
            param.requires_grad = False
        for param in model.sgg.parameters():
            param.requires_grad = False
        
    # Optimizer chỉ optimize các param requires_grad=True
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['lr'],
        weight_decay=1e-2
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
    )
    
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
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
        split='val',
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
    results_path = os.path.join(args.output_dir, 'results.csv')
    history = []

    # 3. Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, (clips, keyframes, captions) in enumerate(progress_bar):
            # clips: list[B] of (15, C, T=16, H, W)
            # keyframes: list[B] of (15, C, H, W)
            # captions: list[B] of caption strings
            
            # Forward
            loss_ce, logits, temporal_emb_seq = model(
                clips, keyframes, captions, mode='training'
            )
            
            # total loss
            loss = total_loss(
                caption_logits=logits,
                caption_targets=captions,
                temporal_emb_seq=temporal_emb_seq,
                lambda_temporal=0.1
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # validate
        avg_val_loss, metrics = validate_epoch(model, val_loader, device)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"BLEU-4: {metrics['bleu4']:.4f} | CIDEr: {metrics['cider']:.4f}")

        scheduler.step()

        # Save results to CSV
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            **metrics
        }
        history.append(epoch_results)
        pd.DataFrame(history).to_csv(results_path, index=False)

        # Plot training curves
        plot_training_curves(results_path, args.output_dir)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics': metrics
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print("Saved new best model")

        # Save last checkpoint (for resume)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'metrics': metrics
        }, os.path.join(args.output_dir, 'last_model.pt'))
    
    print("Training finished!")

if __name__ == "__main__":
    args = parse_args()
    main(args)