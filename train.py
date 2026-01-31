import argparse
import os
import torch
import torch.nn as nn
import random
import time
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
    parser.add_argument('--val_split', type=str, default='val', help='Validation split name (validate/val/test)')
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
    
    # Print parameter counts for each module
    print("\n" + "="*60)
    print("MODEL PARAMETER SUMMARY")
    print("="*60)
    
    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable
    
    # FeatureExtractor (VideoMAE + YOLO backbone)
    fe_total, fe_train = count_params(model.feature_extractor)
    print(f"FeatureExtractor:    {fe_total/1e6:>8.2f}M total, {fe_train/1e6:>8.2f}M trainable")
    
    # VideoMAE
    vm_total, vm_train = count_params(model.feature_extractor.videomae)
    print(f"  - VideoMAE:        {vm_total/1e6:>8.2f}M total, {vm_train/1e6:>8.2f}M trainable")
    
    # YOLO (in feature_extractor)
    yolo_fe_total, yolo_fe_train = count_params(model.feature_extractor.yolo)
    print(f"  - YOLO:            {yolo_fe_total/1e6:>8.2f}M total, {yolo_fe_train/1e6:>8.2f}M trainable")
    
    # SGGWrapper (YOLO + REACT)
    sgg_total, sgg_train = count_params(model.sgg)
    print(f"SGGWrapper:          {sgg_total/1e6:>8.2f}M total, {sgg_train/1e6:>8.2f}M trainable")
    
    # REACT only
    react_total, react_train = count_params(model.sgg.model)
    print(f"  - REACT:           {react_total/1e6:>8.2f}M total, {react_train/1e6:>8.2f}M trainable")
    
    # TemporalEncoder
    te_total, te_train = count_params(model.temporal_encoder)
    print(f"TemporalEncoder:     {te_total/1e6:>8.2f}M total, {te_train/1e6:>8.2f}M trainable")
    
    # QFormer
    qf_total, qf_train = count_params(model.qformer)
    print(f"QFormer:             {qf_total/1e6:>8.2f}M total, {qf_train/1e6:>8.2f}M trainable")
    
    # CaptionHead (DeepSeek-VL2 + projection)
    ch_total, ch_train = count_params(model.caption_head)
    print(f"CaptionHead:         {ch_total/1e6:>8.2f}M total, {ch_train/1e6:>8.2f}M trainable")
    
    # DeepSeek LM only
    lm_total, lm_train = count_params(model.caption_head.lm_decoder)
    print(f"  - DeepSeek-VL2:    {lm_total/1e6:>8.2f}M total, {lm_train/1e6:>8.2f}M trainable")
    
    # Total
    total_params, trainable_params = count_params(model)
    print("-"*60)
    print(f"TOTAL:               {total_params/1e6:>8.2f}M total, {trainable_params/1e6:>8.2f}M trainable")
    print(f"Trainable ratio:     {trainable_params/total_params*100:.2f}%")
    print("="*60 + "\n")
        
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
    
    # Timing accumulators for averaging
    timing_stats = {
        'feature_extractor': [],
        'sgg': [],
        'temporal_encoder': [],
        'qformer': [],
        'caption_head': [],
        'backward': [],
        'optimizer': [],
        'total': []
    }
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Reset timing for each epoch
        for key in timing_stats:
            timing_stats[key] = []
        
        for batch_idx, (clips, keyframes, captions) in enumerate(progress_bar):
            # clips: list[B] of (15, C, T=16, H, W)
            # keyframes: list[B] of (15, C, H, W)
            # captions: list[B] of caption strings
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            batch_start = time.time()
            
            # ============ DETAILED FORWARD WITH TIMING ============
            batch_size = len(clips)
            all_scene_graphs_batch = []
            
            fe_time = 0
            sgg_time = 0
            
            for b_idx in range(batch_size):
                video_clips = clips[b_idx].to(device)
                keyframes_b = keyframes[b_idx].to(device)
                
                scene_graphs_per_video = []
                for clip_idx in range(model.num_clips):
                    clip = video_clips[clip_idx]
                    keyframe = keyframes_b[clip_idx]
                    
                    # Feature Extractor timing
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t1 = time.time()
                    motion_feats, enhanced_feats, obj_boxes = model.feature_extractor(clip, keyframe)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    fe_time += time.time() - t1
                    
                    # SGG timing
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t2 = time.time()
                    sg_triples = model.sgg(keyframe, enhanced_feats)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    sgg_time += time.time() - t2
                    
                    scene_graphs_per_video.append(sg_triples)
                
                all_scene_graphs_batch.append(scene_graphs_per_video)
            
            timing_stats['feature_extractor'].append(fe_time)
            timing_stats['sgg'].append(sgg_time)
            
            # Temporal Encoder timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t3 = time.time()
            temp_emb = model.temporal_encoder(all_scene_graphs_batch)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            timing_stats['temporal_encoder'].append(time.time() - t3)
            
            # QFormer timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t4 = time.time()
            visual_prompts = model.qformer(temp_emb)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            timing_stats['qformer'].append(time.time() - t4)
            
            # CaptionHead timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t5 = time.time()
            loss_ce, logits, _ = model.caption_head(visual_prompts, truth_caption=captions, mode='training')
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            timing_stats['caption_head'].append(time.time() - t5)
            
            # total loss
            loss = total_loss(
                caption_logits=logits,
                caption_targets=captions,
                temporal_emb_seq=temp_emb,
                lambda_temporal=0.1,
                caption_loss_value=loss_ce
            )

            # Backward timing
            optimizer.zero_grad()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t6 = time.time()
            loss.backward()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            timing_stats['backward'].append(time.time() - t6)
            
            # Optimizer timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t7 = time.time()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            timing_stats['optimizer'].append(time.time() - t7)
            
            # Total batch time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            timing_stats['total'].append(time.time() - batch_start)

            train_loss += loss.item()
            
            # Print timing every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_total = sum(timing_stats['total'][-50:]) / min(50, len(timing_stats['total']))
                print(f"\n[Batch {batch_idx+1}] Timing (avg of last 50 batches):")
                print(f"  FeatureExtractor: {sum(timing_stats['feature_extractor'][-50:])/min(50, len(timing_stats['feature_extractor'])):.3f}s")
                print(f"  SGG:              {sum(timing_stats['sgg'][-50:])/min(50, len(timing_stats['sgg'])):.3f}s")
                print(f"  TemporalEncoder:  {sum(timing_stats['temporal_encoder'][-50:])/min(50, len(timing_stats['temporal_encoder'])):.3f}s")
                print(f"  QFormer:          {sum(timing_stats['qformer'][-50:])/min(50, len(timing_stats['qformer'])):.3f}s")
                print(f"  CaptionHead:      {sum(timing_stats['caption_head'][-50:])/min(50, len(timing_stats['caption_head'])):.3f}s")
                print(f"  Backward:         {sum(timing_stats['backward'][-50:])/min(50, len(timing_stats['backward'])):.3f}s")
                print(f"  Optimizer:        {sum(timing_stats['optimizer'][-50:])/min(50, len(timing_stats['optimizer'])):.3f}s")
                print(f"  TOTAL:            {avg_total:.3f}s")
            
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        
        # Print epoch timing summary
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1} TIMING SUMMARY (avg per batch)")
        print(f"{'='*60}")
        print(f"{'Module':<20} {'Time (s)':<12} {'Percentage':<12}")
        print(f"{'-'*60}")
        avg_total = sum(timing_stats['total']) / len(timing_stats['total'])
        for key in ['feature_extractor', 'sgg', 'temporal_encoder', 'qformer', 'caption_head', 'backward', 'optimizer']:
            avg_time = sum(timing_stats[key]) / len(timing_stats[key])
            pct = (avg_time / avg_total) * 100
            print(f"{key:<20} {avg_time:<12.3f} {pct:<12.1f}%")
        print(f"{'-'*60}")
        print(f"{'TOTAL':<20} {avg_total:<12.3f} {'100.0':<12}%")
        print(f"{'='*60}\n")

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