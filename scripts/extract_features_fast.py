"""
Optimized script to extract and cache features with batch processing.
Uses batched VideoMAE and batched YOLO for higher GPU utilization.

Usage:
    python scripts/extract_features_fast.py --video_dir data/videos --output_dir data/features --num_clips 15

Expected speedup: 2-3x faster than sequential version.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import yaml
from concurrent.futures import ThreadPoolExecutor
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_extractor import FeatureExtractor
from models.sgg_wrapper import SGGWrapper


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_frames_from_video(video_path, num_clips=15, frames_per_clip=16):
    """Extract frames from video with optimized reading"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < frames_per_clip:
        cap.release()
        return None, None
    
    clip_length = total_frames // num_clips
    
    clips = []
    keyframes = []
    
    for clip_idx in range(num_clips):
        start_frame = clip_idx * clip_length
        end_frame = min(start_frame + clip_length, total_frames)
        
        frame_indices = np.linspace(start_frame, end_frame - 1, frames_per_clip, dtype=int)
        
        clip_frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize immediately to save memory
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                clip_frames.append(frame_resized)
            else:
                if clip_frames:
                    clip_frames.append(clip_frames[-1])
                else:
                    clip_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        clips.append(np.array(clip_frames))
        
        # Keyframe - resize to 640 for SGG
        keyframe_idx = (start_frame + end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, keyframe_idx)
        ret, keyframe = cap.read()
        if ret:
            keyframe_rgb = cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)
            keyframe_resized = cv2.resize(keyframe_rgb, (640, 640))
            keyframes.append(keyframe_resized)
        else:
            keyframes.append(np.zeros((640, 640, 3), dtype=np.uint8))
    
    cap.release()
    return clips, keyframes


def preprocess_clips_batch(clips_list):
    """
    Batch preprocess all clips for VideoMAE
    clips_list: list of numpy arrays (T, H, W, 3) uint8, already resized to 224
    Returns: tensor (num_clips, T, C, H, W) for VideoMAE
    """
    batch = []
    for clip_frames in clips_list:
        clip_np = clip_frames.astype(np.float32) / 255.0
        clip_tensor = torch.from_numpy(clip_np).permute(0, 3, 1, 2)  # (T, C, H, W)
        batch.append(clip_tensor)
    
    return torch.stack(batch)  # (num_clips, T, C, H, W)


def preprocess_keyframes_batch(keyframes_list):
    """
    Batch preprocess all keyframes for SGG
    keyframes_list: list of numpy arrays (H, W, 3) uint8, already resized to 640
    Returns: tensor (num_clips, C, H, W)
    """
    batch = []
    for keyframe in keyframes_list:
        frame_np = keyframe.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)  # (C, H, W)
        batch.append(frame_tensor)
    
    return torch.stack(batch)  # (num_clips, C, H, W)


def extract_single_video_batched(video_path, feature_extractor, sgg_wrapper, num_clips=15, frames_per_clip=16, device='cuda'):
    """
    Extract features for a single video using batched processing.
    Processes all clips at once through VideoMAE (in chunks to avoid OOM).
    """
    clips, keyframes = extract_frames_from_video(video_path, num_clips, frames_per_clip)
    
    if clips is None:
        return None
    
    scene_graphs = []
    
    with torch.no_grad():
        # ============ BATCHED VIDEOMAE ============
        # Preprocess all clips
        clips_batch = preprocess_clips_batch(clips).to(device)  # (num_clips, T, C, H, W)
        
        # Process VideoMAE in chunks of 5 to avoid OOM
        chunk_size = 5
        motion_feats_list = []
        
        for i in range(0, num_clips, chunk_size):
            chunk = clips_batch[i:i+chunk_size]
            # VideoMAE expects (B, T, C, H, W)
            with torch.cuda.amp.autocast():
                motion_output = feature_extractor.videomae(pixel_values=chunk)
            chunk_feats = motion_output.last_hidden_state.mean(dim=1)  # (chunk_size, 768)
            motion_feats_list.append(chunk_feats.float())
        
        motion_feats_all = torch.cat(motion_feats_list, dim=0)  # (num_clips, 768)
        
        # ============ BATCHED YOLO + SGG ============
        # Preprocess all keyframes
        keyframes_batch = preprocess_keyframes_batch(keyframes)  # (num_clips, C, H, W)
        
        # Convert to numpy for YOLO batch inference
        keyframes_np = [(kf.permute(1, 2, 0).numpy() * 255).astype('uint8') for kf in keyframes_batch]
        
        # YOLO batch inference
        yolo_results = feature_extractor.yolo(keyframes_np, verbose=False)
        
        # Process each clip's SGG (SGG doesn't support true batching)
        for clip_idx in range(num_clips):
            keyframe_tensor = keyframes_batch[clip_idx].to(device)
            motion_feat = motion_feats_all[clip_idx:clip_idx+1]  # (1, 768)
            
            # Get YOLO detections for this keyframe
            if clip_idx < len(yolo_results) and len(yolo_results[clip_idx].boxes) > 0:
                obj_boxes = yolo_results[clip_idx].boxes.xyxy
                num_objs = len(obj_boxes)
                obj_feats = torch.randn(num_objs, 512).to(device)
                
                # Motion enhancement
                motion_in_box = feature_extractor.extract_motion_in_box(motion_feat, obj_boxes, num_objs)
                motion_around_box = feature_extractor.extract_motion_around_box(motion_feat, obj_boxes, num_objs)
                enhanced_feats = feature_extractor.motion_enhance(
                    torch.cat([obj_feats, motion_in_box, motion_around_box], dim=-1)
                )
            else:
                enhanced_feats = torch.zeros(1, 1024).to(device)
            
            # Extract scene graph
            sg_triples = sgg_wrapper(keyframe_tensor, enhanced_feats)
            
            # Serialize scene graph
            sg_serializable = []
            for triple in sg_triples:
                triple_dict = {
                    'subject': {
                        'class': triple['subject']['class'],
                        'class_name': triple['subject']['class_name'],
                        'score': float(triple['subject']['score']),
                        'box': triple['subject']['box'].cpu().tolist() if torch.is_tensor(triple['subject']['box']) else triple['subject']['box'],
                        'feature': triple['subject']['feature'].cpu()
                    },
                    'predicate': {
                        'class': triple['predicate']['class'],
                        'class_name': triple['predicate']['class_name'],
                        'score': float(triple['predicate']['score']),
                        'embedding': triple['predicate']['embedding'].cpu()
                    },
                    'object': {
                        'class': triple['object']['class'],
                        'class_name': triple['object']['class_name'],
                        'score': float(triple['object']['score']),
                        'box': triple['object']['box'].cpu().tolist() if torch.is_tensor(triple['object']['box']) else triple['object']['box'],
                        'feature': triple['object']['feature'].cpu()
                    }
                }
                sg_serializable.append(triple_dict)
            
            scene_graphs.append(sg_serializable)
    
    return {
        'scene_graphs': scene_graphs,
        'motion_feats': motion_feats_all.cpu()
    }


# Thread-safe video loading
video_cache = {}
cache_lock = threading.Lock()

def load_video_async(video_path, num_clips, frames_per_clip):
    """Load video in background thread"""
    return extract_frames_from_video(video_path, num_clips, frames_per_clip)


def main():
    parser = argparse.ArgumentParser(description='Extract and cache features (optimized)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--video_dir', type=str, default='data/videos', help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='data/features', help='Directory to save features')
    parser.add_argument('--annotation_file', type=str, default='data/annotations/train_val_videodatainfo.json', help='Annotation file')
    parser.add_argument('--num_clips', type=int, default=15, help='Number of clips per video')
    parser.add_argument('--frames_per_clip', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--end_idx', type=int, default=-1, help='End index (-1 for all)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of video loading workers')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models
    print('Loading FeatureExtractor...')
    feature_extractor = FeatureExtractor(config).to(device)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    print('Loading SGGWrapper...')
    sgg_wrapper = SGGWrapper(config['sgg_config']).to(device)
    sgg_wrapper.eval()
    
    # Get video list
    with open(args.annotation_file, 'r') as f:
        annotations = json.load(f)
    
    video_ids = set()
    if 'sentences' in annotations:
        for item in annotations['sentences']:
            video_ids.add(item['video_id'])
    elif 'videos' in annotations:
        for video in annotations['videos']:
            video_ids.add(video['video_id'])
    
    video_ids = sorted(list(video_ids))
    print(f'Found {len(video_ids)} unique videos')
    
    if args.end_idx == -1:
        args.end_idx = len(video_ids)
    video_ids = video_ids[args.start_idx:args.end_idx]
    print(f'Processing videos {args.start_idx} to {args.end_idx} ({len(video_ids)} videos)')
    
    # Process videos with prefetching
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # Use ThreadPoolExecutor for async video loading
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Prefetch first batch of videos - larger queue to keep GPU busy
        prefetch_queue = []
        prefetch_size = min(args.num_workers * 4, len(video_ids))
        
        for i in range(prefetch_size):
            video_path = os.path.join(args.video_dir, f'{video_ids[i]}.mp4')
            future = executor.submit(load_video_async, video_path, args.num_clips, args.frames_per_clip)
            prefetch_queue.append((video_ids[i], future))
        
        next_prefetch_idx = prefetch_size
        
        for i, video_id in enumerate(tqdm(video_ids, desc='Extracting features')):
            output_path = os.path.join(args.output_dir, f'{video_id}.pt')
            
            # Skip if already processed
            if os.path.exists(output_path):
                skip_count += 1
                # Still need to consume the prefetched future
                if prefetch_queue:
                    _, future = prefetch_queue.pop(0)
                    future.result()  # Discard result
                # Prefetch next
                if next_prefetch_idx < len(video_ids):
                    video_path = os.path.join(args.video_dir, f'{video_ids[next_prefetch_idx]}.mp4')
                    future = executor.submit(load_video_async, video_path, args.num_clips, args.frames_per_clip)
                    prefetch_queue.append((video_ids[next_prefetch_idx], future))
                    next_prefetch_idx += 1
                continue
            
            video_path = os.path.join(args.video_dir, f'{video_id}.mp4')
            
            if not os.path.exists(video_path):
                print(f'Video not found: {video_path}')
                fail_count += 1
                continue
            
            try:
                # Get prefetched video data or load directly
                if prefetch_queue and prefetch_queue[0][0] == video_id:
                    _, future = prefetch_queue.pop(0)
                    clips, keyframes = future.result()
                else:
                    clips, keyframes = extract_frames_from_video(video_path, args.num_clips, args.frames_per_clip)
                
                # Prefetch next video
                if next_prefetch_idx < len(video_ids):
                    next_video_path = os.path.join(args.video_dir, f'{video_ids[next_prefetch_idx]}.mp4')
                    future = executor.submit(load_video_async, next_video_path, args.num_clips, args.frames_per_clip)
                    prefetch_queue.append((video_ids[next_prefetch_idx], future))
                    next_prefetch_idx += 1
                
                if clips is None:
                    print(f'Failed to extract frames from {video_id}')
                    fail_count += 1
                    continue
                
                # Process with batched extraction
                scene_graphs = []
                
                with torch.no_grad():
                    # Batched VideoMAE - process all 15 clips at once
                    clips_batch = preprocess_clips_batch(clips).to(device)
                    
                    chunk_size = 8  # Increased from 5 for better GPU util
                    motion_feats_list = []
                    
                    for j in range(0, args.num_clips, chunk_size):
                        chunk = clips_batch[j:j+chunk_size]
                        with torch.cuda.amp.autocast():
                            motion_output = feature_extractor.videomae(pixel_values=chunk)
                        chunk_feats = motion_output.last_hidden_state.mean(dim=1)
                        motion_feats_list.append(chunk_feats.float())
                    
                    motion_feats_all = torch.cat(motion_feats_list, dim=0)
                    
                    # Batched YOLO
                    keyframes_np = [(np.array(kf)).astype('uint8') for kf in keyframes]
                    yolo_results = feature_extractor.yolo(keyframes_np, verbose=False)
                    
                    # Process SGG for each clip
                    keyframes_batch = preprocess_keyframes_batch(keyframes)
                    
                    for clip_idx in range(args.num_clips):
                        keyframe_tensor = keyframes_batch[clip_idx].to(device)
                        motion_feat = motion_feats_all[clip_idx:clip_idx+1]
                        
                        if clip_idx < len(yolo_results) and len(yolo_results[clip_idx].boxes) > 0:
                            obj_boxes = yolo_results[clip_idx].boxes.xyxy
                            num_objs = len(obj_boxes)
                            obj_feats = torch.randn(num_objs, 512).to(device)
                            
                            motion_in_box = feature_extractor.extract_motion_in_box(motion_feat, obj_boxes, num_objs)
                            motion_around_box = feature_extractor.extract_motion_around_box(motion_feat, obj_boxes, num_objs)
                            enhanced_feats = feature_extractor.motion_enhance(
                                torch.cat([obj_feats, motion_in_box, motion_around_box], dim=-1)
                            )
                        else:
                            enhanced_feats = torch.zeros(1, 1024).to(device)
                        
                        sg_triples = sgg_wrapper(keyframe_tensor, enhanced_feats)
                        
                        sg_serializable = []
                        for triple in sg_triples:
                            triple_dict = {
                                'subject': {
                                    'class': triple['subject']['class'],
                                    'class_name': triple['subject']['class_name'],
                                    'score': float(triple['subject']['score']),
                                    'box': triple['subject']['box'].cpu().tolist() if torch.is_tensor(triple['subject']['box']) else triple['subject']['box'],
                                    'feature': triple['subject']['feature'].cpu()
                                },
                                'predicate': {
                                    'class': triple['predicate']['class'],
                                    'class_name': triple['predicate']['class_name'],
                                    'score': float(triple['predicate']['score']),
                                    'embedding': triple['predicate']['embedding'].cpu()
                                },
                                'object': {
                                    'class': triple['object']['class'],
                                    'class_name': triple['object']['class_name'],
                                    'score': float(triple['object']['score']),
                                    'box': triple['object']['box'].cpu().tolist() if torch.is_tensor(triple['object']['box']) else triple['object']['box'],
                                    'feature': triple['object']['feature'].cpu()
                                }
                            }
                            sg_serializable.append(triple_dict)
                        
                        scene_graphs.append(sg_serializable)
                
                features = {
                    'scene_graphs': scene_graphs,
                    'motion_feats': motion_feats_all.cpu()
                }
                
                torch.save(features, output_path)
                success_count += 1
                
            except Exception as e:
                print(f'Error processing {video_id}: {e}')
                fail_count += 1
                continue
            
            # Clear GPU cache periodically
            if success_count % 50 == 0:
                torch.cuda.empty_cache()
    
    print(f'\nExtraction complete!')
    print(f'Success: {success_count}')
    print(f'Failed: {fail_count}')
    print(f'Skipped (already exists): {skip_count}')


if __name__ == '__main__':
    main()
