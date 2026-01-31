"""
Script to extract and cache features (VideoMAE + SGG) for all videos.
This eliminates the need to compute features during training.

Usage:
    python scripts/extract_features.py --video_dir data/videos --output_dir data/features --num_clips 15

Output format per video:
    {video_id}.pt containing:
    - 'scene_graphs': list of 15 scene graphs (each is list of triples)
    - 'motion_feats': tensor (15, 768) - VideoMAE features per clip
    
Delete this file after use if needed.
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_extractor import FeatureExtractor
from models.sgg_wrapper import SGGWrapper


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_frames_from_video(video_path, num_clips=15, frames_per_clip=16):
    """
    Extract frames from video: 15 clips x 16 frames each + 15 keyframes
    Returns:
        clips: list of numpy arrays (num_clips, frames_per_clip, H, W, 3)
        keyframes: list of numpy arrays (num_clips, H, W, 3)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames < frames_per_clip:
        cap.release()
        return None, None
    
    # Calculate clip boundaries
    clip_length = total_frames // num_clips
    
    clips = []
    keyframes = []
    
    for clip_idx in range(num_clips):
        start_frame = clip_idx * clip_length
        end_frame = min(start_frame + clip_length, total_frames)
        
        # Sample frames_per_clip frames uniformly from this clip
        frame_indices = np.linspace(start_frame, end_frame - 1, frames_per_clip, dtype=int)
        
        clip_frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clip_frames.append(frame_rgb)
            else:
                # Duplicate last frame if read fails
                if clip_frames:
                    clip_frames.append(clip_frames[-1])
                else:
                    clip_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        clips.append(np.array(clip_frames))
        
        # Keyframe is the central frame of the clip
        keyframe_idx = (start_frame + end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, keyframe_idx)
        ret, keyframe = cap.read()
        if ret:
            keyframe_rgb = cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)
            keyframes.append(keyframe_rgb)
        else:
            keyframes.append(clip_frames[len(clip_frames)//2] if clip_frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    cap.release()
    return clips, keyframes


def preprocess_clip(clip_frames):
    """
    Convert clip frames to tensor for VideoMAE
    clip_frames: numpy array (T, H, W, 3) uint8
    Returns: tensor (C, T, H, W) float32 normalized
    """
    # Resize to 224x224
    resized = []
    for frame in clip_frames:
        frame_pil = Image.fromarray(frame)
        frame_resized = frame_pil.resize((224, 224), Image.BILINEAR)
        resized.append(np.array(frame_resized))
    
    # Stack and normalize
    clip_np = np.stack(resized, axis=0)  # (T, H, W, 3)
    clip_np = clip_np.astype(np.float32) / 255.0
    
    # Convert to (C, T, H, W)
    clip_tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2)
    
    return clip_tensor


def preprocess_keyframe(keyframe):
    """
    Convert keyframe to tensor for SGG
    keyframe: numpy array (H, W, 3) uint8
    Returns: tensor (C, H, W) float32 normalized
    """
    frame_pil = Image.fromarray(keyframe)
    frame_resized = frame_pil.resize((640, 640), Image.BILINEAR)
    frame_np = np.array(frame_resized).astype(np.float32) / 255.0
    frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)
    return frame_tensor


def extract_single_video(video_path, feature_extractor, sgg_wrapper, num_clips=15, frames_per_clip=16, device='cuda'):
    """
    Extract features for a single video
    Returns dict with scene_graphs and motion_feats
    """
    clips, keyframes = extract_frames_from_video(video_path, num_clips, frames_per_clip)
    
    if clips is None:
        return None
    
    scene_graphs = []
    motion_feats_list = []
    
    with torch.no_grad():
        for clip_idx in range(num_clips):
            # Preprocess
            clip_tensor = preprocess_clip(clips[clip_idx]).to(device)
            keyframe_tensor = preprocess_keyframe(keyframes[clip_idx]).to(device)
            
            # Extract features
            motion_feats, enhanced_feats, obj_boxes = feature_extractor(clip_tensor, keyframe_tensor)
            
            # Extract scene graph
            sg_triples = sgg_wrapper(keyframe_tensor, enhanced_feats)
            
            # Convert scene graph to serializable format
            # SGGWrapper returns format: subject/object have: feature, box, class, class_name, score
            # predicate has: embedding, class, class_name, score
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
            motion_feats_list.append(motion_feats.cpu())
    
    # Stack motion features
    motion_feats_tensor = torch.cat(motion_feats_list, dim=0)  # (num_clips, 768)
    
    return {
        'scene_graphs': scene_graphs,
        'motion_feats': motion_feats_tensor
    }


def main():
    parser = argparse.ArgumentParser(description='Extract and cache features for all videos')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--video_dir', type=str, default='data/videos', help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='data/features', help='Directory to save features')
    parser.add_argument('--annotation_file', type=str, default='data/annotations/train_val_videodatainfo.json', help='Annotation file to get video list')
    parser.add_argument('--num_clips', type=int, default=15, help='Number of clips per video')
    parser.add_argument('--frames_per_clip', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of videos to process at once')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing (for resuming)')
    parser.add_argument('--end_idx', type=int, default=-1, help='End index for processing (-1 for all)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models
    print('Loading FeatureExtractor...')
    feature_extractor = FeatureExtractor(config).to(device)
    feature_extractor.eval()
    
    # Freeze feature extractor
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    print('Loading SGGWrapper...')
    sgg_wrapper = SGGWrapper(config['sgg_config']).to(device)
    sgg_wrapper.eval()
    
    # Get video list from annotation file
    with open(args.annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Get unique video IDs
    video_ids = set()
    if 'sentences' in annotations:
        for item in annotations['sentences']:
            video_ids.add(item['video_id'])
    elif 'videos' in annotations:
        for video in annotations['videos']:
            video_ids.add(video['video_id'])
    
    video_ids = sorted(list(video_ids))
    print(f'Found {len(video_ids)} unique videos')
    
    # Apply start/end indices
    if args.end_idx == -1:
        args.end_idx = len(video_ids)
    video_ids = video_ids[args.start_idx:args.end_idx]
    print(f'Processing videos {args.start_idx} to {args.end_idx} ({len(video_ids)} videos)')
    
    # Process videos
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for video_id in tqdm(video_ids, desc='Extracting features'):
        output_path = os.path.join(args.output_dir, f'{video_id}.pt')
        
        # Skip if already processed
        if os.path.exists(output_path):
            skip_count += 1
            continue
        
        video_path = os.path.join(args.video_dir, f'{video_id}.mp4')
        
        if not os.path.exists(video_path):
            print(f'Video not found: {video_path}')
            fail_count += 1
            continue
        
        try:
            features = extract_single_video(
                video_path, 
                feature_extractor, 
                sgg_wrapper,
                num_clips=args.num_clips,
                frames_per_clip=args.frames_per_clip,
                device=device
            )
            
            if features is None:
                print(f'Failed to extract frames from {video_id}')
                fail_count += 1
                continue
            
            # Save features
            torch.save(features, output_path)
            success_count += 1
            
        except Exception as e:
            print(f'Error processing {video_id}: {e}')
            fail_count += 1
            continue
        
        # Clear GPU cache periodically
        if success_count % 100 == 0:
            torch.cuda.empty_cache()
    
    print(f'\nExtraction complete!')
    print(f'Success: {success_count}')
    print(f'Failed: {fail_count}')
    print(f'Skipped (already exists): {skip_count}')


if __name__ == '__main__':
    main()
