import torch
from torch.utils.data import Dataset
import cv2
import random
import os
import json
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, annotation_file, video_dir, split='train', num_clips=15, frames_per_clip=16, is_train=True):
        """
        annotation_file: Path den file JSON annotation (train_val_videodatainfo.json)
        video_dir: Thu muc chua videos (e.g., data/videos/all)
        split: 'train' hoac 'val' de filter du lieu
        is_train: True cho train (random 1 caption), False cho val/test (all captions)
        """
        self.video_data = []
        self.video_dir = video_dir
        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip
        self.is_train = is_train
        
        # Load JSON annotation
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Group captions by video_id and filter by split
        video_captions = {}
        for item in data.get('sentences', []):
            if item.get('split') != split:
                continue
            video_id = item['video_id']
            caption = item['caption'].strip()
            if video_id not in video_captions:
                video_captions[video_id] = []
            video_captions[video_id].append(caption)
        
        # Convert to list format
        for video_id, captions in video_captions.items():
            video_path = f"{video_id}.mp4"
            self.video_data.append((video_path, captions))
        
        print(f"Loaded {len(self.video_data)} videos for split '{split}'")

    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, idx):
        video_path, captions = self.video_data[idx]
        full_video_path = os.path.join(self.video_dir, video_path)

        # Load and split video into clips
        cap = cv2.VideoCapture(full_video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video file: {full_video_path}")
            return self._get_dummy_data(captions)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0 or fps == 0:
            cap.release()
            return self._get_dummy_data(captions)
        
        clip_duration = total_frames / self.num_clips

        clips = []
        keyframes = []
        for i in range(self.num_clips):
            start_frame = int(i * clip_duration)
            end_frame = min(int((i + 1) * clip_duration), total_frames)

            # Extract frames for each clip
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            clip_frames = []        
            for _ in range(self.frames_per_clip):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                clip_frames.append(frame)

            # Padding if needed
            if len(clip_frames) == 0:
                clip_frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
            
            while len(clip_frames) < self.frames_per_clip:
                clip_frames.append(clip_frames[-1].copy())

            clips.append(torch.tensor(np.array(clip_frames)).permute(3, 0, 1, 2).float() / 255.0)

            # Central keyframe
            keyframe_idx = start_frame + (end_frame - start_frame) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, keyframe_idx)
            ret, keyframe = cap.read()
            if ret:
                keyframe = cv2.resize(keyframe, (224, 224))
                keyframes.append(torch.tensor(keyframe).permute(2, 0, 1).float() / 255.0)        
            else:
                if len(keyframes) > 0:
                    keyframes.append(keyframes[-1].clone())
                else:
                    keyframes.append(torch.zeros(3, 224, 224))
        
        cap.release()
        
        # Stack all clips and keyframes
        clips_tensor = torch.stack(clips)
        keyframes_tensor = torch.stack(keyframes)

        # Handle captions
        if self.is_train:
            caption = random.choice(captions) if captions else ""
            return clips_tensor, keyframes_tensor, caption
        else:
            return clips_tensor, keyframes_tensor, captions
    
    def _get_dummy_data(self, captions):
        """Return dummy data when video loading fails"""
        clips_tensor = torch.zeros(self.num_clips, 3, self.frames_per_clip, 224, 224)
        keyframes_tensor = torch.zeros(self.num_clips, 3, 224, 224)
        if self.is_train:
            caption = captions[0] if captions else ""
            return clips_tensor, keyframes_tensor, caption
        else:
            return clips_tensor, keyframes_tensor, captions

def collate_fn(batch):
    clips_batch = [item[0] for item in batch]
    keyframes_batch = [item[1] for item in batch]
    captions_batch = [item[2] for item in batch]

    return clips_batch, keyframes_batch, captions_batch


class CachedVideoDataset(Dataset):
    """
    Dataset that loads pre-extracted features from .pt files.
    Use with scripts/extract_features.py to pre-compute features.
    
    This eliminates VideoMAE + SGG computation during training,
    reducing training time by ~90%.
    """
    def __init__(self, annotation_file, feature_dir, split='train', num_clips=15, is_train=True):
        """
        annotation_file: Path to JSON annotation file
        feature_dir: Directory containing pre-extracted .pt files
        split: 'train' or 'val' to filter data
        is_train: True for train (random 1 caption), False for val/test (all captions)
        """
        self.video_data = []
        self.feature_dir = feature_dir
        self.num_clips = num_clips
        self.is_train = is_train
        
        # Load JSON annotation
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Group captions by video_id and filter by split
        video_captions = {}
        for item in data.get('sentences', []):
            if item.get('split') != split:
                continue
            video_id = item['video_id']
            caption = item['caption'].strip()
            if video_id not in video_captions:
                video_captions[video_id] = []
            video_captions[video_id].append(caption)
        
        # Filter only videos that have pre-extracted features
        missing_count = 0
        for video_id, captions in video_captions.items():
            feature_path = os.path.join(feature_dir, f'{video_id}.pt')
            if os.path.exists(feature_path):
                self.video_data.append((video_id, captions))
            else:
                missing_count += 1
        
        print(f"[CachedVideoDataset] Loaded {len(self.video_data)} videos for split '{split}'")
        if missing_count > 0:
            print(f"[CachedVideoDataset] Warning: {missing_count} videos missing features")

    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, idx):
        video_id, captions = self.video_data[idx]
        feature_path = os.path.join(self.feature_dir, f'{video_id}.pt')
        
        try:
            features = torch.load(feature_path, map_location='cpu')
            scene_graphs = features['scene_graphs']
            motion_feats = features['motion_feats']
        except Exception as e:
            print(f"Warning: Failed to load features for {video_id}: {e}")
            return self._get_dummy_data(captions)
        
        # Handle captions
        if self.is_train:
            caption = random.choice(captions) if captions else ""
            return scene_graphs, motion_feats, caption
        else:
            return scene_graphs, motion_feats, captions
    
    def _get_dummy_data(self, captions):
        """Return dummy data when feature loading fails"""
        # Create empty scene graphs (list of 15 empty lists)
        scene_graphs = [[] for _ in range(self.num_clips)]
        # Create zero motion features (15, 768)
        motion_feats = torch.zeros(self.num_clips, 768)
        
        if self.is_train:
            caption = captions[0] if captions else ""
            return scene_graphs, motion_feats, caption
        else:
            return scene_graphs, motion_feats, captions


def cached_collate_fn(batch):
    """Collate function for CachedVideoDataset"""
    scene_graphs_batch = [item[0] for item in batch]
    motion_feats_batch = [item[1] for item in batch]
    captions_batch = [item[2] for item in batch]
    
    return scene_graphs_batch, motion_feats_batch, captions_batch