import torch
from torch.utils.data import Dataset
import cv2
import random
import os
from utils.caption_utils import get_caption_tokens

class VideoDataset(Dataset):
    def __init__(self, list_file, video_dir, num_clips=15, frames_per_clip=16, is_train=True):
        """
        list_file: Path đến train_list.txt hoặc test_list.txt (format: video_path\tcaption1;caption2;... cho MSRVTT)
        video_dir: Thư mục chứa videos (e.g., path/to/msrvtt_videos)
        is_train: True cho train (random 1 caption), False cho val/test (all captions)
        """

        self.video_data = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                video_path = parts[0]
                captions = parts[1].split(';') if len(parts) > 1 else []
                self.video_data.append((video_path, captions))

        self.video_dir = video_dir
        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip
        self.is_train = is_train

    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, idx):
        video_path, captions = self.video_data[idx]
        full_video_path = os.path.join(self.video_dir, video_path)

        # Load and split video into 15 clips
        cap = cv2.VideoCapture(full_video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {full_video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        clip_duration = total_frames / fps / self.num_clips

        clips = []
        keyframes = []
        for i in range(self.num_clips):
            start_frame = int(i * clip_duration * fps)
            end_frame = min(start_frame + self.frames_per_clip, total_frames)

            # Extract frames each clip
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            clip_frames = []        
            for _ in range(self.frames_per_clip):
                ret, frame = cap.read()
                if not ret:
                    break
                clip_frames.append(frame)

            while len(clip_frames) < self.frames_per_clip:
                clip_frames.append(clip_frames[-1])

            clips.append(torch.tensor(clip_frames).permute(3, 0, 1, 2).float() / 255.0)

            # Central keyframe
            keyframe_idx = start_frame + (end_frame - start_frame) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, keyframe_idx)
            ret, keyframe = cap.read()
            if ret:
                keyframes.append(torch.tensor(keyframe).permute(2, 0, 1).float() / 255.0)        
            else:
                keyframes.append(keyframes[-1])
        cap.release()

        # Handle captions
        if self.is_train:
            caption = random.choice(captions) if captions else ""
            caption_tokens = get_caption_tokens(caption)
            return clips, keyframes, caption_tokens
        else:
            # All captions for val/test
            caption_tokens_list = [get_caption_tokens(c) for c in captions]
            return clips, keyframes, caption_tokens_list

def collate_fn(batch):
    clips_batch = [item[0] for item in batch]
    keyframes_batch = [item[1] for item in batch]
    captions_batch = [item[2] for item in batch]

    return clips_batch, keyframes_batch, captions_batch