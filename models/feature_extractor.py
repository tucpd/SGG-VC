import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor
from ultralytics import YOLO
import numpy as np

# Global cache de luu YOLO model ngoai nn.Module hierarchy
_YOLO_CACHE = {}

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        # YOLO khong nen la attribute cua nn.Module vi no ghi de train() method
        # Luu path de load YOLO khi can, dung global cache
        self._yolo_path = config["sgg_config"]["yolo_model"]
        # Motion enhancement: obj_feats(512) + motion_in_box(768) + motion_around_box(768) = 2048
        self.motion_enhance = nn.Sequential(
            nn.Linear(512 + 768 + 768, 1024),  # 2048 -> 1024
            nn.ReLU(),
            nn.LayerNorm(1024)
        )
    
    @property
    def yolo(self):
        """Lazy load YOLO tu global cache de tranh nn.Module hierarchy"""
        if self._yolo_path not in _YOLO_CACHE:
            _YOLO_CACHE[self._yolo_path] = YOLO(self._yolo_path)
        return _YOLO_CACHE[self._yolo_path]
    
    def forward(self, clip_frames, keyframe):
        """
        Forward cho 1 clip don le (backward compatibility)
        clip_frames: (C, T=16, H, W) hoac (B, C, T, H, W)
        keyframe: (C, H, W) hoac (B, C, H, W)
        """
        device = next(self.parameters()).device
        clip_frames = clip_frames.to(device)
        keyframe = keyframe.to(device)
        
        if clip_frames.dim() == 4:
            clip_frames = clip_frames.permute(1, 0, 2, 3).unsqueeze(0)
        elif clip_frames.dim() == 5 and clip_frames.shape[1] == 3:
            clip_frames = clip_frames.permute(0, 2, 1, 3, 4)
        
        B, T, C, H, W = clip_frames.shape
        if H != 224 or W != 224:
            clips_flat = clip_frames.reshape(B * T, C, H, W)
            clips_flat = torch.nn.functional.interpolate(clips_flat, size=(224, 224), mode='bilinear', align_corners=False)
            clip_frames = clips_flat.reshape(B, T, C, 224, 224)
        
        motion_output = self.videomae(pixel_values=clip_frames)
        motion_feats = motion_output.last_hidden_state.mean(dim=1)

        if keyframe.dim() == 3:
            keyframe_np = (keyframe.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        else:
            keyframe_np = (keyframe.cpu().numpy() * 255).astype('uint8')
        
        results = self.yolo(keyframe_np, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return motion_feats, torch.zeros(1, 1024).to(motion_feats.device), torch.zeros(1, 4).to(motion_feats.device)
        
        obj_boxes = results[0].boxes.xyxy
        num_objs = len(obj_boxes)
        obj_feats = torch.randn(num_objs, 512).to(motion_feats.device)

        motion_in_box = self.extract_motion_in_box(motion_feats, obj_boxes, num_objs)
        motion_around_box = self.extract_motion_around_box(motion_feats, obj_boxes, num_objs)
        enhanced_feats = self.motion_enhance(torch.cat([obj_feats, motion_in_box, motion_around_box], dim=-1))

        return motion_feats, enhanced_feats, obj_boxes
    
    def forward_batch_clips(self, video_clips, keyframes, max_videomae_batch=2):
        """
        BATCH PROCESSING: Xu ly tat ca clips cua 1 video cung luc
        
        video_clips: tensor (num_clips, C, T=16, H, W) - all clips of 1 video
        keyframes: tensor (num_clips, C, H, W) - all keyframes of 1 video
        max_videomae_batch: max clips per VideoMAE forward (to avoid OOM, default 2)
        
        Returns:
            motion_feats_batch: (num_clips, 768) - motion features for all clips
            enhanced_feats_list: list[num_clips] of (num_objs, 1024) - enhanced features per clip
            obj_boxes_list: list[num_clips] of boxes per clip
        """
        device = next(self.parameters()).device
        video_clips = video_clips.to(device)
        keyframes = keyframes.to(device)
        
        num_clips = video_clips.shape[0]
        
        # ============ CHUNKED VIDEOMAE ============
        # Process VideoMAE in chunks to avoid OOM
        # video_clips: (num_clips, C, T, H, W) -> (num_clips, T, C, H, W)
        if video_clips.shape[1] == 3:  # (num_clips, C, T, H, W)
            clips_for_mae = video_clips.permute(0, 2, 1, 3, 4)  # -> (num_clips, T, C, H, W)
        else:
            clips_for_mae = video_clips
        
        N, T, C, H, W = clips_for_mae.shape
        if H != 224 or W != 224:
            clips_flat = clips_for_mae.reshape(N * T, C, H, W)
            clips_flat = torch.nn.functional.interpolate(clips_flat, size=(224, 224), mode='bilinear', align_corners=False)
            clips_for_mae = clips_flat.reshape(N, T, C, 224, 224)
        
        # Process in chunks to avoid OOM (default chunk=2)
        motion_feats_chunks = []
        for i in range(0, num_clips, max_videomae_batch):
            chunk = clips_for_mae[i:i+max_videomae_batch]
            with torch.cuda.amp.autocast():  # Use mixed precision for memory efficiency
                motion_output = self.videomae(pixel_values=chunk)
            chunk_feats = motion_output.last_hidden_state.mean(dim=1)  # (chunk_size, 768)
            motion_feats_chunks.append(chunk_feats.float())  # Convert back to float32
        
        motion_feats_batch = torch.cat(motion_feats_chunks, dim=0)  # (num_clips, 768)
        
        # ============ BATCH YOLO ============
        # Convert keyframes to numpy batch for YOLO
        keyframes_np = []
        for i in range(num_clips):
            kf = keyframes[i]  # (C, H, W)
            kf_np = (kf.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            keyframes_np.append(kf_np)
        
        # YOLO batch inference
        results = self.yolo(keyframes_np, verbose=False)
        
        # ============ PROCESS EACH CLIP's DETECTIONS ============
        enhanced_feats_list = []
        obj_boxes_list = []
        
        for clip_idx in range(num_clips):
            motion_feat = motion_feats_batch[clip_idx:clip_idx+1]  # (1, 768)
            
            if clip_idx < len(results) and len(results[clip_idx].boxes) > 0:
                obj_boxes = results[clip_idx].boxes.xyxy
                num_objs = len(obj_boxes)
                obj_feats = torch.randn(num_objs, 512).to(device)
                
                motion_in_box = self.extract_motion_in_box(motion_feat, obj_boxes, num_objs)
                motion_around_box = self.extract_motion_around_box(motion_feat, obj_boxes, num_objs)
                enhanced_feats = self.motion_enhance(torch.cat([obj_feats, motion_in_box, motion_around_box], dim=-1))
            else:
                obj_boxes = torch.zeros(1, 4).to(device)
                enhanced_feats = torch.zeros(1, 1024).to(device)
            
            enhanced_feats_list.append(enhanced_feats)
            obj_boxes_list.append(obj_boxes)
        
        return motion_feats_batch, enhanced_feats_list, obj_boxes_list
    
    def extract_motion_in_box(self, motion_feats, boxes, num_objs):
        if motion_feats.dim() == 2:
            motion_feats = motion_feats[0]
        return motion_feats.unsqueeze(0).expand(num_objs, -1)

    def extract_motion_around_box(self, motion_feats, boxes, num_objs):
        if motion_feats.dim() == 2:
            motion_feats = motion_feats[0]
        return motion_feats.unsqueeze(0).expand(num_objs, -1)