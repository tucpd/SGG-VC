import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor
from ultralytics import YOLO

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
        clip_frames: (C, T=16, H, W) hoac (B, C, T, H, W)
        keyframe: (C, H, W) hoac (B, C, H, W)
        """
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Move inputs to correct device
        clip_frames = clip_frames.to(device)
        keyframe = keyframe.to(device)
        
        # VideoMAE expects: (batch, num_frames, channels, height, width)
        # Input clip_frames: (C, T, H, W) -> need to convert
        if clip_frames.dim() == 4:
            # (C, T, H, W) -> (1, T, C, H, W)
            clip_frames = clip_frames.permute(1, 0, 2, 3).unsqueeze(0)
        elif clip_frames.dim() == 5 and clip_frames.shape[1] == 3:
            # (B, C, T, H, W) -> (B, T, C, H, W)
            clip_frames = clip_frames.permute(0, 2, 1, 3, 4)
        
        # Resize to 224x224 if needed (VideoMAE expects 224x224)
        B, T, C, H, W = clip_frames.shape
        if H != 224 or W != 224:
            clip_frames = clip_frames.reshape(B * T, C, H, W)
            clip_frames = torch.nn.functional.interpolate(clip_frames, size=(224, 224), mode='bilinear', align_corners=False)
            clip_frames = clip_frames.reshape(B, T, C, 224, 224)
        
        # VideoMAE forward
        motion_output = self.videomae(pixel_values=clip_frames)
        motion_feats = motion_output.last_hidden_state.mean(dim=1)  # (B, 768)

        # Object detection on central keyframe
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

        # Motion Enhancement
        motion_in_box = self.extract_motion_in_box(motion_feats, obj_boxes, num_objs)
        motion_around_box = self.extract_motion_around_box(motion_feats, obj_boxes, num_objs)
        enhanced_feats = self.motion_enhance(torch.cat([obj_feats, motion_in_box, motion_around_box], dim=-1))

        return motion_feats, enhanced_feats, obj_boxes
    
    def extract_motion_in_box(self, motion_feats, boxes, num_objs):
        # Expand motion features for each detected object
        if motion_feats.dim() == 2:
            motion_feats = motion_feats[0]
        return motion_feats.unsqueeze(0).expand(num_objs, -1)

    def extract_motion_around_box(self, motion_feats, boxes, num_objs):
        # Similar expansion for surrounding motion context
        if motion_feats.dim() == 2:
            motion_feats = motion_feats[0]
        return motion_feats.unsqueeze(0).expand(num_objs, -1)

        
       