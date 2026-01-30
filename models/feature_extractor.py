import torch
import torch.nn as nn
from transformers import VideoMAEModel
from ultralytics import YOLO

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.yolo = YOLO(config["sgg_config"]["yolo_model"])
        self.motion_enhance = nn.Sequential(
            nn.Linear(768 + 512 + 512, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024)
        )
    
    def forward(self, clip_frames, keyframe):
        # Motion features from 16 frames/clip
        # clip_frames: (C, T=16, H, W)
        if clip_frames.dim() == 4:
            clip_frames = clip_frames.unsqueeze(0)
        motion_output = self.videomae(clip_frames)
        motion_feats = motion_output.last_hidden_state.mean(dim=1)

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

        
       