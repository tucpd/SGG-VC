import torch
import torch.nn as nn
from transformers import VideoMAE
from ultralytics import YOLO

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.videomae = VideoMAE.from_pretrained("MCG-NJU/videomae-base")
        self.yolo = YOLO(["sgg_config"]["yolo_model"])
        self.motion_enhance = nn.Sequential(
            nn.Linear(768 + 512, 1024),  # Fuse motion inside/around box
            nn.ReLU(),
            nn.LayerNorm(1024)
        )
    
    def forward(self, clip_frames, keyframe):
        # Motion features from 16 frames/ clip
        motion_feats = self.videomae(clip_frames).last_hidden_state.mean(dim=1) # (B, 768)

        # Object detection on central keyframe
        results = self.yolo(keyframe) # Detect objects: boxes, feats
        obj_boxes = results[0].boxes.xyxy # (N, 4)
        obj_feats = results[0].feats # (N, 512) # Assume feat extraction from YOLO

        # Motion Enhancement: Fuse motion inside/around box
        # Assume extract_motion_in_box as helper func
        motion_in_box = self.extract_motion_in_box(motion_feats, obj_boxes)
        motion_around_box = self.extract_motion_around_box(motion_feats, obj_boxes)
        enhanced_feats = self.motion_enhance(torch.cat([obj_feats, motion_in_box, motion_around_box], dim=-1))

        return motion_feats, enhanced_feats, obj_boxes # Return for SGG input
    
    def extract_motion_in_box(self, motion_feats, boxes):
        # Placeholder for motion in box extraction
        return motion_feats.mean(dim=1)

    def extract_motion_around_box(self, motion_feats, boxes):
        # Placeholder for motion around box extraction
        return motion_feats.mean(dim=1)

        
       