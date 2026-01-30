import torch
import torch.nn as nn
from models.feature_extractor import FeatureExtractor
from models.temporal_encoder import TemporalSGEncoder
from models.qformer import QFormer
from models.decoder import CaptionHead
from models.sgg_wrapper import SGGWrapper

class SGGClassCap(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        self.sgg = SGGWrapper(config["sgg_config"])
        self.temporal_encoder = TemporalSGEncoder(config)
        self.qformer = QFormer(config)
        self.caption_head = CaptionHead(config["decoder_config"])
        self.num_clips = config.get("temporal_encoder_config", {}).get("num_clips", 15)

    def forward(self, video_clips_batch, keyframes_batch, caption_tokens_batch=None, mode='training'):
        """
        video_clips_batch: list[B] of tensors (num_clips=15, C, T=16, H, W)
        keyframes_batch: list[B] of tensors (num_clips=15, C, H, W)
        caption_tokens_batch: list[B] of token tensors
        """
        batch_size = len(video_clips_batch)
        all_scene_graphs_batch = []
        
        for b_idx in range(batch_size):
            video_clips = video_clips_batch[b_idx]
            keyframes = keyframes_batch[b_idx]
            
            scene_graphs_per_video = []
            for clip_idx in range(self.num_clips):
                clip = video_clips[clip_idx] if video_clips.dim() == 5 else video_clips
                keyframe = keyframes[clip_idx] if keyframes.dim() == 4 else keyframes
                
                motion_feats, enhanced_feats, obj_boxes = self.feature_extractor(clip, keyframe)
                
                sg_triples = self.sgg(keyframe, enhanced_feats)
                scene_graphs_per_video.append(sg_triples)
            
            all_scene_graphs_batch.append(scene_graphs_per_video)
        
        temp_emb = self.temporal_encoder(all_scene_graphs_batch)
        visual_prompts = self.qformer(temp_emb)

        if mode == "training":
            loss, logits, _ = self.caption_head(visual_prompts, truth_caption=caption_tokens_batch, mode=mode)
            return loss, logits, temp_emb
        else:
            generated = self.caption_head(visual_prompts, mode=mode)
            return generated