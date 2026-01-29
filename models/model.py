import torch
import torch.nn as nn
from models.feature_extractor import FeatureExtractor
from models.temporal_encoder import TemporalSGEncoder
from models.qformer import QFormer
from models.decoder import CaptionHead
from sgg.sgg_benchmark import SGGModel
from config import config

class SGGClassCap(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = FeatureExtractor(config)
        self.sgg = SGGModel(config["sgg_config"]) # REACT from SGG
        self.temporal_encoder = TemporalSGEncoder(config)
        self.qformer = QFormer(config)
        self.caption_head = CaptionHead(config["decoder_config"])

    def forward(self, video_clips, keyframes, caption_tokens=None, mode='training'):
        scene_graphs = []
        for clip, keyframe in zip(video_clips, keyframes):
            motion, enhanced, _ = self.feature_extractor(clip, keyframe)
            sg = self.sgg(enhanced)
            scene_graphs.append(sg)

        temp_emb = self.temporal_encoder(scene_graphs)
        visual_prompts = self.qformer(temp_emb)

        if mode == "training":
            loss, logits = self.caption_head(visual_prompts, truth_caption=caption_tokens, mode=mode)
            return loss, logits
        else:
            generated = self.caption_head(visual_prompts, mode=mode)
            return generated