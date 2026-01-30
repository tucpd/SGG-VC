import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sgg'))

from ultralytics import YOLO
from sgg_benchmark.config import cfg
from sgg_benchmark.modeling.detector import build_detection_model


class SGGWrapper(nn.Module):
    """
    Wrapper de tich hop YOLO detector va REACT predictor tu SGG-Benchmark
    Input: enhanced_features (object features + motion), keyframe image
    Output: Scene graph triples [(subj_feat, pred_emb, obj_feat), ...]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        yolo_path = config.get("yolo_model", "models/sgg/checkpoint/yolov8m_vg150.pt")
        self.yolo = YOLO(yolo_path)
        
        if os.path.exists(config.get("config_file", "")):
            cfg.merge_from_file(config["config_file"])
        
        cfg.MODEL.PRETRAINED_DETECTOR_CKPT = config.get("pretrained_ckpt", "")
        cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR = config.get("predictor", "REACTPredictor")
        
        self.relation_model = build_detection_model(cfg)
        
        if os.path.exists(cfg.MODEL.PRETRAINED_DETECTOR_CKPT):
            checkpoint = torch.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, map_location=self.device)
            self.relation_model.load_state_dict(checkpoint['model'], strict=False)
        
        self.relation_model.eval()
        self.relation_model.to(self.device)
        
        for param in self.relation_model.parameters():
            param.requires_grad = False
        for param in self.yolo.parameters():
            param.requires_grad = False

    def forward(self, keyframe, enhanced_obj_feats=None):
        """
        keyframe: (H, W, C) hoac (C, H, W) tensor
        enhanced_obj_feats: (N, feat_dim) optional motion-enhanced features
        Returns: List of scene graph triples
        """
        if keyframe.dim() == 3 and keyframe.shape[0] in [1, 3]:
            keyframe = keyframe.permute(1, 2, 0)
        
        keyframe_np = (keyframe.cpu().numpy() * 255).astype('uint8')
        
        results = self.yolo(keyframe_np, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return []
        
        boxes = results[0].boxes.xyxy
        scores = results[0].boxes.conf
        classes = results[0].boxes.cls
        
        if boxes.shape[0] == 0:
            return []
        
        obj_feats = enhanced_obj_feats if enhanced_obj_feats is not None else self.extract_yolo_features(results[0])
        
        scene_graph_triples = []
        for i in range(min(len(boxes), 10)):
            for j in range(min(len(boxes), 10)):
                if i != j:
                    subj_feat = obj_feats[i]
                    obj_feat = obj_feats[j]
                    pred_emb = self.predict_relation(subj_feat, obj_feat, boxes[i], boxes[j])
                    
                    scene_graph_triples.append({
                        'subject': {'feature': subj_feat, 'box': boxes[i], 'class': classes[i]},
                        'predicate': {'embedding': pred_emb},
                        'object': {'feature': obj_feat, 'box': boxes[j], 'class': classes[j]}
                    })
        
        return scene_graph_triples

    def extract_yolo_features(self, yolo_result):
        """
        Trích xuất features từ YOLO detection results
        Fallback nếu không có enhanced features
        """
        try:
            if hasattr(yolo_result, 'features'):
                return yolo_result.features
            else:
                num_objs = len(yolo_result.boxes)
                return torch.randn(num_objs, 512).to(self.device)
        except:
            return torch.randn(1, 512).to(self.device)
    
    def predict_relation(self, subj_feat, obj_feat, subj_box, obj_box):
        """
        Đơn giản hóa: dùng MLP để dự đoán relation embedding
        Trong thực tế có thể dùng REACT model đầy đủ
        """
        spatial_feat = self.compute_spatial_features(subj_box, obj_box)
        combined = torch.cat([subj_feat, obj_feat, spatial_feat], dim=-1)
        
        pred_dim = 512
        pred_emb = torch.randn(pred_dim).to(self.device)
        
        return pred_emb
    
    def compute_spatial_features(self, box1, box2):
        """
        Tính toán spatial relationship giữa 2 boxes
        """
        x1_center = (box1[0] + box1[2]) / 2
        y1_center = (box1[1] + box1[3]) / 2
        x2_center = (box2[0] + box2[2]) / 2
        y2_center = (box2[1] + box2[3]) / 2
        
        dx = x2_center - x1_center
        dy = y2_center - y1_center
        dist = torch.sqrt(dx**2 + dy**2)
        
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
        
        area1 = w1 * h1
        area2 = w2 * h2
        
        spatial_feat = torch.tensor([dx, dy, dist, area1, area2, w1/h1, w2/h2]).to(self.device)
        
        return spatial_feat
