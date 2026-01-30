import torch
import torch.nn as nn
import numpy as np
import cv2
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sgg'))

from sgg_benchmark.config import cfg
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.structures.image_list import to_image_list
from sgg_benchmark.data.build import build_transforms


def load_vg150_classes(dict_file):
    """Load object and predicate classes from VG150 dict file"""
    with open(dict_file, 'r') as f:
        data = json.load(f)
    
    # idx_to_label: object classes (1-150), add background at index 0
    obj_classes = {0: '__background__'}
    for k, v in data['idx_to_label'].items():
        obj_classes[int(k)] = v
    
    # idx_to_predicate: relation classes (1-50), add background at index 0
    rel_classes = {0: '__background__'}
    for k, v in data['idx_to_predicate'].items():
        rel_classes[int(k)] = v
    
    return obj_classes, rel_classes


def create_fake_statistics(obj_classes, rel_classes):
    """Create minimal statistics dict for model initialization"""
    num_obj = len(obj_classes)
    num_rel = len(rel_classes)
    
    # Create fake frequency matrix (uniform distribution)
    fg_matrix = torch.ones(num_obj, num_obj, num_rel) / num_rel
    pred_dist = torch.ones(num_rel) / num_rel
    
    return {
        'fg_matrix': fg_matrix,
        'pred_dist': pred_dist,
        'obj_classes': list(obj_classes.values()),
        'rel_classes': list(rel_classes.values()),
        'predicate_new_order': list(range(num_rel)),
        'predicate_new_order_count': [1] * num_rel,
        'pred_freq': torch.ones(num_rel),
        'triplet_freq': {},
        'pred_weight': torch.ones(num_rel),
    }


class SGGWrapper(nn.Module):
    """
    Wrapper tich hop REACT predictor tu SGG-Benchmark
    Input: keyframe image (tensor hoac numpy)
    Output: Scene graph triples voi features that su tu model
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        config_file = config.get("config_file", "models/sgg/configs/VG150/react_yolov8m.yaml")
        weights_path = config.get("pretrained_ckpt", "models/sgg/checkpoint/react_VG150/best_model_epoch_9.pth")
        dict_file = config.get("dict_file", "models/sgg/datasets/VG150/VG-SGG-dicts-with-attri.json")
        
        # Load classes from dict file first (needed for model init)
        self.obj_classes, self.rel_classes = load_vg150_classes(dict_file)
        print(f"[SGGWrapper] Loaded {len(self.obj_classes)} object classes, {len(self.rel_classes)} relation classes")
        
        # Create fake statistics for model initialization
        fake_stats = create_fake_statistics(self.obj_classes, self.rel_classes)
        
        # Configure cfg for VG150 with REACT predictor
        # Match checkpoint training config exactly
        cfg.MODEL.META_ARCHITECTURE = "GeneralizedYOLO"
        cfg.MODEL.BACKBONE.TYPE = "yolo"  # Registry name is "yolo", not "yolov8"
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.BACKBONE.NMS_THRESH = 0.001
        cfg.MODEL.YOLO.SIZE = "yolov8m"
        cfg.MODEL.YOLO.OUT_CHANNELS = [192]  # Must be list, backbone code uses [0]
        cfg.MODEL.YOLO.IMG_SIZE = 640
        
        # ROI Box Head - match checkpoint
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "YOLOV8FeatureExtractor"
        cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 512
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 151
        cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
        cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
        cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (0.0625,)  # Single scale
        
        # ROI Relation Head - match checkpoint
        cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR = "REACTPredictor"
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR = "RelationFeatureExtractor"
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
        cfg.MODEL.ROI_RELATION_HEAD.USE_UNION_FEATURES = False
        cfg.MODEL.ROI_RELATION_HEAD.USE_SPATIAL_FEATURES = True
        cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM = 200
        cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES = 51
        cfg.MODEL.ROI_RELATION_HEAD.MLP_HEAD_DIM = 512
        cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM = 512
        cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM = 2048
        cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS = True
        cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_RELATION_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN = True
        
        # ROI Heads general
        cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100
        cfg.MODEL.ROI_HEADS.NMS_FILTER_DUPLICATES = True
        cfg.MODEL.ROI_HEADS.NMS = 0.2
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.3
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.1
        
        cfg.MODEL.RELATION_ON = True
        cfg.MODEL.FLIP_AUG = False
        cfg.MODEL.BOX_HEAD = False
        
        # Input and data
        cfg.INPUT.MIN_SIZE_TEST = 640
        cfg.INPUT.MAX_SIZE_TEST = 640
        cfg.INPUT.MIN_SIZE_TRAIN = 640
        cfg.INPUT.MAX_SIZE_TRAIN = 640
        cfg.INPUT.PADDING = True
        cfg.DATALOADER.SIZE_DIVISIBILITY = 32
        cfg.DATASETS.TRAIN = ("VG150_train",)
        cfg.TEST.CUSTUM_EVAL = True
        
        # Create output dir if not exists
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        # Save statistics cache
        dataset_names = cfg.DATASETS.TRAIN
        data_statistics_name = ''.join(dataset_names) + '_statistics'
        save_file = os.path.join(cfg.OUTPUT_DIR, "{}.cache".format(data_statistics_name))
        if not os.path.exists(save_file):
            torch.save(fake_stats, save_file)
            print(f"[SGGWrapper] Created statistics cache at {save_file}")
        
        self.cfg = cfg
        
        self.model = build_detection_model(cfg)
        self.model.to(self.device)
        
        self.checkpointer = DetectronCheckpointer(cfg, self.model)
        if os.path.exists(weights_path):
            self.checkpointer.load(weights_path)
            print(f"[SGGWrapper] Loaded REACT weights from {weights_path}")
        else:
            print(f"[SGGWrapper] Warning: weights not found at {weights_path}")
        
        self.transform = build_transforms(cfg, is_train=False)
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.rel_conf = config.get("rel_conf", 0.1)
        self.box_conf = config.get("box_conf", 0.3)
        
        # Patch train method cua cac submodule co train() bi override boi ultralytics
        self._patch_train_methods()
    
    def _patch_train_methods(self):
        """Patch train() method cua cac submodule de tranh conflict voi ultralytics"""
        from functools import partial
        
        def safe_train(module, mode=True):
            """Safe train method that uses nn.Module.train directly"""
            nn.Module.train(module, mode)
            return module
        
        # Patch backbone train method
        if hasattr(self.model, 'backbone'):
            self.model.backbone.train = partial(safe_train, self.model.backbone)
    
    def train(self, mode=True):
        """Override train() de khong anh huong den SGG model (luon eval)"""
        # SGGWrapper luon o che do eval, chi train cac layer khac
        super().train(mode)
        # Giu SGG model o eval mode
        self.model.eval()
        return self

    def forward(self, keyframe, enhanced_obj_feats=None):
        """
        keyframe: tensor (C, H, W) hoac (H, W, C) voi gia tri [0, 1] hoac [0, 255]
        enhanced_obj_feats: optional - chua su dung trong phien ban nay
        Returns: List of scene graph triples voi features that su
        """
        self.model.eval()
        
        if isinstance(keyframe, torch.Tensor):
            if keyframe.dim() == 3 and keyframe.shape[0] in [1, 3]:
                keyframe_np = keyframe.permute(1, 2, 0).cpu().numpy()
            else:
                keyframe_np = keyframe.cpu().numpy()
            
            if keyframe_np.max() <= 1.0:
                keyframe_np = (keyframe_np * 255).astype(np.uint8)
            else:
                keyframe_np = keyframe_np.astype(np.uint8)
        else:
            keyframe_np = keyframe
        
        if keyframe_np.shape[2] == 3:
            keyframe_np = cv2.cvtColor(keyframe_np, cv2.COLOR_BGR2RGB)
        
        orig_size = keyframe_np.shape[:2]
        keyframe_resized = cv2.resize(keyframe_np, (640, 640))
        
        target = torch.LongTensor([-1])
        image_tensor, _ = self.transform(keyframe_resized, target)
        image_list = to_image_list(image_tensor, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_list, targets=None)
        
        if len(predictions) == 0:
            return []
        
        scene_graph_triples = self._extract_scene_graph(predictions[0], orig_size)
        
        return scene_graph_triples

    def forward_batch(self, keyframes, enhanced_obj_feats_list=None):
        """
        BATCH PROCESSING: Xu ly nhieu keyframes cung luc
        Luu y: REACT model khong ho tro batch, phai xu ly tung keyframe
        Nhung co the gom YOLO detection va cac phep xu ly khac
        
        keyframes: tensor (num_clips, C, H, W) - all keyframes of 1 video
        enhanced_obj_feats_list: optional list of enhanced features per clip
        
        Returns: list[num_clips] of scene graph triples
        """
        self.model.eval()
        
        num_clips = keyframes.shape[0]
        scene_graphs_batch = []
        
        # REACT model khong ho tro true batch, xu ly tung keyframe
        # Nhung van nhanh hon vi giam overhead Python loop
        for i in range(num_clips):
            kf = keyframes[i]  # (C, H, W)
            sg_triples = self.forward(kf, None)
            scene_graphs_batch.append(sg_triples)
        
        return scene_graphs_batch

    def _extract_scene_graph(self, boxlist, orig_size):
        """
        Chuyen doi BoxList predictions thanh scene graph triples
        """
        height, width = orig_size
        boxlist = boxlist.resize((width, height))
        
        if not boxlist.has_field('pred_rel_scores'):
            return []
        
        boxes = boxlist.bbox
        pred_labels = boxlist.get_field('pred_labels')
        pred_scores = boxlist.get_field('pred_scores')
        rel_pair_idxs = boxlist.get_field('rel_pair_idxs')
        rel_scores = boxlist.get_field('pred_rel_scores')
        rel_labels = boxlist.get_field('pred_rel_labels')
        
        if rel_scores.shape[0] == 0:
            return []
        
        max_rel_scores = rel_scores[:, 1:].max(dim=1)[0]
        
        combined_scores = max_rel_scores * pred_scores[rel_pair_idxs[:, 0]] * pred_scores[rel_pair_idxs[:, 1]]
        
        valid_mask = combined_scores > self.rel_conf
        
        scene_graph_triples = []
        
        for idx in range(rel_pair_idxs.shape[0]):
            if not valid_mask[idx]:
                continue
            
            subj_idx = rel_pair_idxs[idx, 0].item()
            obj_idx = rel_pair_idxs[idx, 1].item()
            
            subj_box = boxes[subj_idx]
            obj_box = boxes[obj_idx]
            subj_class = pred_labels[subj_idx].item()
            obj_class = pred_labels[obj_idx].item()
            subj_score = pred_scores[subj_idx].item()
            obj_score = pred_scores[obj_idx].item()
            
            pred_class = rel_labels[idx].item()
            pred_score = combined_scores[idx].item()
            
            subj_feat = self._encode_object_feature(subj_box, subj_class, subj_score)
            obj_feat = self._encode_object_feature(obj_box, obj_class, obj_score)
            pred_emb = self._encode_predicate_feature(pred_class, pred_score, subj_box, obj_box)
            
            triple = {
                'subject': {
                    'feature': subj_feat,
                    'box': subj_box,
                    'class': subj_class,
                    'class_name': self.obj_classes.get(subj_class, 'unknown'),
                    'score': subj_score
                },
                'predicate': {
                    'embedding': pred_emb,
                    'class': pred_class,
                    'class_name': self.rel_classes.get(pred_class, 'unknown'),
                    'score': pred_score
                },
                'object': {
                    'feature': obj_feat,
                    'box': obj_box,
                    'class': obj_class,
                    'class_name': self.obj_classes.get(obj_class, 'unknown'),
                    'score': obj_score
                }
            }
            scene_graph_triples.append(triple)
        
        return scene_graph_triples

    def _encode_object_feature(self, box, class_id, score):
        """
        Tao object feature vector tu box, class va score
        Ket hop spatial + semantic information
        """
        x1, y1, x2, y2 = box.tolist() if isinstance(box, torch.Tensor) else box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect = w / (h + 1e-6)
        
        spatial_feat = torch.tensor([x1, y1, x2, y2, cx, cy, w, h, area, aspect, score], 
                                     dtype=torch.float32, device=self.device)
        
        class_onehot = torch.zeros(len(self.obj_classes) + 1, device=self.device)
        if class_id < len(class_onehot):
            class_onehot[class_id] = 1.0
        
        padding = torch.zeros(512 - 11 - len(class_onehot), device=self.device)
        
        feature = torch.cat([spatial_feat, class_onehot, padding])
        
        return feature

    def _encode_predicate_feature(self, pred_class, pred_score, subj_box, obj_box):
        """
        Tao predicate embedding tu class, score va spatial relation
        """
        s_x1, s_y1, s_x2, s_y2 = subj_box.tolist() if isinstance(subj_box, torch.Tensor) else subj_box
        o_x1, o_y1, o_x2, o_y2 = obj_box.tolist() if isinstance(obj_box, torch.Tensor) else obj_box
        
        s_cx, s_cy = (s_x1 + s_x2) / 2, (s_y1 + s_y2) / 2
        o_cx, o_cy = (o_x1 + o_x2) / 2, (o_y1 + o_y2) / 2
        
        dx = o_cx - s_cx
        dy = o_cy - s_cy
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        s_area = (s_x2 - s_x1) * (s_y2 - s_y1)
        o_area = (o_x2 - o_x1) * (o_y2 - o_y1)
        area_ratio = s_area / (o_area + 1e-6)
        
        spatial_feat = torch.tensor([dx, dy, dist, angle, area_ratio, pred_score], 
                                     dtype=torch.float32, device=self.device)
        
        class_onehot = torch.zeros(len(self.rel_classes) + 1, device=self.device)
        if pred_class < len(class_onehot):
            class_onehot[pred_class] = 1.0
        
        padding = torch.zeros(512 - 6 - len(class_onehot), device=self.device)
        
        embedding = torch.cat([spatial_feat, class_onehot, padding])
        
        return embedding

    def get_class_names(self):
        """Tra ve danh sach object va relation classes"""
        return {
            'obj_classes': self.obj_classes,
            'rel_classes': self.rel_classes
        }
