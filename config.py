config = {
    "embed_dim": 1024,
    "proj_dims": {
        "temporal_in": 768, # VideoMAE output dim
        "object_in": 512, # YOLO feature dim
        "context_in": 512, # CLIP or global context dim
        "out": 1024
    },
    "semantic_dim": 1024,
    "decoder_config": {
        "num_query_tokens": 15,
        "hidden_size": 1024,
        "proj_out_dim": 1280,
        "num_heads": 8,
        "decoder_model_path": "deepseek-ai/deepseek-vl2-tiny",
        "use_fp16": True,
        "num_beams": 5,
        "max_new_tokens": 32,
    },
    "sgg_config": {
        "predictor": "REACTPredictor",
        "yolo_model": "models/sgg/checkpoint/yolov8m_vg150.pt",  # Or yolov-world for open-vocab
        "config_file": "models/sgg/configs/VG150/react_yolov8m.yaml",  # From SGG submodule
        "pretrained_ckpt": "models/sgg/checkpoint/react_PSG/best_model_epoch_11.pth"  # Download from SGG model zoo
    },
    "temporal_encoder_config": {
        "num_layers": 4,
        "num_heads": 8,
        "embed_dim": 1024,
        "num_clips": 15
    },
    "dataset": {
        "type": "msrvtt",
        "annotation_file": "data/annotations/train_val_videodatainfo.json",
        "video_dir": "data/videos/all"
    },
    "training": {
        "batch_size": 8,
        "lr": 1e-5,
        "num_epochs": 10,   
        "num_workers": 4,
        "freeze_sgg": True  # Phase 1 freeze
    }
}