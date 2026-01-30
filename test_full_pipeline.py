"""
Test script for SGGClassCap full pipeline
Test: FeatureExtractor -> SGGWrapper -> TemporalSGEncoder -> QFormer -> CaptionHead
"""
import torch
import numpy as np
import yaml
import os
import sys
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config():
    """Load config from yaml file"""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dummy_data(batch_size=1, num_clips=15, img_size=224, device="cuda"):
    """Create dummy video clips and keyframes for testing"""
    # video_clips: (batch_size,) list of (num_clips, C, T, H, W) tensors
    # keyframes: (batch_size,) list of (num_clips, C, H, W) tensors
    
    video_clips_batch = []
    keyframes_batch = []
    
    for b in range(batch_size):
        # VideoMAE expects (C, T, H, W) format
        video_clips = torch.randn(num_clips, 3, 16, img_size, img_size).to(device)
        keyframes = torch.randn(num_clips, 3, img_size, img_size).to(device)
        
        # Normalize to [0, 1]
        video_clips = (video_clips - video_clips.min()) / (video_clips.max() - video_clips.min())
        keyframes = (keyframes - keyframes.min()) / (keyframes.max() - keyframes.min())
        
        video_clips_batch.append(video_clips)
        keyframes_batch.append(keyframes)
    
    return video_clips_batch, keyframes_batch


def load_real_image_as_keyframes(image_path, num_clips=15, device="cuda"):
    """Load a real image and replicate as keyframes for testing"""
    from PIL import Image
    import torchvision.transforms as T
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).to(device)  # (3, 224, 224)
    
    # Replicate for all clips
    keyframes = img_tensor.unsqueeze(0).repeat(num_clips, 1, 1, 1)  # (num_clips, 3, 224, 224)
    
    # Create dummy video clips based on the image
    video_clips = keyframes.unsqueeze(2).repeat(1, 1, 16, 1, 1)  # (num_clips, 3, 16, 224, 224)
    
    return [video_clips], [keyframes]


def test_individual_components(config):
    """Test each component individually before full pipeline"""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Test SGGWrapper
    print("\n[1/5] Testing SGGWrapper...")
    from models.sgg_wrapper import SGGWrapper
    sgg = SGGWrapper(config["sgg_config"])
    
    # Create test image
    test_img = torch.rand(3, 224, 224).to(device)
    scene_graph = sgg(test_img)
    print(f"   -> SGGWrapper output: {len(scene_graph)} triples")
    if len(scene_graph) > 0:
        print(f"   -> First triple keys: {scene_graph[0].keys()}")
        print(f"   -> Subject feature shape: {scene_graph[0]['subject']['feature'].shape}")
    
    # 2. Test TemporalSGEncoder
    print("\n[2/5] Testing TemporalSGEncoder...")
    from models.temporal_encoder import TemporalSGEncoder
    temporal_encoder = TemporalSGEncoder(config).to(device)
    
    # Create fake scene graphs batch (B, num_clips, triples)
    num_clips = config["temporal_encoder_config"]["num_clips"]
    scene_graphs_batch = [[scene_graph for _ in range(num_clips)]]  # (B=1, num_clips)
    
    temp_emb = temporal_encoder(scene_graphs_batch)
    print(f"   -> TemporalSGEncoder output shape: {temp_emb.shape}")
    
    # 3. Test QFormer
    print("\n[3/5] Testing QFormer...")
    from models.qformer import QFormer
    qformer = QFormer(config).to(device)
    
    visual_prompts = qformer(temp_emb)
    print(f"   -> QFormer output shape: {visual_prompts.shape}")
    
    # 4. Test CaptionHead
    print("\n[4/5] Testing CaptionHead...")
    from models.decoder import CaptionHead
    
    # Use DeepSeek-VL2-tiny from config
    test_decoder_config = config["decoder_config"].copy()
    
    caption_head = CaptionHead(test_decoder_config)
    
    # Test training mode
    dummy_caption = ["A man is walking on the street."]
    try:
        loss, logits, _ = caption_head(visual_prompts, truth_caption=dummy_caption, mode='training')
        print(f"   -> Training loss: {loss.item():.4f}")
        print(f"   -> Logits shape: {logits.shape}")
    except Exception as e:
        print(f"   -> Training mode error: {e}")
    
    # Test inference mode
    try:
        generated = caption_head(visual_prompts, mode='inference')
        print(f"   -> Generated tokens shape: {generated.shape if hasattr(generated, 'shape') else type(generated)}")
    except Exception as e:
        print(f"   -> Inference mode error: {e}")
    
    # 5. Test FeatureExtractor (separate because it's heavy)
    print("\n[5/5] Testing FeatureExtractor...")
    from models.feature_extractor import FeatureExtractor
    try:
        feature_extractor = FeatureExtractor(config)
        feature_extractor.to(device)
        
        clip = torch.rand(3, 16, 224, 224).to(device)
        keyframe = torch.rand(3, 224, 224).to(device)
        
        motion_feats, enhanced_feats, boxes = feature_extractor(clip, keyframe)
        print(f"   -> Motion features shape: {motion_feats.shape}")
        print(f"   -> Enhanced features shape: {enhanced_feats.shape}")
        print(f"   -> Detected boxes: {boxes.shape if hasattr(boxes, 'shape') else len(boxes)}")
    except Exception as e:
        print(f"   -> FeatureExtractor error: {e}")
    
    print("\n" + "="*60)
    print("INDIVIDUAL COMPONENT TESTS COMPLETED")
    print("="*60)


def test_full_pipeline(config, use_real_image=False, image_path=None):
    """Test the complete SGGClassCap pipeline"""
    print("\n" + "="*60)
    print("TESTING FULL SGGCLASSCAP PIPELINE")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Use DeepSeek-VL2-tiny from config
    test_config = config.copy()
    
    print("\n[1] Loading SGGClassCap model...")
    from models.model import SGGClassCap
    model = SGGClassCap(test_config)
    model.to(device)
    print("   -> Model loaded successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   -> Total parameters: {total_params:,}")
    print(f"   -> Trainable parameters: {trainable_params:,}")
    
    print("\n[2] Preparing input data...")
    if use_real_image and image_path:
        video_clips_batch, keyframes_batch = load_real_image_as_keyframes(image_path, device=device)
        print(f"   -> Loaded real image from {image_path}")
    else:
        video_clips_batch, keyframes_batch = create_dummy_data(batch_size=1, device=device)
        print("   -> Created dummy data")
    
    print(f"   -> Video clips shape: {video_clips_batch[0].shape}")
    print(f"   -> Keyframes shape: {keyframes_batch[0].shape}")
    
    print("\n[3] Testing inference mode...")
    model.eval()
    try:
        with torch.no_grad():
            generated = model(video_clips_batch, keyframes_batch, mode='inference')
        
        # Decode generated tokens
        if hasattr(generated, 'shape'):
            print(f"   -> Generated tokens shape: {generated.shape}")
            tokenizer = model.caption_head.tokenizer
            captions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            print(f"   -> Generated caption: {captions}")
        else:
            print(f"   -> Generated output: {generated}")
        
        print("   -> Inference mode: SUCCESS ✓")
    except Exception as e:
        import traceback
        print(f"   -> Inference mode: FAILED ✗")
        print(f"   -> Error: {e}")
        traceback.print_exc()
    
    print("\n[4] Testing training mode...")
    model.train()
    try:
        dummy_captions = ["A person is walking in the park near trees."]
        loss, logits, temp_emb = model(video_clips_batch, keyframes_batch, 
                                        caption_tokens_batch=dummy_captions, 
                                        mode='training')
        
        print(f"   -> Loss: {loss.item():.4f}")
        print(f"   -> Logits shape: {logits.shape}")
        print(f"   -> Temporal embedding shape: {temp_emb.shape}")
        
        # Test backward pass
        loss.backward()
        print("   -> Backward pass: SUCCESS ✓")
        print("   -> Training mode: SUCCESS ✓")
    except Exception as e:
        import traceback
        print(f"   -> Training mode: FAILED ✗")
        print(f"   -> Error: {e}")
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("FULL PIPELINE TEST COMPLETED")
    print("="*60)


def test_memory_usage(config):
    """Test memory usage of the pipeline"""
    print("\n" + "="*60)
    print("TESTING MEMORY USAGE")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    torch.cuda.reset_peak_memory_stats()
    
    # Use DeepSeek-VL2-tiny from config
    test_config = config.copy()
    
    print("Loading model...")
    from models.model import SGGClassCap
    model = SGGClassCap(test_config)
    model.to("cuda")
    
    print(f"After model load: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("Creating data...")
    video_clips_batch, keyframes_batch = create_dummy_data(batch_size=1, device="cuda")
    
    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        _ = model(video_clips_batch, keyframes_batch, mode='inference')
    
    print(f"After forward pass: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("\nCleaning up...")
    del model
    torch.cuda.empty_cache()
    print(f"After cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test SGGClassCap pipeline")
    parser.add_argument("--mode", type=str, default="full", choices=["components", "full", "memory", "all"],
                       help="Test mode: components, full, memory, or all")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    args = parser.parse_args()
    
    print("Loading config...")
    config = load_config()
    
    if args.mode == "components" or args.mode == "all":
        test_individual_components(config)
    
    if args.mode == "full" or args.mode == "all":
        test_full_pipeline(config, 
                          use_real_image=args.image is not None, 
                          image_path=args.image)
    
    if args.mode == "memory" or args.mode == "all":
        test_memory_usage(config)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)
