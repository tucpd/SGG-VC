import os
import torch 
import pandas as pd
import matplotlib.pyplot as plt
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

def validate_epoch(model, val_loader, device):
    model.eval()
    all_gts = {}
    all_res = {}

    from utils.caption_utils import get_tokenizer
    tokenizer = get_tokenizer()

    with torch.no_grad():
        for batch_idx, (clips, keyframes, caption_lists) in enumerate(val_loader):
            
            # Chi dung inference mode - generate captions
            generated_ids = model(clips, keyframes, mode='inference')
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i, gen_cap in enumerate(generated_captions):
                vid = f'vid_{batch_idx * len(clips) + i}'
                
                # Dung TAT CA reference captions de tinh metrics
                refs = caption_lists[i] if isinstance(caption_lists[i], list) else [caption_lists[i]]
                ref_captions = [str(ref) for ref in refs]

                all_gts[vid] = ref_captions
                all_res[vid] = [gen_cap]

    # Tinh metrics voi all references (bo METEOR vi hay bi treo)
    scorers = {
        'BLEU-4': (Bleu(4), 'Bleu_4'),
        'CIDEr': (Cider(), 'CIDEr'),
        'ROUGE': (Rouge(), 'ROUGE_L')
    }

    metrics = {}
    for name, (scorer, key) in scorers.items():
        try:
            score, _ = scorer.compute_score(all_gts, all_res)
            if isinstance(score, list):
                score = score[-1]  # BLEU returns list [b1, b2, b3, b4]
            metrics[name.lower().replace('-', '')] = score
        except Exception as e:
            print(f"Warning: Failed to compute {name}: {e}")
            metrics[name.lower().replace('-', '')] = 0.0
    
    # Khong co val_loss, tra ve 0 de compatible voi train.py
    return 0.0, metrics

def plot_training_curves(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    
    # Training Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # Metrics curve
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['bleu4'], label='BLEU-4', marker='o')
    plt.plot(df['epoch'], df['cider'], label='CIDEr', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'metrics_curve.png'))
    plt.close()


def validate_epoch_cached(model, val_loader, device):
    """
    Validation function for cached feature training.
    Similar to validate_epoch but uses scene_graphs instead of video clips.
    """
    model.eval()
    all_gts = {}
    all_res = {}

    from utils.caption_utils import get_tokenizer
    tokenizer = get_tokenizer()

    with torch.no_grad():
        for batch_idx, (scene_graphs, motion_feats, caption_lists) in enumerate(val_loader):
            
            # Generate captions using cached features
            generated_ids = model(scene_graphs, motion_feats, mode='inference')
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i, gen_cap in enumerate(generated_captions):
                vid = f'vid_{batch_idx * len(scene_graphs) + i}'
                
                # Use ALL reference captions for metrics
                refs = caption_lists[i] if isinstance(caption_lists[i], list) else [caption_lists[i]]
                ref_captions = [str(ref) for ref in refs]

                all_gts[vid] = ref_captions
                all_res[vid] = [gen_cap]

    # Compute metrics
    scorers = {
        'BLEU-4': (Bleu(4), 'Bleu_4'),
        'CIDEr': (Cider(), 'CIDEr'),
        'ROUGE': (Rouge(), 'ROUGE_L')
    }

    metrics = {}
    for name, (scorer, key) in scorers.items():
        try:
            score, _ = scorer.compute_score(all_gts, all_res)
            if isinstance(score, list):
                score = score[-1]  # BLEU returns list [b1, b2, b3, b4]
            metrics[name.lower().replace('-', '')] = score
        except Exception as e:
            print(f"Warning: Failed to compute {name}: {e}")
            metrics[name.lower().replace('-', '')] = 0.0
    
    return 0.0, metrics