import os
import torch 
import pandas as pd
import matplotlib.pyplot as plt
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

def validate_epoch(model, val_loader, device):
    model.eval()
    total_val_loss = 0.0
    all_gts = {}
    all_res = {}

    with torch.no_grad():
        for batch_idx, (clips, keyframes, caption_lists) in enumerate(val_loader):
            
            ce_loss, logits, _ = model(clips, keyframes, caption_lists, mode='training')
            total_val_loss += ce_loss.item()

            generated_ids = model(clips, keyframes, mode='inference')
            
            from utils.caption_utils import get_tokenizer
            tokenizer = get_tokenizer()
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i, gen_cap in enumerate(generated_captions):
                vid = f'vid_{batch_idx * len(clips) + i}'
                
                refs = caption_lists[i] if isinstance(caption_lists[i], list) else [caption_lists[i]]
                ref_captions = [str(ref) for ref in refs]

                all_gts[vid] = ref_captions
                all_res[vid] = [gen_cap]

    avg_val_loss = total_val_loss / len(val_loader)

    # Tinh metrics
    scorers = {
        'BLEU-4': (Bleu(4), 'Bleu_4'),
        'CIDEr': (Cider(), 'CIDEr'),
        'METEOR': (Meteor(), 'METEOR'),
        'ROUGE': (Rouge(), 'ROUGE_L')
    }

    metrics = {}
    for name, (scorer, key) in scorers.items():
        score, _ = scorer.compute_score(all_gts, all_res)
        metrics[name.lower().replace('-', '')] = score if not isinstance(score, dict) else score[key]
    
    return avg_val_loss, metrics

def plot_training_curves(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    
    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # Metrics curve
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['bleu4'], label='BLEU-4')
    plt.plot(df['epoch'], df['cider'], label='CIDEr')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Captioning Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'metrics_curve.png'))
    plt.close()