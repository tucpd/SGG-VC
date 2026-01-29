import torch
import torch.nn as nn
import torch.nn.functional as F

# Hàm loss Euclidean
def euclidean_loss(pred, target):
    return torch.norm(pred - target, p=2, dim=-1).mean()

# Hàm loss Cosine
def cosine_loss(pred, target):
    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # Tính Cosine similarity
    return 1 - cos_sim.mean()  # Cosine loss, minimize similarity = maximize cosine distance

def caption_loss(logits, targets, ignore_index=-100, label_smoothing=0.1):
    """
    Standard cross-entropy with optional label smoothing.
    logits: (B, seq_len, vocab_size)
    targets: (B, seq_len)
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing
    )

# Auxiliary: Temporal consistency loss
def temporal_consistency_loss(temporal_emb_seq, temperature=0.07):
    """
    temporal_emb_seq: (B, num_clips, embed_dim) - sequence of clip embeddings
    Enforce similarity giữa consecutive clips (positive pairs), dissimilarity với non-consecutive.
    Dùng NT-Xent style contrastive loss đơn giản.
    """
    B, T, D = temporal_emb_seq.shape

    # Normalize
    temporal_emb_seq = F.normalize(temporal_emb_seq, dim=-1)

    # Shift để lấy consecutive
    pos_pairs = temporal_emb_seq[:, :-1] * temporal_emb_seq[:, 1:]  # (B, T-1, D)
    pos_sim = pos_pairs.sum(dim=-1) / temperature  # (B, T-1)

    all_sim = torch.matmul(temporal_emb_seq, temporal_emb_seq.transpose(1, 2)) / temperature  # (B, T, T)

    # Mask positive (consecutive + self)
    mask = torch.eye(T, device=temporal_emb_seq.device).bool()
    mask[:, 1:] = mask[:, 1:] | mask[:, :-1]  # consecutive
    mask = mask.unsqueeze(0).expand(B, -1, -1)

    neg_sim = all_sim.masked_fill(mask, float('-inf'))
    
    # Loss: -log( exp(pos) / (exp(pos) + sum(exp(neg))) )
    logsumexp = torch.logsumexp(all_sim, dim=-1)  # approx
    pos_loss = -pos_sim + logsumexp[:, :-1]
    
    return pos_loss.mean()

# Tổng loss trong training
def total_loss(caption_logits, caption_targets, temporal_emb_seq=None, lambda_temporal=0.1):
    """
    Tổng hợp loss:
    - caption_loss: chính
    - temporal_consistency_loss: auxiliary cho temporal encoder
    """
    ce_loss = caption_loss(caption_logits, caption_targets)
    
    temp_loss = 0.0
    if temporal_emb_seq is not None:
        temp_loss = temporal_consistency_loss(temporal_emb_seq)
    
    return ce_loss + lambda_temporal * temp_loss
