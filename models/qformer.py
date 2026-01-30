import torch
import torch.nn as nn

class QFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_query_tokens = config["decoder_config"]["num_query_tokens"]
        hidden_size = config["decoder_config"]["hidden_size"]
        num_heads = config["decoder_config"]["num_heads"]

        self.query_tokens = nn.Parameter(torch.randn(1, self.num_query_tokens, hidden_size))
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(hidden_size, config["decoder_config"]["proj_out_dim"])  # Added for fusion

    def forward(self, temporal_scene_emb):
        B = temporal_scene_emb.size(0)    
        query_tokens = self.query_tokens.expand(B, -1, -1) # (B, num_query_tokens, hidden_size)
        attn_output, _ = self.attn(query_tokens, temporal_scene_emb, temporal_scene_emb)
        visual_prompts = self.proj(attn_output)

        return visual_prompts