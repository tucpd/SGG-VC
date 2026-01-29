import torch
import torch.nn as nn

class TemporalSGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["temporal_encoder_config"]["embed_dim"]
        self.num_clips = config["temporal_encoder_config"]["num_clips"]

        # Node/Edge embedding
        self.node_embed = nn.Linear(512 + 300 + 128, self.embed_dim) # Visual + Class embed + Positional encoding
        self.edge_embed = nn.Linear(self.embed_dim * 3, self.embed_dim) # Subject + Predicate + Object

        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config["temporal_encoder_config"]["num_heads"],
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["temporal_encoder_config"]["num_layers"]
        )
        # Change Detection
        self.change_detector = nn.Linear(2 * self.embed_dim, 1)  # Compare consecutive graphs

    def forward(self, scene_graphs):
        # scene_graphs: List of 15 graphs, each is list of triples (subj_feat, pred_emb, obj_feat)
        embedded_graphs = []
        for graph in scene_graphs:
            nodes = [self.node_embed(torch.cat([subj_feat, subj_class_emb, pos_enc])) for subj_feat, _, obj_feat in graph]
            edges = [self.edge_embed(torch.cat([subj_emb, pred_emb, obj_emb])) for subj_emb, pred_emb, obj_emb in nodes]  # Simplified
            graph_emb = self.graph_pool(torch.stack(nodes + edges)) # Pool to vector
            embedded_graphs.append(graph_emb)
        
        # Temporal attention on 15 embeddings
        temp_seq = torch.stack(embedded_graphs, dim=1) # (B, 15, embed_dim)
        temp_emb = self.temporal_encoder(temp_seq) # (B, 15, embed_dim)
        
        # Change detection
        changes = [self.change_detector(torch.cat([temp_seq[:, i], temp_seq[:, i+1]], dim=-1)).sigmoid() for i in range(self.num_clips-1)]
        
        return temp_emb.mean(dim=1)  # Unified embedding
        
        