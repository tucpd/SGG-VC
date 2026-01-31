import torch
import torch.nn as nn
import math

class TemporalSGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["temporal_encoder_config"]["embed_dim"]
        self.num_clips = config["temporal_encoder_config"]["num_clips"]

        # Node/Edge embedding layers
        self.node_embed = nn.Sequential(
            nn.Linear(512, self.embed_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim)
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(512, self.embed_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim)
        )
        
        # Class and positional embeddings
        self.class_embed = nn.Embedding(200, 128)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, 128))

        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config["temporal_encoder_config"]["num_heads"],
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["temporal_encoder_config"]["num_layers"]
        )
        
        # Graph pooling
        self.graph_pool = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim)
        )
        
        # Change Detection
        self.change_detector = nn.Sequential(
            nn.Linear(2 * self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, scene_graphs_batch):
        """
        scene_graphs_batch: list[B] of list[num_clips] of scene graph triples
        Each triple: {'subject': {...}, 'predicate': {...}, 'object': {...}}
        Returns: (B, num_clips, embed_dim) temporal embeddings
        """
        batch_size = len(scene_graphs_batch)
        embedded_graphs_batch = []
        
        for b_idx in range(batch_size):
            scene_graphs = scene_graphs_batch[b_idx]
            embedded_graphs = []
            
            for clip_idx, graph_triples in enumerate(scene_graphs):
                if len(graph_triples) == 0:
                    graph_emb = torch.zeros(1, self.embed_dim).to(self.get_device())
                else:
                    # Extract and embed nodes
                    node_embeddings = []
                    edge_embeddings = []
                    
                    for triple in graph_triples[:20]:
                        subj_feat = triple['subject']['feature']
                        obj_feat = triple['object']['feature']
                        pred_emb = triple['predicate']['embedding']
                        
                        # Move to device (important for cached features loaded from CPU)
                        device = self.get_device()
                        subj_feat = subj_feat.to(device)
                        obj_feat = obj_feat.to(device)
                        pred_emb = pred_emb.to(device)
                        
                        if subj_feat.dim() == 1:
                            subj_feat = subj_feat.unsqueeze(0)
                        if obj_feat.dim() == 1:
                            obj_feat = obj_feat.unsqueeze(0)
                        if pred_emb.dim() == 1:
                            pred_emb = pred_emb.unsqueeze(0)
                        
                        subj_node_emb = self.node_embed(subj_feat)
                        obj_node_emb = self.node_embed(obj_feat)
                        edge_emb = self.edge_embed(pred_emb)
                        
                        node_embeddings.append(subj_node_emb)
                        node_embeddings.append(obj_node_emb)
                        edge_embeddings.append(edge_emb)
                    
                    if len(node_embeddings) > 0:
                        all_embeddings = torch.cat(node_embeddings + edge_embeddings, dim=0)
                        graph_emb = self.graph_pool(all_embeddings.mean(dim=0, keepdim=True))
                    else:
                        graph_emb = torch.zeros(1, self.embed_dim).to(self.get_device())
                
                embedded_graphs.append(graph_emb.squeeze(0))
            
            # Pad to num_clips if needed
            while len(embedded_graphs) < self.num_clips:
                embedded_graphs.append(torch.zeros(self.embed_dim).to(self.get_device()))
            
            embedded_graphs = torch.stack(embedded_graphs[:self.num_clips])
            embedded_graphs_batch.append(embedded_graphs)
        
        # Stack batch: (B, num_clips, embed_dim)
        temp_seq = torch.stack(embedded_graphs_batch, dim=0)
        
        # Temporal attention across clips
        temp_emb = self.temporal_encoder(temp_seq)
        
        # Change detection between consecutive clips
        changes = []
        for i in range(self.num_clips - 1):
            change_input = torch.cat([temp_seq[:, i], temp_seq[:, i+1]], dim=-1)
            change_score = self.change_detector(change_input)
            changes.append(change_score)
        
        return temp_emb
    
    def get_device(self):
        return next(self.parameters()).device