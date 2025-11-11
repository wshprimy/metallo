"""
Reusable attention modules for multi-modal fusion.
Leverages proven attention mechanisms from transformers library.
"""

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention using BERT's self-attention mechanism.
    Allows one modality to attend to another using bidirectional attention.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Create a minimal config for BertSelfAttention
        class AttentionConfig:
            def __init__(self):
                self.hidden_size = hidden_dim
                self.num_attention_heads = num_heads
                self.attention_probs_dropout_prob = dropout
                self.position_embedding_type = "absolute"
                self.is_decoder = False
                self.hidden_dropout_prob = dropout
        
        config = AttentionConfig()
        self.attention = BertSelfAttention(config)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, query, key_value):
        """
        Args:
            query: [B, N_q, D] - query features
            key_value: [B, N_kv, D] - key and value features
        Returns:
            output: [B, N_q, D] - attended features
        """
        B, N_q, D = query.shape
        N_kv = key_value.shape[1]
        
        # Concatenate query and key_value for bidirectional attention
        # BertSelfAttention performs self-attention on the combined sequence
        combined = torch.cat([query, key_value], dim=1)  # [B, N_q + N_kv, D]
        
        # Create attention mask to allow query positions to attend to key_value positions
        # Mask shape: [B, 1, N_q + N_kv, N_q + N_kv]
        # We want: query attends to key_value, key_value attends to itself
        attention_mask = torch.zeros(B, 1, N_q + N_kv, N_q + N_kv, device=query.device)
        
        # Query positions (0:N_q) can attend to key_value positions (N_q:)
        attention_mask[:, :, :N_q, N_q:] = 0  # Allow query -> key_value
        attention_mask[:, :, :N_q, :N_q] = -10000  # Block query -> query
        
        # Key_value positions can attend to themselves
        attention_mask[:, :, N_q:, N_q:] = 0  # Allow key_value -> key_value
        attention_mask[:, :, N_q:, :N_q] = -10000  # Block key_value -> query
        
        # Apply BertSelfAttention with the mask
        attended = self.attention(combined, attention_mask=attention_mask)[0]  # [B, N_q + N_kv, D]
        
        # Extract only the query part (first N_q positions)
        query_attended = attended[:, :N_q, :]  # [B, N_q, D]
        
        # Residual connection + layer norm
        output = self.layer_norm(query + self.dropout(query_attended))
        return output


class SpatialCrossAttention(nn.Module):
    """
    Cross-attention for spatial feature maps.
    Projects spatial features to sequences and applies cross-attention.
    """
    
    def __init__(self, query_channels, kv_channels, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project channels to common hidden dimension
        self.query_proj = nn.Conv2d(query_channels, hidden_dim, kernel_size=1)
        self.kv_proj = nn.Conv2d(kv_channels, hidden_dim, kernel_size=1)
        
        # Cross-attention module
        self.cross_attention = CrossModalAttention(hidden_dim, num_heads, dropout)
        
    def forward(self, query_feat, kv_feat):
        """
        Args:
            query_feat: [B, C_q, H_q, W_q] - query feature map
            kv_feat: [B, C_kv, H_kv, W_kv] - key/value feature map
        Returns:
            output: [B, hidden_dim, H_q, W_q] - attended feature map
        """
        B = query_feat.size(0)
        
        # Project to common dimension
        query_proj = self.query_proj(query_feat)  # [B, D, H_q, W_q]
        kv_proj = self.kv_proj(kv_feat)  # [B, D, H_kv, W_kv]
        
        # Flatten spatial dimensions
        H_q, W_q = query_proj.shape[2:]
        H_kv, W_kv = kv_proj.shape[2:]
        
        query_flat = query_proj.flatten(2).transpose(1, 2)  # [B, H_q*W_q, D]
        kv_flat = kv_proj.flatten(2).transpose(1, 2)  # [B, H_kv*W_kv, D]
        
        # Apply cross-attention
        attended = self.cross_attention(query_flat, kv_flat)  # [B, H_q*W_q, D]
        
        # Reshape back to spatial format
        output = attended.transpose(1, 2).reshape(B, self.hidden_dim, H_q, W_q)
        
        return output


class DynamicGatingNetwork(nn.Module):
    """
    Learns dynamic weights for multi-modal fusion.
    Uses MLP to predict importance weights that sum to 1.
    """
    
    def __init__(self, input_dim, num_modalities=2, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.num_modalities = num_modalities
        
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modalities),
        )
        
    def forward(self, *features):
        """
        Args:
            *features: Variable number of feature tensors [B, D]
        Returns:
            weights: [B, num_modalities] - normalized weights summing to 1
        """
        # Concatenate all features
        combined = torch.cat(features, dim=1)  # [B, num_modalities * D]
        
        # Compute gate logits
        logits = self.gate_network(combined)  # [B, num_modalities]
        
        # Softmax to get normalized weights
        weights = torch.softmax(logits, dim=1)  # [B, num_modalities]
        
        return weights