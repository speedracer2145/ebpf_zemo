import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, struct_dim=768, sem_dim=384, hidden_dim=512, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        
        # Projection layers to map both streams to a shared hidden space
        self.proj_struct = nn.Linear(struct_dim, hidden_dim)
        self.proj_sem = nn.Linear(sem_dim, hidden_dim)
        
        # Multi-head attention layer
        # Structural embedding queries the semantic embedding
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Layer norm and Dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, e_struct, e_sem):
        """
        Inputs:
            e_struct: [batch, 768]
            e_sem:    [batch, 384]
        Outputs:
            e_fused:  [batch, 512]
        """
        # 1. Project to shared space [batch, 512]
        x_struct = self.proj_struct(e_struct)
        x_sem = self.proj_sem(e_sem)
        
        # Add a sequence dimension for MHA if not present [batch, 1, 512]
        if len(x_struct.shape) == 2:
            x_struct = x_struct.unsqueeze(1)
        if len(x_sem.shape) == 2:
            x_sem = x_sem.unsqueeze(1)
            
        # 2. Cross-Attention
        # Q = structural, K=V = semantic
        attn_output, _ = self.attn(query=x_struct, key=x_sem, value=x_sem)
        
        # 3. Residual connection and normalization
        x = x_struct + self.dropout(attn_output)
        x = self.norm(x)
        
        # Return reshaped output [batch, 512]
        return x.squeeze(1)

if __name__ == "__main__":
    # Test fusion layer forward pass
    fusion = CrossAttentionFusion()
    
    # Mock embeddings
    e_struct = torch.randn(2, 768)
    e_sem = torch.randn(2, 384)
    
    e_fused = fusion(e_struct, e_sem)
    print(f"Fused embedding shape: {e_fused.shape}")
    print(f"Fused embedding snippet: {e_fused[0][:5]}")
