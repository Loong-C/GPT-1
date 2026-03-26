import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.pwff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT-1 使用 GELU
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        # 残差连接 + LayerNorm
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.pwff(self.norm2(x)))
        return x

class GPT1(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=768, n_head=12, n_layers=12):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_model * 4) 
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 参数初始化（论文提到使用 N(0, 0.02)）
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # 1. Embedding 层
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        
        # 2. 生成因果掩码 (下三角矩阵)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).view(1, 1, seq_len, seq_len)
        
        # 3. 经过 12 层 Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # 4. 输出头
        logits = self.head(self.ln_f(x))
        return logits