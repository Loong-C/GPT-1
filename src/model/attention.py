import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        
        # 定义 Q, K, V 的线性映射
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # 1. 线性变换并切分为多头
        q = self.w_q(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        # scores shape: [batch, head, seq, seq]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. 应用因果掩码 (Causal Mask)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 4. 合并多头
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.fc(context)