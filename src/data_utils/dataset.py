import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class GPTDataset(Dataset):
    def __init__(self, txt_path, tokenizer_path, max_len=512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        
        # 读取原始文本并分词
        # 注意：6GB 文本直接 encode 会爆内存，这里采用逐行读取或分块处理
        with open(txt_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        print("正在进行分词编码（这一步可能较慢）...")
        # encode_batch 会比循环 encode 快得多
        tokens = self.tokenizer.encode(full_text).ids
        
        # 将长 token 序列切分成长度为 max_len 的块
        self.chunks = [
            tokens[i : i + max_len] 
            for i in range(0, len(tokens) - max_len, max_len)
        ]
        print(f"数据处理完成，共 {len(self.chunks)} 个样本。")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # GPT 的训练目标是：输入前 n 个词，预测第 n+1 个词
        chunk = self.chunks[idx]
        
        # x 是输入，y 是标签（x 向右平移一位）
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y