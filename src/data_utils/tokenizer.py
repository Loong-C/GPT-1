import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_gpt1_tokenizer(corpus_path, output_dir="model_save/tokenizer"):
    """
    按照 GPT-1 论文参数训练 BPE 分词器
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 初始化 BPE 模型
    # GPT-1 特有的特殊标记：<unk> 占位，之后微调会用到 [START], [DELIM], [EXTRACT]
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    # 2. 配置训练器
    # vocab_size = 40,000 merges + 基础字符
    trainer = BpeTrainer(
        vocab_size=40000, 
        show_progress=True,
        special_tokens=["<unk>", "<pad>", "<end>"] 
    )

    # 3. 开始训练 (这可能需要一些时间，取决于你的 CPU)
    print("正在训练 BPE 分词器，请稍候...")
    tokenizer.train(files=[corpus_path], trainer=trainer)

    # 4. 保存模型
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    print(f"分词器训练完成并保存至: {output_dir}")

if __name__ == "__main__":
    CORPUS_FILE = "data/processed/all_books.txt"
    train_gpt1_tokenizer(CORPUS_FILE)