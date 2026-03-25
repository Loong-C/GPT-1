import os
import ftfy
import spacy
from tqdm import tqdm

# 配置路径
INPUT_DIR = "data/books1/epubtxt"
OUTPUT_FILE = "data/processed/all_books.txt"
LOG_INTERVAL = 100  # 每处理100本书打印一次进度

# 加载 spacy 仅用于基础的分句/清洗参考（论文中提到了 spacy [cite: 188]）
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def clean_text(text):
    """
    根据 GPT-1 论文要求进行文本清洗 [cite: 188]
    """
    # 1. 使用 ftfy 修复编码、标准化标点和空格 [cite: 188]
    text = ftfy.fix_text(text)
    
    # 2. 移除多余的空白字符
    text = " ".join(text.split())
    
    return text

def preprocess_books():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 获取所有 txt 文件
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    print(f"找到 {len(files)} 个待处理文件。")

    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for file_name in tqdm(files, desc="Processing Books"):
            file_path = os.path.join(INPUT_DIR, file_name)
            
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                    content = infile.read()
                    
                    if not content.strip():
                        continue
                        
                    # 执行清洗
                    cleaned_content = clean_text(content)
                    
                    # 写入文件，每本书后面加一个换行符，保持书籍间的界限
                    outfile.write(cleaned_content + "\n")
                    count += 1
            except Exception as e:
                print(f"跳过文件 {file_name}: {e}")

    print(f"预处理完成！共处理 {count} 本书。")
    print(f"最终合并文件保存在: {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_books()