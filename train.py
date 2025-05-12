import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import re
from collections import defaultdict


# 读取文本文件
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# 清洗文本
def clean_text(text):
    # text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    return text.lower()  # 转换为小写


# 分词
def tokenize(text):
    return text.split()


# 建立词汇表
def build_vocab(text):
    vocab = defaultdict(int)
    tokens = tokenize(text)
    for token in tokens:
        vocab[token] += 1
    index_to_word = {i: word for i, word in enumerate(vocab.keys())}
    word_to_index = {word: i for i, word in index_to_word.items()}
    return word_to_index, index_to_word


# 数据集类
class TextDataset(Dataset):
    def __init__(self, encoded_text, seq_length):
        self.encoded_text = encoded_text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        x = self.encoded_text[idx:idx + self.seq_length]
        y = self.encoded_text[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x), torch.tensor(y)


# LSTM 模型
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out


# 超参数
embed_size = 128
hidden_size = 256
seq_length = 30  # 输入序列的长度
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# 读取并处理文本数据
file_path = 'I, Robot (Isaac Asimov) (Z-Library).txt'  # 替换为你的文件路径
raw_text = read_text_file(file_path)
cleaned_text = clean_text(raw_text)
word_to_index, index_to_word = build_vocab(cleaned_text)
encoded_text = [word_to_index[word] for word in cleaned_text.split()]

# 准备数据
dataset = TextDataset(encoded_text, seq_length)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 检查 CUDA 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和优化器
vocab_size = len(word_to_index)
model = TextGenerator(vocab_size, embed_size, hidden_size).to(device)  # 移动模型到 GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # 移动数据到 GPU

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 文本生成函数
def generate_text(model, start_text, gen_length, word_to_index, index_to_word):
    model.eval()
    generated = start_text
    start_text=clean_text(start_text)
    input_eval = []

    # 初始化输入序列
    for word in start_text.split():
        if word in word_to_index:
            input_eval.append(word_to_index[word])
        else:
            print(f"警告: '{word}' 不在词汇表中，已跳过。")

    # 检查是否有有效的输入
    if len(input_eval) < seq_length:
        raise ValueError("起始文本长度小于序列长度，请提供更长的文本。")

    # 截取最后 seq_length 个 token
    input_eval = input_eval[-seq_length:]
    input_eval = torch.tensor(input_eval).unsqueeze(0).to(device)  # 移动到 GPU

    for _ in range(gen_length):
        predictions = model(input_eval)
        mx=torch.argmax(predictions,-1)
        predicted_index = mx[-1][-1].item()

        # 检查预测的索引是否在词汇表中
        if predicted_index in index_to_word:
            predicted_word = index_to_word[predicted_index]
        else:
            print(f"警告: 预测的索引 {predicted_index} 不在词汇表中，使用 '<unk>' 替代。")
            predicted_word = '<unk>'  # 或者其他占位符

        generated += ' ' + predicted_word

        # 更新输入序列，保持长度为 seq_length
        input_eval = torch.cat((input_eval[:, 1:], torch.tensor([[predicted_index]]).to(device)), dim=1)

    return generated

# torch.save(model.state_dict(), 'model.pth')
# 示例生成文本
model.load_state_dict(torch.load('model.pth'))
start_text = "A dawn broke over the African plains, where a tribe of hominids struggled for survival. The sun rose, casting long shadows as they scavenged for food. Among them, a curious figure named Moonwatcher gazed at the vast sky, feeling an inexplicable longing. One fateful night, a strange object descended from the heavens, a smooth black monolith. It stood silent and imposing, radiating an aura of mystery. As Moonwatcher approached, a surge of intelligence flooded his mind, igniting a spark of evolution that would change the course of humanity forever. The stars beckoned, and a new chapter was about to begin."
generated_text = generate_text(model, start_text, gen_length=500, word_to_index=word_to_index,
                               index_to_word=index_to_word)
print("生成的文本:", generated_text)