from torch.utils.data import Dataset
import torch
import os
import random
import json
from datasets import load_dataset, Features, Value

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话前处理: 以一定概率随机插入 system 消息

    特点:
    - 只有当首条消息不是 system 角色时才可能插入
    - add_system_ratio 控制插入概率 (默认 20%), 引入随机性可提升模型
      对有/无 system prompt 两种情况的泛化能力
    - system 内容从预定义的中英文 prompt 池中随机抽取, 覆盖不同表达风格
    - 若样本包含 tools 字段, 说明是 tool-use/function-calling 数据,
      此时必须保持原始对话结构, 不随机插入 system, 避免破坏工具调用模板
    """
    # tool-use 数据完整保留, 不做随机 system 注入
    if any(conv.get("tools") for conv in conversations):
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]

    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + conversations

    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """
    对话后处理: 清理模板渲染后多余的空白块。

    特点:
    - 针对带 CoT(chain-of-thought) 格式的模型, apply_chat_template 有时会
      渲染出多余空行。
    - 大部分情况下 (概率 1 - empty_think_ratio = 80%) 删除该空白块,
      防止模型学到无意义的空白格式。
    - 保留少量空白块 (empty_think_ratio = 20%), 让模型也能处理该边界情况。
    """
    if "\n\n\n\n" in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace("\n\n\n\n", "")

    return prompt_content


# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset —— 自回归预训练数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标: Next-Token Prediction（下一个 token 预测）
# 数据格式: {"text": "一段原始文本"}
# 训练特点:
#   - 模型对整段文本的每个位置都进行预测, 没有"只学回复"的区分
#   - 使用 BOS/EOS 标记文本边界, 让模型学会文本的起止
#   - PAD token 对应的 label 置 -100, 不参与 loss 计算, 节省无效梯度
#   - labels 直接 clone 自 input_ids（即 X 和 Y 错位一格: Y[t] = X[t+1]）
#   - 官方训练脚本中 PretrainDataset 返回二元组: (input_ids, labels)
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1: tokenize 原始文本, 留出首尾各 1 个 token 的位置给 BOS/EOS
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,  # 预留 BOS + EOS 的位置
            truncation=True,
        ).input_ids

        # Step 2: 拼接 BOS + token 序列 + EOS, 构成完整序列
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Step 3: 右侧用 PAD 补齐到 max_length, 保证 batch 内等长
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Step 4: labels 与 input_ids 完全相同, 但 PAD 位置置 -100
        # CrossEntropyLoss 会自动忽略 -100, 不计入 loss
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # 官方版本返回二元组, 不返回 attention_mask
        return input_ids, labels


# ──────────────────────────────────────────────────────────────────────────────
# 2. SFTDataset —— 有监督微调 (Supervised Fine-Tuning) 数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标: 让模型学会"只预测 assistant 回复", 忽略 user/system 输入
# 数据格式: {"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}]}
# 训练特点:
#   - 通过 generate_labels 扫描 bos_id（assistant 回复起始标记）定位每段回复,
#     仅将 assistant 回复的 token 位置设为有效 label, 其余全部为 -100
#   - 这样做的意义: 让 loss 只反映模型对"正确回答"的拟合, 不浪费梯度在
#     用户输入的复现上（用户输入只作为 context, 不是预测目标）
#   - 支持 tool-use/function-calling:
#     若 system 消息携带 "tools" 字段, 会解析并透传给 apply_chat_template
#     若 assistant 消息携带 "tool_calls" 字段, 会解析为 Python 对象
#   - 与 PretrainDataset 的关键区别: 标签是"稀疏"的, 只有 assistant 部分非 -100
#   - 官方训练脚本中 SFTDataset 返回二元组: (input_ids, labels)
# ──────────────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 显式声明 features, 兼容 sft_t2t_mini.jsonl 中混合出现的普通对话数据和 tool-use 数据
        # 若不声明 features, datasets 会根据前几条样本自动推断 schema
        # 当后续样本多出 tools/tool_calls 字段时, 会出现 schema cast 错误
        features = Features({
            "conversations": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                    "reasoning_content": Value("string"),
                    "tools": Value("string"),
                    "tool_calls": Value("string"),
                }
            ]
        })

        self.samples = load_dataset(
            "json",
            data_files=jsonl_path,
            split="train",
            features=features,
        )

        # 预先 tokenize assistant 回复的起始标记 (BOS + "assistant\n")
        # 用于在 generate_labels 中定位每段 assistant 回复的开始位置
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n",
            add_special_tokens=False,
        ).input_ids

        # 预先 tokenize assistant 回复的结束标记 (EOS + "\n")
        # 用于在 generate_labels 中定位每段 assistant 回复的结束位置
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n",
            add_special_tokens=False,
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        将多轮对话转换为模型输入的字符串

        特点:
        - 复制原始 conversations, 防止修改原始数据
        - 检测 system 消息中是否携带 tools 字段 (tool-use/function-calling 场景)
          若有则 json.loads 后透传给 apply_chat_template, 生成标准 tool-use 格式的提示词
        - 检测 assistant 消息中是否携带 tool_calls 字段
          若有则 json.loads 后保留在 message 中, 让 tokenizer 的 chat template 正确渲染
        - add_generation_prompt=False: 不在末尾追加"请模型续写"的 prompt,
          因为训练时需要完整的 input+output 序列, 而非开放续写
        """
        messages = []
        tools = None

        for message in conversations:
            # datasets 返回的样本可能不是普通 dict, 这里转成 dict 便于修改
            message = dict(message)

            # system 消息中的 tools 是工具定义, 通常是 JSON 字符串
            if message.get("role") == "system" and message.get("tools"):
                tools = (
                    json.loads(message["tools"])
                    if isinstance(message["tools"], str)
                    else message["tools"]
                )

            # assistant 消息中的 tool_calls 是工具调用结果, 通常也是 JSON 字符串
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])

            messages.append(message)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def generate_labels(self, input_ids):
        """
        生成 SFT 训练所需的稀疏标签序列

        算法逻辑 (滑动窗口扫描):
        1. 初始化全 -100 的 labels, 默认所有位置不计算 loss
        2. 逐位扫描 input_ids, 检测是否匹配 bos_id (assistant 回复起始)
        3. 匹配到 bos_id 后, 向后扫描直到找到 eos_id (回复结束)
        4. 将 [start, end + len(eos_id)) 区间内的 label 设为对应的 input_ids 值,
           即将这段 assistant 回复参与 loss 计算
        5. EOS token 本身也计入 label, 让模型学会何时停止生成
        6. 跳过已处理区间, 继续扫描下一段 assistant 回复 (支持多轮对话)
        """
        labels = [-100] * len(input_ids)
        i = 0

        while i < len(input_ids):
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                # 跳过 bos_id 本身, 从 assistant 实际内容开始
                start = i + len(self.bos_id)
                end = start

                # 向后扫描, 找到 eos_id 的位置
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                # 将 assistant 回复 (含 EOS) 区间的 label 设为真实 token_id
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]

                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1

        return labels

    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1: 随机决定是否插入 system prompt (数据增强)
        # 若样本包含 tools, pre_processing_chat 会直接返回原始 conversations
        conversations = pre_processing_chat(sample["conversations"])

        # Step 2: 用 chat template 渲染完整对话字符串
        prompt = self.create_chat_prompt(conversations)

        # Step 3: 清理可能出现的空白块
        prompt = post_processing_chat(prompt)

        # Step 4: tokenize 并截断到 max_length, 不足则右侧 PAD 补齐
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (
            self.max_length - len(input_ids)
        )

        # Step 5: 生成稀疏标签, 只有 assistant 回复部分有效 label
        labels = self.generate_labels(input_ids)

        # 官方版本返回二元组, 不返回 attention_mask
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. DPODataset —— 直接偏好优化（Direct Preference Optimization）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标: 让模型学会"偏好好回答, 远离坏回答"，使输出更符合人类偏好
# 数据格式: {"chosen": [{role, content}...], "rejected": [{role, content}...]}
#   - chosen: 人类标注的更优回答对话
#   - rejected: 人类标注的较差回答对话
# 训练特点:
#   - 每条样本同时返回 chosen 和 rejected 两份 tokenized 序列,
#     训练时 DPO loss 会最大化 chosen 回复的对数似然、最小化 rejected 的
#   - loss_mask 的设计与 SFT 一致: 只有 assistant 回复部分为 1,
#     其余为 0, 保证对比信号仅来自模型的实际输出部分
#   - 采用"错位"方式构造输入输出对: x 取 [:-1], y 取 [1:],
#     即 x[t] 预测 y[t] = input[t+1], 标准自回归格式
#   - mask 同样错位取 [1:], 与 y 对齐, 方便在训练时直接做 masked loss
#   - max_length 默认 4096, 比 SFT 更长, 因为 DPO 数据通常包含完整对话上下文
# ──────────────────────────────────────────────────────────────────────────────
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # pad_token_id 若不存在则回退到 0, 保证补齐操作不会崩溃
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

        # 与 SFTDataset 相同: 预先 tokenize assistant 回复的起止标记,
        # 用于 generate_loss_mask 中精确定位 assistant 回复区间
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n",
            add_special_tokens=False,
        ).input_ids

        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n",
            add_special_tokens=False,
        ).input_ids

        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]      # 优质回答对话列表, 格式: [{role, content}, ...]
        rejected = sample["rejected"]  # 劣质回答对话列表, 格式同上

        # Step 1: 将 chosen / rejected 对话分别渲染为字符串
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen,
            tokenize=False,
            add_generation_prompt=False,
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected,
            tokenize=False,
            add_generation_prompt=False,
        )
        rejected_prompt = post_processing_chat(rejected_prompt)

        # Step 2: tokenize 并 padding 到 max_length(统一序列长度, 方便 batch)
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_encoding["input_ids"]

        # Step 3: 生成 loss mask, 只有 assistant 回复部分为 1
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # Step 4: 构造自回归训练对, x=[:-1] 作为输入, y=[1:] 作为目标
        #         mask[1:] 与 y 对齐, 决定哪些位置的 loss 计入梯度
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        # 官方版本不返回 attention_mask, 只返回 DPO 训练所需的输入、目标和 loss mask
        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
        }

    def generate_loss_mask(self, input_ids):
        """
        生成 DPO 训练所需的 loss mask (0/1 二值序列)

        与 SFTDataset.generate_labels 逻辑完全相同, 区别在于:
        - SFT 返回的是具体 token_id (用于 CE loss)
        - DPO 返回的是 0/1 掩码 (用于 masked 对数似然计算)
        算法: 扫描 bos_id -> 找到 eos_id -> 区间内置 1, 其余置 0
        """
        loss_mask = [0] * len(input_ids)
        i = 0

        while i < len(input_ids):
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start

                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                # 将 assistant 回复 (含 EOS) 区间的 mask 置 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1

                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1

        return loss_mask


# ──────────────────────────────────────────────────────────────────────────────
# 4. RLAIFDataset —— 基于 AI 反馈的强化学习数据集 (用于 PPO / GRPO)
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标: 为 RL 训练提供"问题-参考答案"对, 由 actor 在线采样生成回复,
#           再由 reward model 或规则函数打分优化
# 数据格式: {"conversations": [{"role": "...", "content": "..."}, ...]}
# 训练特点 (与前三个 Dataset 的核心区别):
#   - 不做离线 tokenize: 只返回原始字符串 prompt 和 answer,
#     让 RL trainer (PPO/GRPO) 在线 rollout 时自行 tokenize,
#     因为 RL 需要动态生成回复并实时打分, 无法预先固定 token 序列
#   - create_chat_prompt 会剥离最后一条 assistant 消息,
#     将其余对话渲染为带 add_generation_prompt=True 的 prompt,
#     供 actor 模型续写
#   - thinking_ratio 控制是否开启 open_thinking,
#     让 RL 训练阶段可以混合思考/非思考格式
#   - 返回值是 dict{"prompt": str, "answer": str}, 而非 tensor,
#     这是 RL 数据集与 SL 数据集 (返回 tensor) 的最显著差异
# ──────────────────────────────────────────────────────────────────────────────
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio

        self.samples = load_dataset("json", data_files=jsonl_path, split="train")

        # 保留 bos_id / eos_id 以兼容未来可能的 mask 扩展
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant",
            add_special_tokens=False,
        ).input_ids

        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}",
            add_special_tokens=False,
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        从对话列表中构造 RL rollout 所需的 prompt

        处理逻辑:
        1. 先经过 pre_processing_chat, 与 SFT 阶段保持 system prompt 数据增强逻辑
        2. 随机决定是否开启 open_thinking
        3. 去掉最后一条消息, 只保留上下文
        4. add_generation_prompt=True, 在末尾追加 assistant 开始回复的引导标记
        """
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio

        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True,
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample["conversations"])

        return {
            "prompt": prompt,
            "answer": "",
        }


if __name__ == "__main__":
    pass