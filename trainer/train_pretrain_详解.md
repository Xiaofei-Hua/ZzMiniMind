# train_pretrain.py 全面详解

> 本文档逐行、逐函数讲解 `train_pretrain.py` 及其依赖的 `trainer_utils.py`，确保你能独立理解预训练脚本的每一行代码。
> 前置知识：已掌握 `lm_dataset.py` 的内容（PretrainDataset 的使用方式、tokenize/PAD/label 流程）。

---

## 一、文件总览：这个脚本做了什么？

`train_pretrain.py` 是 MiniMind 的**预训练启动脚本**，它完成了从"原始文本数据"到"模型学会语言规律"的全过程。

整体流程图：

```
命令行参数解析 (argparse)
        │
        ▼
初始化环境（分布式、随机种子）
        │
        ▼
配置模型参数 (ZzMindConfig) + 检查断点续训
        │
        ▼
设置混合精度 (autocast + GradScaler)
        │
        ▼
可选：初始化 WandB 实验跟踪
        │
        ▼
创建模型、数据集、优化器、DataLoader
        │
        ▼
循环训练 epoch → train_epoch()
  ├── 前向传播（模型内置 loss 计算）
  ├── 反向传播（混合精度梯度缩放）
  ├── 梯度累积 + 裁剪 + 参数更新
  ├── 日志记录
  └── 定期保存检查点
```

---

## 二、导入区（第 1-32 行）

### 2.1 包路径设置

```python
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```

**为什么要这样做？**
- 这个脚本需要 `from model.ZzModel import ...` 和 `from dataset.lm_dataset import ...` 这样的跨目录导入。
- `__file__` 是当前脚本的路径（如 `/path/to/MiniMind/trainer/train_pretrain.py`）
- `os.path.dirname(__file__)` → `/path/to/MiniMind/trainer/`
- `os.path.join(..., "..")` → `/path/to/MiniMind/trainer/..` 即 `/path/to/MiniMind/`
- `os.path.abspath(...)` → 解析为绝对路径 `/path/to/MiniMind`
- `sys.path.append(...)` → 把项目根目录加入 Python 搜索路径，这样 `from model.ZzModel import ...` 才能找到模块

| os.path 函数 | 作用 |
|-------------|------|
| `os.path.dirname(path)` | 获取路径的目录部分 |
| `os.path.join(a, b)` | 安全拼接路径（自动处理 `/`） |
| `os.path.abspath(path)` | 转为绝对路径 |

### 2.2 标准库导入

```python
import argparse    # 命令行参数解析
import time        # 计时（统计训练速度）
import warnings    # 警告控制
from contextlib import nullcontext  # 空上下文管理器
```

#### `argparse` — 命令行参数解析

用于让用户在命令行中覆盖默认参数，例如：
```bash
python train_pretrain.py --epochs 3 --batch_size 64 --learning_rate 1e-4
```

基本用法：
```python
parser = argparse.ArgumentParser()           # 创建解析器
parser.add_argument("--epochs", type=int, default=1)  # 定义参数
args = parser.parse_args()                    # 解析命令行
print(args.epochs)                            # 使用参数值
```

| add_argument 参数 | 含义 |
|------------------|------|
| `"--epochs"` | 参数名，命令行用 `--epochs 3` 传入 |
| `type=int` | 自动转为 int 类型 |
| `default=1` | 不传时使用的默认值 |
| `help="..."` | `--help` 时显示的说明文字 |
| `choices=[0, 1]` | 只允许 0 或 1 |
| `action="store_true"` | 布尔开关，出现即为 True（不需要传值） |

#### `time` — 计时

```python
start_time = time.time()          # 获取当前时间戳（秒，浮点数）
elapsed = time.time() - start_time # 计算经过的时间
```

#### `warnings` — 警告控制

```python
warnings.filterwarnings("ignore")  # 忽略所有警告
```
- 训练过程中 PyTorch 会输出大量 DeprecationWarning 等，影响日志阅读。
- `"ignore"` 表示全部忽略，保持输出清洁。

#### `nullcontext` — 空上下文管理器

```python
from contextlib import nullcontext

# 用法：
with nullcontext():    # 什么都不做
    do_something()

with some_real_ctx():  # 真正的上下文管理器
    do_something()
```

- 为什么需要它？后面会讲到混合精度的 `autocast`，它只在 GPU 上有效。在 CPU 上训练时，需要一个"什么都不做"的替代品，`nullcontext()` 就是这个角色。

### 2.3 PyTorch 核心导入

```python
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
```

#### `torch.distributed` — 分布式训练

| 函数 | 作用 |
|------|------|
| `dist.is_initialized()` | 检查分布式环境是否已初始化 |
| `dist.get_rank()` | 获取当前进程的全局编号（0, 1, 2...） |
| `dist.get_world_size()` | 获取总进程数（即 GPU 数量） |
| `dist.init_process_group(backend="nccl")` | 初始化分布式通信（NCCL 是 NVIDIA GPU 专用后端） |

**什么是分布式训练？**
- 当模型太大或数据太多，一张 GPU 放不下时，用多张 GPU 同时训练。
- 每张 GPU 运行一个独立的 Python 进程，各自处理不同的数据子集，然后通过 `dist` 通信同步梯度。
- `rank` 是进程的全局编号，`local_rank` 是单机内的 GPU 编号。

#### `optim` — 优化器

```python
optimizer = optim.AdamW(model.parameters(), lr=5e-4)
```

- `AdamW` 是 Adam 优化器的改进版本（解耦权重衰减），是目前训练 Transformer 的标准选择。
- `model.parameters()` 返回模型所有可训练参数。

#### `DistributedDataParallel (DDP)` — 数据并行

```python
model = DistributedDataParallel(model, device_ids=[local_rank])
```

- 把模型包一层，自动在每次 backward 后同步各 GPU 的梯度。
- 包了之后，需要用 `model.module.xxx` 访问原始模型的属性（因为模型被包在 `.module` 里了）。

#### `DataLoader` — 数据加载器

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
```

| 参数 | 含义 |
|------|------|
| `dataset` | 数据集对象（PretrainDataset 实例） |
| `batch_size=32` | 每次取 32 条数据组成一个 batch |
| `shuffle=True` | 每 epoch 打乱数据顺序 |
| `sampler=...` | 自定义采样器（与 shuffle 互斥） |
| `num_workers=1` | 用 1 个子进程预加载数据 |
| `pin_memory=True` | 把数据固定在内存中，加速 CPU→GPU 传输 |
| `batch_sampler=...` | 自定义 batch 采样器 |

**DataLoader 做了什么？**
```python
for input_ids, labels, attention_mask in loader:
    # loader 自动做：
    # 1. 调用 dataset.__getitem__(i) 取出多条数据
    # 2. 把每条数据的 Tensor 沿第 0 维堆叠成 batch tensor
    # input_ids: [batch_size, max_length]
    # labels:    [batch_size, max_length]
    # attention_mask: [batch_size, max_length]
```

#### `DistributedSampler` — 分布式采样器

```python
train_sampler = DistributedSampler(train_ds)
```

- 把数据集**均匀分配**给各 GPU。
- 例如 1000 条数据、2 张 GPU → GPU0 取前 500 条，GPU1 取后 500 条。
- **注意：** 使用 `DistributedSampler` 时，DataLoader 的 `shuffle` 必须设为 `False`（采样器自行控制顺序），并通过 `sampler.set_epoch(epoch)` 在每轮打乱。

### 2.4 项目模块导入

```python
from model.ZzModel import ZzMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler,
)
```

- `ZzMindConfig`：模型配置类（hidden_size、层数等参数的容器）
- `PretrainDataset`：预训练数据集类（已在 `lm_dataset_详解.md` 中详细讲解）
- 后面的工具函数将在讲解 `trainer_utils.py` 时逐一说明

---

## 三、train_epoch 函数（第 35-130 行）

这是**训练的核心循环**——一个 epoch 内的所有训练逻辑。

### 3.1 函数签名

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
```

| 参数 | 含义 |
|------|------|
| `epoch` | 当前 epoch 编号（从 0 开始） |
| `loader` | DataLoader 实例 |
| `iters` | 这个 epoch 的总 batch 数（用于计算进度和 ETA） |
| `start_step` | 断点续训时的起始步数（默认 0） |
| `wandb` | 实验跟踪对象（None 则不记录） |

### 3.2 数据遍历（第 39-46 行）

```python
for step, (input_ids, labels, attention_mask) in enumerate(
    loader, start=start_step + 1
):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)
    attention_mask = attention_mask.to(args.device)
```

**`enumerate(loader, start=start_step + 1)`：**
- `enumerate` 给每次迭代附加编号。
- `start` 参数指定起始编号。例如断点续训已训练 500 步，则 `start=501`，日志中显示的步数从 501 开始。

**`.to(args.device)`：**
- 把 Tensor 从 CPU 移到 GPU（如 `"cuda:0"`）。
- 模型在 GPU 上，数据也必须在 GPU 上才能计算。

### 3.3 学习率调度（第 48-51 行）

```python
lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

for param_group in optimizer.param_groups:
    param_group["lr"] = lr
```

**`get_lr` 的实现（来自 trainer_utils.py）：**

```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

这是一个**余弦退火（Cosine Annealing）**学习率调度：

```
lr
│\
│ \___________
│             \___________
│                         \
│                          \___
└──────────────────────────────→ step
```

- 训练初期：学习率从约 `0.55 * lr` 缓慢下降
- 训练中期：稳定下降
- 训练末期：降到约 `0.1 * lr`（极小值），让模型收敛

**`optimizer.param_groups`：**
- 优化器内部可以有多组参数（每组可以有不同的学习率等超参）。
- 默认只有一组（即 `model.parameters()`），所以 `[-1]` 取最后一组。
- 直接修改 `param_group["lr"]` 即可在运行时动态调整学习率。

### 3.4 前向传播（第 53-63 行）

```python
with autocast_ctx:
    res = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps
```

**`autocast_ctx` — 混合精度上下文：**

```python
autocast_ctx = (
    nullcontext() if device_type == "cpu"
    else torch.cuda.amp.autocast(dtype=dtype)
)
```

- GPU 上：`torch.cuda.amp.autocast(dtype=torch.bfloat16)` — 自动混合精度
- CPU 上：`nullcontext()` — 不做任何精度转换

**什么是混合精度训练？**
- 默认所有计算用 float32（32位浮点数），精度高但慢、占内存。
- 混合精度让**大部分运算用 bfloat16/float16**（16位），只在必要时用 float32。
- 好处：速度快约 2 倍，显存省约一半，精度几乎无损。

**`autocast` 如何工作：**
- 它是一个上下文管理器（`with ... :`），在这个块内的运算会被自动选择精度。
- 某些运算对精度敏感（如 LayerNorm、Softmax），autocast 会自动用 float32。
- 其他运算（如矩阵乘法）用 bfloat16，加速计算。

**bfloat16 vs float16：**
| | float16 | bfloat16 |
|---|---------|----------|
| 数值范围 | 小（容易溢出） | 与 float32 相同 |
| 精度 | 较高 | 较低 |
| 稳定性 | 需要 GradScaler | 不需要（但本脚本仍用了） |

**`model(input_ids, labels=labels, attention_mask=attention_mask)` — 模型前向传播：**

模型内部做了什么（参见 ZzModel.py 第 634-675 行）：

```
input_ids → Embedding → Transformer Blocks (×N) → hidden_states → lm_head → logits
```

当传入 `labels` 时，模型**内部自动计算 loss**：

```python
# ZzMindForCausalLM.forward() 内部：
shift_logits = logits[..., :-1, :].contiguous()   # 去掉最后一个位置的预测
shift_labels = labels[..., 1:].contiguous()         # 去掉第一个位置的 label
loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),  # 展平为 [batch*seq_len, vocab_size]
    shift_labels.view(-1),               # 展平为 [batch*seq_len]
    ignore_index=-100,                    # 忽略 PAD 位置
)
```

这就是**自回归的错位预测**：用位置 t 的 logits 预测位置 t+1 的 token。

**`res.aux_loss` — MoE 辅助损失：**
- 如果模型使用了 MoE（Mixture of Experts），会有一个额外的 `aux_loss`（负载均衡损失），防止所有 token 都涌向同一个专家。
- 如果没有使用 MoE，`aux_loss = 0`，不影响普通模型。

**`loss / accumulation_steps` — 梯度累积：**
- 把 loss 除以累积步数，这样多次 backward 累加的梯度等价于一个大 batch 的一次梯度。
- 梯度累积在后面讲解。

**`res` 是什么类型？**
- 返回 `CausalLMOutputWithPast` 对象（HuggingFace 定义的数据类），包含：
  - `res.loss`：交叉熵损失
  - `res.logits`：模型输出 logits（训练时不直接使用）
  - `res.past_key_values`：KV 缓存（推理时加速，训练时不用）
  - `res.aux_loss`：MoE 辅助损失

### 3.5 反向传播 + 梯度累积（第 65-78 行）

```python
scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

这是训练循环中**最关键的代码段**，涉及混合精度的反向传播和梯度累积。

#### `scaler.scale(loss).backward()`

**`GradScaler` 是什么？**
- 混合精度训练中，float16 的梯度可能**下溢**（太小变成 0）。
- GradScaler 的原理：在 backward 之前，先**放大** loss（乘以一个大数），这样梯度也跟着放大，不会下溢。
- `scaler.scale(loss)` → 放大 loss
- `.backward()` → 反向传播，计算梯度

**为什么 bfloat16 也用了 scaler？**
```python
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
```
- 注意 `enabled` 参数：当 `dtype == "float16"` 时才真正启用缩放。
- 当 `dtype == "bfloat16"` 时，`enabled=False`，scaler 变成"透传"（不做任何缩放），但 API 调用仍然合法，保持代码统一。

#### 梯度累积

```
正常训练：loss.backward() → 更新 → 清零（每个 step 更新一次）

梯度累积（accumulation_steps=8）：
step 1: loss/8.backward()  → 累积梯度 1/8
step 2: loss/8.backward()  → 累积梯度 2/8
...
step 8: loss/8.backward()  → 累积梯度 8/8 = 完整梯度 → 更新 → 清零
```

- 效果等价于用 `batch_size × accumulation_steps` 的大 batch 训练，但显存只需要 `batch_size` 的量。
- `if step % args.accumulation_steps == 0` → 每累积 N 步才执行一次参数更新。

#### `scaler.unscale_(optimizer)`

- 把之前放大的梯度**还原**回真实值（除以缩放因子）。
- 必须在 `clip_grad_norm_` 之前调用，否则裁剪的是放大后的假梯度。

#### `clip_grad_norm_` — 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

- 计算所有参数梯度的 L2 范数，如果超过 `grad_clip`（默认 1.0），就等比缩小。
- **为什么需要？** 防止梯度爆炸——某些情况下梯度可能突然变得极大，导致参数更新太猛，模型崩溃。
- 梯度裁剪像一个"安全阀"，限制最大步长。

#### `scaler.step(optimizer)` 和 `scaler.update()`

- `scaler.step(optimizer)`：执行 `optimizer.step()`（参数更新），但如果梯度中有 inf/nan，则跳过更新（保护模型）。
- `scaler.update()`：根据本轮是否出现 inf/nan，动态调整缩放因子。

#### `optimizer.zero_grad(set_to_none=True)`

- 清零梯度，为下一轮累积做准备。
- `set_to_none=True` → 不只是把梯度设为 0，而是设为 `None`，更省内存。

### 3.6 日志记录（第 80-95 行）

```python
if step % args.log_interval == 0 or step == iters:
    spend_time = time.time() - start_time
    current_loss = loss.item() * args.accumulation_steps  # 恢复真实损失值
    current_lr = optimizer.param_groups[-1]["lr"]
    eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

    Logger(
        f"Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) "
        f"loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
    )

    if wandb:
        wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})
```

**关键知识点：**

| 表达式 | 含义 |
|--------|------|
| `loss.item()` | 从 Tensor 中取出 Python 浮点数（离开计算图） |
| `* args.accumulation_steps` | 还原真实 loss（之前除过了） |
| `:.6f` | 格式化为 6 位小数 |
| `:.12f` | 格式化为 12 位小数（学习率很小，需要高精度显示） |

**ETA 计算逻辑：**
```python
eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
```
- `spend_time / (step + 1)` → 平均每步耗时
- `* iters` → 剩余所有步的总耗时
- `// 60` → 转为分钟
- `- spend_time // 60` → 减去已花的时间，得到剩余时间

**`Logger` 函数（来自 trainer_utils.py）：**
```python
def Logger(content):
    if is_main_process():
        print(content)
```
- 只在主进程（rank 0）打印，分布式训练时避免每张 GPU 都输出一份日志。

### 3.7 模型保存（第 97-129 行）

```python
if (step % args.save_interval == 0 or step == iters) and is_main_process():
    model.eval()
    # ... 保存逻辑 ...
    model.train()
```

**`model.eval()` 和 `model.train()`：**
- `eval()` → 切换到评估模式（关闭 Dropout、BatchNorm 使用全局统计量）
- `train()` → 切回训练模式
- 保存前切 eval 是好习惯，虽然保存权重不受影响，但保持模式一致性。

#### 保存模型权重

```python
if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    state_dict = model.module.state_dict()
else:
    state_dict = model.state_dict()

state_dict = {k: v.half() for k, v in state_dict.items()}
torch.save(state_dict, ckp)
```

**`state_dict()` 是什么？**
- 返回一个字典 `{参数名: 参数Tensor}`，包含模型所有参数。
- 例如 `{"model.embed_tokens.weight": tensor([...]), "lm_head.weight": tensor([...]), ...}`
- 只保存参数值，不保存计算图。

**为什么要 `model.module.state_dict()`？**
- DDP 包装后，模型的原始参数在 `model.module` 里，直接 `model.state_dict()` 会有 `module.` 前缀。

**`v.half()` — 半精度存储：**
- 把 float32 参数转为 float16 再保存，文件大小减半。
- 加载时会自动还原。

**`torch.save(state_dict, path)` — 保存到磁盘：**
- 使用 Python 的 pickle 序列化，保存为 `.pth` 文件。

#### 保存完整训练状态（断点续训用）

```python
lm_checkpoint(
    lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
    scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir="../checkpoints",
)
```

这会额外保存一份 `_resume.pth`，包含：
- 模型参数
- 优化器状态（动量、方差估计）
- GradScaler 状态
- 当前 epoch 和 step
- WandB 实验 ID

---

## 四、主函数 `__main__`（第 132-360 行）

### 4.1 命令行参数定义（第 133-206 行）

使用 `argparse` 定义所有可配置参数，按功能分为 6 组：

| 参数组 | 参数 | 默认值 | 说明 |
|--------|------|--------|------|
| **基础** | `--epochs` | 1 | 训练轮数 |
| | `--batch_size` | 32 | 批大小 |
| | `--learning_rate` | 5e-4 | 学习率 |
| **硬件** | `--device` | cuda:0 或 cpu | 训练设备 |
| | `--dtype` | bfloat16 | 混合精度类型 |
| | `--num_workers` | 1 | 数据加载线程数 |
| **策略** | `--accumulation_steps` | 8 | 梯度累积步数 |
| | `--grad_clip` | 1.0 | 梯度裁剪阈值 |
| | `--log_interval` | 100 | 日志间隔 |
| | `--save_interval` | 100 | 保存间隔 |
| **模型** | `--hidden_size` | 512 | 隐藏层维度 |
| | `--num_hidden_layers` | 8 | Transformer 层数 |
| | `--max_seq_len` | 512 | 最大序列长度 |
| | `--use_moe` | 0 | 是否使用 MoE |
| **数据** | `--data_path` | ../dataset/pretrain_hq.jsonl | 数据文件 |
| | `--from_weight` | none | 加载已有权重 |
| | `--from_resume` | 0 | 是否断点续训 |
| **实验** | `--use_wandb` | False | 是否用 WandB |
| | `--wandb_project` | ZzMind-Pretrain | WandB 项目名 |

**默认值中的 `"cuda:0" if torch.cuda.is_available() else "cpu"`：**
- `torch.cuda.is_available()` → 检测是否有可用的 NVIDIA GPU
- 有 GPU 用 GPU，没有用 CPU

**`action="store_true"`（第 202 行）：**
- `--use_wandb` 是一个布尔开关，出现即为 True，不出现为 False。
- 不需要传值：`python train.py --use_wandb`（而非 `--use_wandb True`）

### 4.2 初始化环境（第 217-224 行）

```python
local_rank = init_distributed_mode()
if dist.is_initialized():
    args.device = f"cuda:{local_rank}"

setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
```

#### `init_distributed_mode()`（trainer_utils.py 第 26-33 行）

```python
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 单卡模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```

**分布式训练的启动方式：**
```bash
# 单卡（不启动分布式）
python train_pretrain.py

# 多卡（使用 torchrun 启动）
torchrun --nproc_per_node=4 train_pretrain.py
```

- `torchrun` 会自动设置 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 等环境变量。
- 如果 `RANK` 不存在（单卡模式），返回 `local_rank=0`。
- 如果存在（多卡模式），调用 `dist.init_process_group(backend="nccl")` 初始化通信。

| 环境变量 | 含义 |
|----------|------|
| `RANK` | 全局进程编号 |
| `LOCAL_RANK` | 单机内的 GPU 编号 |
| `WORLD_SIZE` | 总进程数 |

**`torch.cuda.set_device(local_rank)`：**
- 让当前进程绑定到指定的 GPU。
- 之后所有 `cuda` 操作默认在这个 GPU 上执行。

#### `setup_seed()`（trainer_utils.py 第 36-43 行）

```python
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

- 同时设置 Python、NumPy、PyTorch 的随机种子，确保实验可复现。
- 不同进程用 `42 + rank` 不同种子，保证每个 GPU 的数据增强有差异性。
- `cudnn.deterministic = True` → CuDNN 只使用确定性算法（牺牲一点速度换可复现性）
- `cudnn.benchmark = False` → 不自动搜索最快算法（因为搜索本身是随机的）

### 4.3 配置模型和检查点（第 233-250 行）

```python
os.makedirs(args.save_dir, exist_ok=True)

lm_config = ZzMindConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
    use_moe=bool(args.use_moe),
)

ckp_data = (
    lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
    if args.from_resume == 1
    else None
)
```

**`os.makedirs(path, exist_ok=True)`：**
- 递归创建目录（如果中间目录不存在也会创建）。
- `exist_ok=True` → 目录已存在时不报错。

**`ZzMindConfig`：**
- 模型配置类，把所有超参数打包成一个对象。
- 只传入 3 个参数（hidden_size、层数、是否 MoE），其余使用类中定义的默认值。
- 默认值：vocab_size=6400, num_attention_heads=8, num_key_value_heads=2, ...

**`lm_checkpoint()` 的两种模式：**
- **保存模式**（传入 model）：保存模型权重 + 训练状态到磁盘
- **加载模式**（不传 model）：从磁盘读取训练状态，返回字典或 None

### 4.4 设置混合精度（第 259-266 行）

```python
device_type = "cuda" if "cuda" in args.device else "cpu"
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

autocast_ctx = (
    nullcontext() if device_type == "cpu"
    else torch.cuda.amp.autocast(dtype=dtype)
)
```

**`"cuda" in args.device`：**
- `args.device` 可能是 `"cuda:0"`、`"cuda:1"` 或 `"cpu"`
- `"cuda" in "cuda:0"` → `True`（子串检查）

**三目运算符的嵌套：**
```python
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
# 等价于：
if args.dtype == "bfloat16":
    dtype = torch.bfloat16
else:
    dtype = torch.float16
```

### 4.5 WandB 实验跟踪（第 275-289 行）

```python
wandb = None
if args.use_wandb and is_main_process():
    import swanlab as wandb

    wandb_id = ckp_data.get("wandb_id") if ckp_data else None
    resume = "must" if wandb_id else None

    wandb_run_name = f"ZzMind-Pretrain-Epoch-{args.epochs}-..."
    wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
```

**为什么 `import swanlab as wandb`？**
- SwanLab 是国产的实验跟踪平台，API 与 WandB 兼容。
- 用 `as wandb` 别名，后面代码中统一用 `wandb.log()` 等接口，不关心底层是哪个平台。

**`resume="must"`：**
- 断点续训时，必须恢复到之前的实验记录（用 wandb_id 定位），不能创建新实验。
- 首次训练时 `resume=None`，创建新实验。

### 4.6 创建训练组件（第 300-327 行）

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

#### `init_model()`（trainer_utils.py 第 123-158 行）

```python
def init_model(lm_config, from_weight="pretrain", tokenizer_path=None, save_dir="../out", device="cuda"):
    from transformers import AutoTokenizer
    from model.ZzModel import ZzMindForCausalLM

    if tokenizer_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        tokenizer_path = os.path.join(project_root, "model")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = ZzMindForCausalLM(lm_config)

    if from_weight != "none":
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"可训练参数: {total_params / 1E6:.3f} 百万")

    return model.to(device), tokenizer
```

| 步骤 | 说明 |
|------|------|
| `AutoTokenizer.from_pretrained(path)` | 从 `model/` 目录加载 tokenizer 配置（tokenizer_config.json 等） |
| `ZzMindForCausalLM(config)` | 根据配置创建模型（随机初始化权重） |
| `torch.load(path, map_location=device)` | 加载权重文件，直接放到目标设备上 |
| `model.load_state_dict(weights, strict=False)` | 把权重载入模型。`strict=False` 允许键不完全匹配（如 vocab_size 变化时） |
| `p.numel()` | 计算参数张量的元素个数 |
| `model.to(device)` | 把整个模型移到 GPU |

**`torch.load(path, map_location=device)`：**
- `map_location` 指定加载到哪个设备。权重保存在 CPU 上时，指定 `map_location="cuda:0"` 可以避免先加载到 CPU 再搬到 GPU。

**`model.load_state_dict(weights, strict=False)`：**
- `strict=True`（默认）：权重字典的键必须与模型参数名完全一致，否则报错。
- `strict=False`：允许有不匹配的键（缺失的键用随机初始化，多余的键忽略）。这在微调时很有用（如改变了 vocab_size）。

#### GradScaler

```python
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
```

| 参数 | 含义 |
|------|------|
| `enabled=True` | 启用梯度缩放（float16 时需要） |
| `enabled=False` | 禁用缩放，所有方法变成透传（bfloat16 和 CPU 时用） |

### 4.7 断点续训状态恢复（第 311-327 行）

```python
start_epoch, start_step = 0, 0
if ckp_data:
    model.load_state_dict(ckp_data["model"])
    optimizer.load_state_dict(ckp_data["optimizer"])
    scaler.load_state_dict(ckp_data["scaler"])
    start_epoch = ckp_data["epoch"]
    start_step = ckp_data.get("step", 0)

if dist.is_initialized():
    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    model = DistributedDataParallel(model, device_ids=[local_rank])
```

**为什么要恢复 optimizer 和 scaler？**
- AdamW 优化器内部维护了每个参数的**一阶动量**（m）和**二阶动量**（v），这些是在训练过程中逐步积累的统计量。
- 如果不恢复，optimizer 会重新初始化动量为 0，相当于丢弃了之前积累的信息，训练会不稳定。
- GradScaler 同理，它内部有缩放因子的历史记录。

**`_ddp_params_and_buffers_to_ignore`：**
- 告诉 DDP：`freqs_cos` 和 `freqs_sin`（RoPE 位置编码的缓存）不需要跨 GPU 同步。
- 因为这些是固定值（只依赖于位置），每个 GPU 独立计算即可，同步反而浪费带宽。

### 4.8 训练循环（第 329-360 行）

```python
for epoch in range(start_epoch, args.epochs):
    if train_sampler:
        train_sampler.set_epoch(epoch)

    if epoch == start_epoch and start_step > 0:
        # 断点续训：跳过已训练的 batch
        batch_sampler = SkipBatchSampler(
            train_sampler or range(len(train_ds)), args.batch_size, start_step
        )
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, ...)
        train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb)
    else:
        # 正常训练
        loader = DataLoader(train_ds, batch_size=args.batch_size,
                           shuffle=(train_sampler is None), sampler=train_sampler, ...)
        train_epoch(epoch, loader, len(loader), 0, wandb)
```

**`train_sampler.set_epoch(epoch)`：**
- DistributedSampler 使用 epoch 作为随机种子。
- 不调用的话，每个 epoch 的数据顺序完全相同，模型会"记住"顺序而非学到规律。

#### `SkipBatchSampler`（trainer_utils.py 第 160-188 行）

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue      # 跳过这个 batch
                yield batch        # 返回这个 batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch            # 返回最后不满一个 batch 的残余

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
```

**断点续训的数据跳过逻辑：**
- 假设训练到第 300 步中断了（`skip_batches=300`）
- SkipBatchSampler 会正常遍历采样器，但前 300 个 batch 不 `yield`（丢弃）
- 从第 301 个 batch 开始正常 `yield`
- 这样就跳过了已经训练过的数据

**`yield` 关键字：**
- 把函数变成**生成器**（generator），每次 `yield` 返回一个值，下次调用从 `yield` 后继续。
- DataLoader 用 `for batch in sampler` 遍历时，每次取一个 batch。

**`len` 的计算：**
```python
total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
```
- 这是**向上取整除法**：`(a + b - 1) // b` 等价于 `ceil(a / b)`
- 例如 100 条数据、batch_size=32 → `(100 + 31) // 32 = 4` 个 batch

---

## 五、trainer_utils.py 中的 lm_checkpoint 详解

这个函数同时承担"保存"和"加载"两个职责，通过是否传入 `model` 参数来区分。

### 5.1 保存模式（model 不为 None）

```python
def lm_checkpoint(lm_config, weight="full_sft", model=None, optimizer=None,
                  epoch=0, step=0, wandb=None, save_dir="checkpoints", **kwargs):
```

**安全写入（原子保存）：**
```python
ckp_tmp = ckp_path + ".tmp"
torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
os.replace(ckp_tmp, ckp_path)
```
- 先写到 `.tmp` 临时文件，写完再用 `os.replace` 重命名。
- `os.replace` 是**原子操作**——要么完全成功，要么完全失败。
- 如果直接 `torch.save` 到目标路径，写入过程中断电/中断会留下一个**损坏的文件**。用临时文件可以避免这个问题。

**保存两份文件：**

| 文件 | 内容 | 用途 |
|------|------|------|
| `pretrain_512.pth` | 模型权重（half 精度） | 推理加载 / 继续微调 |
| `pretrain_512_resume.pth` | 模型权重 + 优化器 + epoch/step + wandb_id | 断点续训 |

### 5.2 加载模式（model 为 None）

```python
if os.path.exists(resume_path):
    ckp_data = torch.load(resume_path, map_location="cpu")
    saved_ws = ckp_data.get("world_size", 1)
    current_ws = dist.get_world_size() if dist.is_initialized() else 1

    if saved_ws != current_ws:
        ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws

    return ckp_data
return None
```

**GPU 数量变化的处理：**
- 如果保存时用 4 张 GPU（每个 GPU 处理 1/4 的 step），现在用 2 张 GPU，step 需要按比例调整。
- `step = 300 * 4 // 2 = 600`（原来 4 卡各训练 300 步 = 总共 1200 步的数据，现在 2 卡需要各训练 600 步）

---

## 六、完整训练流程图

```
python train_pretrain.py --epochs 2 --batch_size 32
│
├── 1. argparse 解析参数
│
├── 2. init_distributed_mode()
│   ├── 单卡 → local_rank=0
│   └── 多卡 → dist.init_process_group("nccl")
│
├── 3. setup_seed(42 + rank)
│
├── 4. ZzMindConfig(hidden_size=512, num_hidden_layers=8, ...)
│
├── 5. [可选] lm_checkpoint() 加载断点
│
├── 6. autocast_ctx = cuda.amp.autocast(bfloat16) 或 nullcontext()
│
├── 7. init_model() → model + tokenizer
│
├── 8. PretrainDataset(data_path, tokenizer, max_length=512)
│
├── 9. AdamW(model.parameters(), lr=5e-4)
│
├── 10. [多卡] DistributedDataParallel(model)
│
└── 11. for epoch in range(epochs):
        │
        ├── DataLoader(PretrainDataset) → batch
        │
        └── train_epoch():
            │
            ├── input_ids, labels, mask → .to(device)
            ├── get_lr() → 更新学习率
            ├── autocast → model(input_ids, labels) → loss
            ├── scaler.scale(loss).backward()
            ├── [每 N 步] unscale → clip → step → update → zero_grad
            ├── [每 log_interval 步] Logger(loss, lr, eta)
            └── [每 save_interval 步] torch.save() + lm_checkpoint()
```

---

## 七、关键 Python / PyTorch 知识点速查

### 上下文管理器

```python
with some_context_manager:
    # 在这个块内，上下文管理器生效
    do_something()
# 离开块后，上下文管理器自动清理

# 常见上下文管理器：
with torch.no_grad():        # 禁用梯度计算（节省内存）
with torch.cuda.amp.autocast():  # 混合精度
with nullcontext():           # 什么都不做（占位符）
```

### f-string 格式化

```python
name = "ZzMind"
f"Model: {name}, loss: {0.5:.6f}"
# → "Model: ZzMind, loss: 0.500000"

f"{name}_{512}.pth"
# → "ZzMind_512.pth"

f"{name}{'_moe' if True else ''}.pth"
# → "ZzMind_moe.pth"
```

### isinstance 类型检查

```python
isinstance(model, DistributedDataParallel)  # True/False
```

### 字典推导式

```python
state_dict = {k: v.half() for k, v in state_dict.items()}
# 等价于：
new_dict = {}
for k, v in state_dict.items():
    new_dict[k] = v.half()
```

### Tensor 操作

| 操作 | 示例 | 说明 |
|------|------|------|
| `.to(device)` | `tensor.to("cuda:0")` | 移到 GPU |
| `.item()` | `loss.item()` | 取出 Python 标量 |
| `.half()` | `tensor.half()` | 转 float16 |
| `.state_dict()` | `model.state_dict()` | 获取模型参数字典 |
| `.load_state_dict()` | `model.load_state_dict(d)` | 加载参数到模型 |
| `.parameters()` | `model.parameters()` | 获取所有可训练参数的迭代器 |
| `.numel()` | `tensor.numel()` | 元素总数 |
| `.requires_grad` | `param.requires_grad` | 是否需要梯度 |

---

## 八、常见问题

### Q1: 梯度累积为什么能模拟大 batch？
因为 PyTorch 的梯度是**累加**的。`loss.backward()` 计算的梯度会加到 `param.grad` 上，不会覆盖。所以：
```
loss1/8.backward()  → grad += ∇loss1/8
loss2/8.backward()  → grad += ∇loss2/8
...
loss8/8.backward()  → grad += ∇loss8/8
最终 grad = (∇loss1 + ... + ∇loss8) / 8 = ∇(mean loss)
```
等价于用 batch_size×8 的一次 forward+backward。

### Q2: 为什么保存权重用 `.half()`，但训练用 bfloat16？
- `.half()` = float16（2 字节），是通用的半精度存储格式。
- 训练用 bfloat16 是为了数值稳定性。
- 保存时统一用 float16 是因为文件大小减半的好处大于精度差异（加载后会转回训练精度）。

### Q3: autocast 块内为什么只包了 forward，不包 backward？
因为 `autocast` 只影响 **forward** 计算的精度。PyTorch 的 autocast 会自动记录哪些运算用了低精度，backward 时对应的梯度运算也自动使用匹配的精度。不需要（也不应该）把 backward 包在 autocast 里。

### Q4: 为什么 DistributedSampler 和 shuffle 不能同时用？
- `DistributedSampler` 自己管理数据分配和顺序。
- `shuffle=True` 会让 DataLoader 再打乱一次，两者冲突。
- 分布式训练时用 `sampler=train_sampler, shuffle=False`。
- 单卡训练时用 `sampler=None, shuffle=True`。

### Q5: 断点续训为什么要跳过已训练的 batch？
因为数据顺序由 `sampler.set_epoch(epoch)` 决定（同一个 epoch 内顺序固定）。如果不跳过，模型会重复训练前 N 个 batch 的数据，导致训练不均匀。SkipBatchSampler 确保从中断处继续。
