# Attention 模块全面解析

> 面向对象：有算法竞赛基础、刚学完 Transformer 理论的同学。
> 目标：从数学原理到代码实现，不跳步，完整理解项目中 `Attention` 类的每一行。

---

## 目录

1. [先修知识：什么是注意力机制](#1-先修知识什么是注意力机制)
2. [从最简单的注意力到缩放点积注意力](#2-从最简单的注意力到缩放点积注意力)
3. [多头注意力 (Multi-Head Attention)](#3-多头注意力-multi-head-attention)
4. [分组查询注意力 GQA (Grouped-Query Attention)](#4-分组查询注意力-gqa-grouped-query-attention)
5. [KV Cache 推理加速](#5-kv-cache-推理加速)
6. [旋转位置编码 RoPE](#6-旋转位置编码-rope)
7. [Flash Attention](#7-flash-attention)
8. [逐行代码解析](#8-逐行代码解析)
9. [数据流全图](#9-数据流全图)
10. [关键公式速查表](#10-关键公式速查表)

---

## 1. 先修知识：什么是注意力机制

### 1.1 直觉

假设你在阅读一句话：

> "小明把苹果给了**小红**，**她**很开心。"

当你理解"她"指的是谁时，你的大脑会"注意"到前面的"小红"。**注意力机制就是在模拟这个过程** —— 让模型在处理当前词时，能够"回头看"句子中的其他词，并决定哪些词更相关。

### 1.2 Q、K、V 的类比

注意力机制借用了信息检索的思想：

| 概念 | 含义 | 图书馆类比 |
|------|------|-----------|
| **Q** (Query) | 我现在想查什么 | 你在搜索框输入的关键词 |
| **K** (Key) | 每本书/文档的标签 | 每本书的书名/关键词标签 |
| **V** (Value) | 每本书/文档的内容 | 书的实际内容 |

**核心逻辑**：用 Q 和所有 K 做比较（算相似度），得到一组权重，再用这组权重对 V 做加权求和。

### 1.3 线性变换的作用

原始输入 `x` 是一个向量（比如词嵌入）。我们不让 x 直接当 Q/K/V，而是用**三个可学习的权重矩阵** W_Q、W_K、W_V 对 x 做线性变换：

```
Q = x @ W_Q    # (hidden_size,) @ (hidden_size, d_model) → (d_model,)
K = x @ W_K
V = x @ W_V
```

为什么要这样做？因为"查询"和"被查询"可能需要不同的表示空间。比如同样是"苹果"这个词，作为 Query 时可能在问"谁是接收者？"，作为 Key 时可能在表达"我是一个水果"。

---

## 2. 从最简单的注意力到缩放点积注意力

### 2.1 最朴素的注意力

```python
# Q: (seq_len, d), K: (seq_len, d), V: (seq_len, d)
scores = Q @ K.T          # (seq_len, seq_len)  每对 (i,j) 算一个点积
weights = softmax(scores)  # (seq_len, seq_len)  归一化为概率分布
output = weights @ V       # (seq_len, d)        加权求和
```

点积 `Q[i] · K[j]` 衡量的是向量 Q[i] 和 K[j] 的**相似程度**。点积越大 → 越相关 → 注意力权重越大。

### 2.2 为什么要"缩放" (Scale)?

**问题**：当维度 d 很大时，点积的绝对值会变得很大。

> 数学原因：假设 Q 和 K 的每个分量独立、均值为 0、方差为 1，则点积 `Q·K = Σ q_i * k_i` 的方差为 d。所以当 d=512 时，点积的方差就是 512。

softmax 对大值非常敏感：

```
输入 [1, 2, 3] → softmax → [0.09, 0.24, 0.67]  (分布还算平滑)
输入 [10, 20, 30] → softmax → [0.00, 0.00, 1.00]  (几乎变成了 one-hot)
```

当点积值域过大时，softmax 的梯度会**趋近于零**（梯度消失），模型无法学习。

**解决方案**：除以 √d

```
scaled_scores = (Q @ K.T) / √d
```

这就是 **Scaled Dot-Product Attention**（缩放点积注意力）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

### 2.3 因果掩码 (Causal Mask)

在语言模型中，预测第 t 个词时**不能看到第 t+1 个词**（否则就是作弊）。所以需要一个掩码矩阵：

```
对于一个长度为 4 的序列，因果掩码长这样：

     pos0  pos1  pos2  pos3
pos0 [  0   -inf  -inf  -inf ]
pos1 [  0     0   -inf  -inf ]
pos2 [  0     0     0   -inf ]
pos3 [  0     0     0     0  ]

0 表示正常参与计算，-inf 经 softmax 后变成 0（完全不注意）
```

上三角为 -inf → 每个 token 只能看到自己及之前的 token → 这就是"因果"的含义。

代码中的实现：
```python
scores[:, :, :, -sql:] += torch.triu(
    torch.full((sql, sql), float("-inf"), device=scores.device),
    diagonal=1    # 主对角线上方第1条线开始填 -inf
)
```

`torch.triu(..., diagonal=1)` 生成一个严格上三角矩阵（主对角线为 0，上方全为 -inf）。

---

## 3. 多头注意力 (Multi-Head Attention)

### 3.1 为什么需要多头？

单头注意力只能学一种"关注模式"。但语言中同时存在多种关系：

- 语法关系："她"指代"小红"
- 语义关系："开心"和"高兴"近义
- 位置关系：相邻词往往相关

**多头注意力**让模型同时学习多种关注模式。

### 3.2 具体做法

将 `hidden_size` 拆分成 `num_heads` 份，每份大小为 `head_dim = hidden_size / num_heads`。

```
例如：hidden_size=512, num_heads=8 → head_dim=64

原始 Q: (bsz, sql, 512)
拆分为: (bsz, sql, 8, 64)   → 8 个头，每个头维度 64
```

每个头**独立**做缩放点积注意力，然后把所有头的结果**拼接**回去：

```
各头输出: [(bsz, sql, 64)] × 8
拼接后:   (bsz, sql, 512)
最后用 o_proj 线性变换回 hidden_size
```

### 3.3 多头的数学表达

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

其中每个头：

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

---

## 4. 分组查询注意力 GQA (Grouped-Query Attention)

### 4.1 标准多头注意力 (MHA) 的问题

标准 MHA 中，每个 Q-head 都有自己对应的 K-head 和 V-head：

```
Q heads: [q0, q1, q2, q3, q4, q5, q6, q7]   (8个)
K heads: [k0, k1, k2, k3, k4, k5, k6, k7]   (8个)
V heads: [v0, v1, v2, v3, v4, v5, v6, v7]   (8个)

q0 只和 k0/v0 计算，q1 只和 k1/v1 计算 ...
```

在**推理**时需要缓存 K 和 V（见第 5 节）。8 个头意味着要缓存 8 份 K 和 8 份 V，占用大量显存。

### 4.2 GQA 的做法

将多个 Q-head 共享同一组 K/V-head：

```
配置：num_attention_heads=8, num_key_value_heads=2

Q heads: [q0, q1, q2, q3, q4, q5, q6, q7]   (8个)
K heads: [k0, k0, k0, k0, k1, k1, k1, k1]   (2个，每个被4个Q共享)
V heads: [v0, v0, v0, v0, v1, v1, v1, v1]   (2个，每个被4个Q共享)

n_rep = 8 / 2 = 4  → 每个 KV head 被复制 4 份
```

**好处**：KV Cache 大小减少 4 倍，推理速度大幅提升，性能损失很小。

### 4.3 三种注意力对比

| 类型 | Q heads | K/V heads | 显存 | 质量 |
|------|---------|-----------|------|------|
| MHA (Multi-Head) | 8 | 8 | 最高 | 最好 |
| GQA (Grouped-Query) | 8 | 2~4 | 中等 | 几乎不变 |
| MQA (Multi-Query) | 8 | 1 | 最低 | 略有下降 |

本项目默认配置：`num_attention_heads=8, num_key_value_heads=2`，即 **GQA**，每个 KV head 被 4 个 Q head 共享。

### 4.4 代码中的 `repeat_kv` 函数

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x 形状: (bsz, sql, num_kv_heads, head_dim)
    # 例: (1, 10, 2, 64)
    batch_size, seq_len, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    # 步骤1: 在第3维插入新维度 → (bsz, sql, kv_heads, 1, head_dim)
    # 步骤2: expand 复制 → (bsz, sql, kv_heads, n_rep, head_dim)
    # 步骤3: reshape 合并 → (bsz, sql, kv_heads * n_rep, head_dim)
    # 例: (1, 10, 2, 4, 64) → (1, 10, 8, 64)
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, num_key_value_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, num_key_value_heads * n_rep, head_dim)
    )
```

这个函数做的事情就是：把 2 个 KV head 复制扩展成 8 个，使得 K/V 的 head 数量和 Q 匹配，才能做矩阵乘法。

**注意**：`expand` 不实际复制内存，只是创建一个"视图"，非常高效。`reshape` 也是在可能的情况下只改 stride。

---

## 5. KV Cache 推理加速

### 5.1 问题：自回归生成的重复计算

语言模型生成文本时是**逐 token 生成**的：

```
第1步：输入 "今天"        → 预测 "天气"
第2步：输入 "今天天气"    → 预测 "真"
第3步：输入 "今天天气真"  → 预测 "好"
...
```

问题是：第 3 步计算注意力时，"今天"的 K 和 V 在第 1、2 步**已经算过了**，不必要重复计算。

### 5.2 KV Cache 方案

**缓存**每一步算出的 K 和 V：

```
第1步：算出 K₁, V₁，存入 cache
第2步：只需算新 token 的 K₂, V₂，和 cache 拼接：K = [K₁, K₂], V = [V₁, V₂]
第3步：只需算新 token 的 K₃, V₃，和 cache 拼接：K = [K₁, K₂, K₃], V = [V₁, V₂, V₃]
```

这样每步只计算**一个新 token** 的 K 和 V，而不是整个序列。

### 5.3 代码实现

```python
# forward 中的相关代码
if past_key_value is not None:
    # past_key_value[0] 是之前所有 token 的 K，past_key_value[1] 是 V
    xk = torch.cat([past_key_value[0], xk], dim=1)  # 在序列维度拼接
    xv = torch.cat([past_key_value[1], xv], dim=1)
past_kv = (xk, xv) if use_cache else None  # 返回新的 cache 给下一步用
```

`dim=1` 是序列长度维度，`torch.cat` 把旧的和新的沿序列方向拼起来。

---

## 6. 旋转位置编码 RoPE

### 6.1 为什么需要位置编码？

注意力机制本身是**置换不变的** —— 打乱输入顺序，输出只是对应位置打乱，值不变。这意味着模型不知道词的顺序，而词序对语言至关重要：

```
"狗咬人" ≠ "人咬狗"
```

所以需要显式地把位置信息注入模型。

### 6.2 RoPE 的核心思想

RoPE (Rotary Position Embedding) 的思路：**根据位置旋转 Q 和 K 向量**。

关键性质：两个向量的点积在旋转后，只取决于它们的**相对角度差**。所以 RoPE 天然编码了**相对位置关系**。

### 6.3 二维直觉

假设 Q 和 K 都是二维向量，位置分别是 m 和 n：

```
旋转 Q_m：角度 = m * θ
旋转 K_n：角度 = n * θ

旋转后的点积 = |Q| * |K| * cos((m-n) * θ)
                              ↑
                     只依赖相对位置 m-n！
```

### 6.4 推广到高维

对于 d 维向量，把它分成 d/2 对，每对施加不同频率的旋转：

```
第 0 对 (dim 0, 1): 旋转角度 = pos * θ₀ = pos / (base^(0/d))
第 1 对 (dim 2, 3): 旋转角度 = pos * θ₁ = pos / (base^(2/d))
第 2 对 (dim 4, 5): 旋转角度 = pos * θ₂ = pos / (base^(4/d))
...
```

θ 从大到小，对应频率从高到低。**低维旋转快（捕捉局部位置关系），高维旋转慢（捕捉远距离位置关系）。**

### 6.5 高效实现：不是真做旋转矩阵乘法

二维旋转矩阵是：
```
R(θ) = [cos(θ), -sin(θ)]
       [sin(θ),  cos(θ)]
```

对一个向量 [x₀, x₁] 旋转后：
```
[x₀'] = [cos(θ), -sin(θ)] [x₀]
[x₁']   [sin(θ),  cos(θ)] [x₁]

即：x₀' = x₀ * cos(θ) - x₁ * sin(θ)
    x₁' = x₀ * sin(θ) + x₁ * cos(θ)
```

对于 d 维向量，把所有 d/2 对的结果拼起来。代码中的 `rotate_half` 巧妙地实现了这一点：

```python
def rotate_half(x):
    # 把 x 的后半截取负后拼到前半截前面
    # x = [x0, x1, x2, x3, ..., x_{d-2}, x_{d-1}]
    # 结果 = [-x_{d/2}, ..., -x_{d-1}, x0, ..., x_{d/2-1}]
    return torch.cat(
        (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]),
        dim=-1
    )
```

然后：
```python
q_embd = q * cos + rotate_half(q) * sin
```

这等价于对每对 (x_{2i}, x_{2i+1}) 做旋转，只是用了一种**统一的向量化写法**避免了逐对循环。

**为什么可以这样做？** 把上面的旋转公式写成逐元素形式：

```
x' = [x₀·cos(θ₀) - x₁·sin(θ₀),   →  x·cos + rotate_half(x)·sin 的前半部分
      x₁·cos(θ₀) + x₀·sin(θ₀),        = x₀·cos + (-x₁)·sin ✓
      x₂·cos(θ₁) - x₃·sin(θ₁),
      x₃·cos(θ₁) + x₂·sin(θ₁),        后半部分
      ...]                               = x₁·cos + x₀·sin ✓ (因为 rotate_half 把 x₀ 移到了对应位置)
```

### 6.6 `precompute_freqs_cis` 预计算

```python
# 1. 计算每个维度对的频率 θ_i
freqs = 1 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
# freqs = [θ₀, θ₁, θ₂, ..., θ_{d/2-1}]

# 2. 每个位置乘以每个频率 → 得到旋转角度矩阵
t = torch.arange(end)              # [0, 1, 2, ..., end-1]
freqs = torch.outer(t, freqs)      # (end, d/2)，freqs[pos][i] = pos * θ_i

# 3. 计算 cos 和 sin，然后复制拼接到完整维度
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # (end, d)
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # (end, d)
```

为什么要 `cat` 两份？因为 `rotate_half` 把后半截移到前半截，所以 cos/sin 也需要对应地复制一份，使逐元素乘法能正确匹配。

---

## 7. Flash Attention

### 7.1 标准注意力的问题

标准注意力的中间过程：

```
xq:    (bsz, heads, sql, head_dim)     → 存入 GPU 显存
xk:    (bsz, heads, sql, head_dim)     → 存入 GPU 显存
scores: (bsz, heads, sql, sql)         → 存入 GPU 显存  ← O(n²) 显存！
weights: (bsz, heads, sql, sql)        → 存入 GPU 显存
output: (bsz, heads, sql, head_dim)    → 存入 GPU 显存
```

序列长度为 n 时，`scores` 和 `weights` 各需要 `O(n²)` 显存。当 n 很大时（如 32K），这非常浪费。

### 7.2 Flash Attention 的核心思想

**分块计算 (Tiling)**：把 Q、K、V 切成小块，每块放入 GPU 的 SRAM（片上高速缓存，小但极快）中计算，避免将中间的 O(n²) 矩阵写入 HBM（全局显存，大但慢）。

这就像：你不用把整道大题的草稿全写在纸上，可以在脑子里分步算，只把最终答案写下来。

### 7.3 效果

| 指标 | 标准注意力 | Flash Attention |
|------|-----------|-----------------|
| 显存 | O(n²) | O(n) |
| 速度 | 基准 | 2~4x 加速 |
| 数值结果 | 基准 | 等价（IO-aware 精确计算） |

### 7.4 代码中的使用

```python
if self.flash and (sql > 1) and (past_key_value is None) and (...):
    # 满足条件时使用 Flash Attention
    output = F.scaled_dot_product_attention(
        xq, xk, xv,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=True,   # 自动处理因果掩码
    )
```

使用条件解析：
- `self.flash`：运行环境支持 Flash Attention（PyTorch ≥ 2.0 且 CUDA 可用）
- `sql > 1`：只有序列长度 > 1 时才有意义（单个 token 不需要注意力）
- `past_key_value is None`：使用 KV Cache 时不走 Flash Attention（因为 K/V 长度和 Q 不一致）
- `attention_mask` 为空或全为 1：没有特殊的 padding mask

不满足条件时，退回到手动实现的注意力计算（第 277-290 行）。

---

## 8. 逐行代码解析

### 8.1 `__init__` 初始化

```python
class Attention(nn.Module):
    def __init__(self, args: ZzConfig):
        super().__init__()

        # ---------- 确定 KV head 数量 ----------
        # 如果配置没指定 num_key_value_heads，就退化为标准 MHA（KV heads = Q heads）
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        # Q heads 数量必须能被 KV heads 整除（否则无法均匀分组）
        assert args.num_attention_heads % self.num_key_value_heads == 0

        # ---------- 基本参数 ----------
        self.n_local_heads = args.num_attention_heads      # Q 的 head 数 (默认 8)
        self.n_local_kv_heads = self.num_key_value_heads   # KV 的 head 数 (默认 2)
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 KV head 被共享的次数 (4)
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个 head 的维度 (64)

        # ---------- QKV 投影层 ----------
        # 注意：这三个线性层都没有 bias（bias=False），这是 LLaMA 系列模型的惯例
        self.q_proj = nn.Linear(
            args.hidden_size,                           # 输入维度 512
            args.num_attention_heads * self.head_dim,   # 输出维度 8*64=512
            bias=False
        )
        # K 和 V 的输出维度更小：2*64=128（因为是 GQA，KV head 更少）
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)

        # ---------- 输出投影层 ----------
        # 把多头拼接后的结果映射回 hidden_size
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        # ---------- Dropout ----------
        self.attn_dropout = nn.Dropout(args.dropout)   # 对注意力权重 dropout
        self.resid_dropout = nn.Dropout(args.dropout)   # 对最终输出 dropout
        self.dropout = args.dropout

        # ---------- Flash Attention 开关 ----------
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")  # PyTorch 版本检查
            and args.flash_attention   # 配置开关
        )
```

### 8.2 `forward` 前向传播

```python
def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    # x 形状: (batch_size, seq_len, hidden_size) 例: (2, 10, 512)
    bsz, sql, _ = x.shape

    # ===== 步骤 1: 线性投影得到 Q、K、V =====
    xq = self.q_proj(x)   # (bsz, sql, 512)
    xk = self.k_proj(x)   # (bsz, sql, 128)
    xv = self.v_proj(x)   # (bsz, sql, 128)

    # reshape 成多头形式: (bsz, sql, num_heads, head_dim)
    xq = xq.view(bsz, sql, self.n_local_heads, self.head_dim)      # (bsz, sql, 8, 64)
    xk = xk.view(bsz, sql, self.n_local_kv_heads, self.head_dim)   # (bsz, sql, 2, 64)
    xv = xv.view(bsz, sql, self.n_local_kv_heads, self.head_dim)   # (bsz, sql, 2, 64)

    # ===== 步骤 2: 应用旋转位置编码 RoPE =====
    cos, sin = position_embeddings    # 预计算好的 cos/sin，形状 (max_seq, head_dim)
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
    # 注意：只对 Q 和 K 施加 RoPE，V 不需要位置信息

    # ===== 步骤 3: KV Cache 拼接 =====
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=1)  # 在序列维度拼接
        xv = torch.cat([past_key_value[1], xv], dim=1)
    past_kv = (xk, xv) if use_cache else None

    # ===== 步骤 4: 调整维度顺序，为矩阵乘法做准备 =====
    xq = xq.transpose(1, 2)                                # (bsz, 8, sql, 64)
    xk = repeat_kv(xk, self.n_rep).transpose(1, 2)         # (bsz, sql, 2, 64) → repeat → (bsz, sql, 8, 64) → (bsz, 8, sql, 64)
    xv = repeat_kv(xv, self.n_rep).transpose(1, 2)         # 同上

    # ===== 步骤 5: 计算注意力 =====
    if (self.flash and (sql > 1) and (past_key_value is None) and ...):
        # 走 Flash Attention 快速路径
        output = F.scaled_dot_product_attention(xq, xk, xv, ...)
    else:
        # 手动计算路径
        # 5a. 计算 Q·K^T / √d
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores 形状: (bsz, 8, sql, sql)

        # 5b. 添加因果掩码（只在最新的 sql 个位置之间加）
        scores[:, :, :, -sql:] += torch.triu(
            torch.full((sql, sql), float("-inf"), device=scores.device),
            diagonal=1
        )
        # 为什么要 scores[:, :, :, -sql:] 而不是直接 scores？
        # 因为如果用了 KV Cache，K 的长度 > Q 的长度，只有最后 sql 个位置是新的

        # 5c. 添加 attention_mask（处理 padding）
        if attention_mask is not None:
            # attention_mask: (bsz, sql)，1 表示有效 token，0 表示 padding
            # 扩展维度以匹配 scores 的 (bsz, heads, sql_q, sql_k) 形状
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 把 0 变成很大的负数，softmax 后趋近于 0（不注意 padding）
            extended_attention_mask = (1.0 - extended_attention_mask) * 1E-9
            scores = scores + extended_attention_mask

        # 5d. Softmax 归一化
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 为什么要 float() 再 type_as()？
        # 因为 softmax 对精度敏感，先转 float32 计算，再转回原始精度

        # 5e. Dropout
        scores = self.attn_dropout(scores)

        # 5f. 加权求和
        output = scores @ xv    # (bsz, 8, sql, 64)

    # ===== 步骤 6: 合并多头，输出投影 =====
    output = output.transpose(1, 2).reshape(bsz, sql, -1)
    # (bsz, 8, sql, 64) → transpose → (bsz, sql, 8, 64) → reshape → (bsz, sql, 512)

    # ⚠️ 注意这里有一个 bug：应该是 self.o_proj 而不是 self.q_proj
    output = self.resid_dropout(self.q_proj(output))
    # 正确写法：output = self.resid_dropout(self.o_proj(output))

    return output, past_kv
```

> **发现一个 Bug**：第 293 行 `self.q_proj(output)` 应该是 `self.o_proj(output)`。输出投影应该用专门的 `o_proj` 层，而不是复用 `q_proj`。这是一个影响模型正确性的错误。

---

## 9. 数据流全图

```
输入 x: (bsz, sql, 512)
    │
    ├── q_proj ──→ (bsz, sql, 512) ──→ view ──→ (bsz, sql, 8, 64)
    │                                                      │
    │                                                RoPE (apply_rotary_pos_emb)
    │                                                      │
    │                                              transpose ──→ (bsz, 8, sql, 64) ← Q
    │
    ├── k_proj ──→ (bsz, sql, 128) ──→ view ──→ (bsz, sql, 2, 64)
    │                                                │
    │                                          RoPE (apply_rotary_pos_emb)
    │                                                │
    │                                    + past_kv[0] (如果使用 KV Cache)
    │                                                │
    │                                    repeat_kv(n_rep=4)
    │                                                │
    │                                        transpose ──→ (bsz, 8, sql, 64) ← K
    │
    └── v_proj ──→ (bsz, sql, 128) ──→ view ──→ (bsz, sql, 2, 64)
                                                     │
                                           + past_kv[1] (如果使用 KV Cache)
                                                     │
                                         repeat_kv(n_rep=4)
                                                     │
                                             transpose ──→ (bsz, 8, sql, 64) ← V


Q (bsz, 8, sql, 64)
K (bsz, 8, sql, 64)  ──→ Q @ K^T ──→ (bsz, 8, sql, sql)
V (bsz, 8, sql, 64)       │
                           ├── / √64 (缩放)
                           ├── + causal_mask (因果掩码)
                           ├── + attention_mask (padding 掩码)
                           ├── softmax
                           ├── dropout
                           └── @ V ──→ (bsz, 8, sql, 64)
                                        │
                                  transpose + reshape ──→ (bsz, sql, 512)
                                        │
                                    o_proj ──→ (bsz, sql, 512)
                                        │
                                    dropout
                                        │
                                    output
```

---

## 10. 关键公式速查表

| 公式 | 含义 |
|------|------|
| $Q = XW_Q,\ K = XW_K,\ V = XW_V$ | 线性投影得到 Q、K、V |
| $\text{score} = \frac{QK^T}{\sqrt{d_k}}$ | 缩放点积计算注意力分数 |
| $\text{Attn} = \text{softmax}(\text{score} + \text{mask})$ | 归一化为概率分布 |
| $\text{output} = \text{Attn} \cdot V$ | 加权求和 |
| $\text{GQA}: n\_rep = \frac{n\_heads}{n\_kv\_heads}$ | GQA 中每个 KV head 被共享的次数 |
| $\theta_i = \frac{1}{\text{base}^{2i/d}}$ | RoPE 频率公式 |
| $q' = q \odot \cos(\theta) + \text{rotate\_half}(q) \odot \sin(\theta)$ | RoPE 旋转等价实现 |

---

## 总结

本项目的 Attention 模块是一个现代的、工程化的注意力实现，融合了以下关键技术：

1. **GQA (分组查询注意力)** — 平衡推理效率和模型质量
2. **RoPE (旋转位置编码)** — 用旋转编码相对位置，支持长度外推
3. **KV Cache** — 自回归推理时缓存避免重复计算
4. **Flash Attention** — 训练时减少显存占用、加速计算
5. **因果掩码 + Padding 掩码** — 保证自回归的正确性和变长序列处理

理解了这一个模块，你就理解了当前主流大模型（LLaMA、Qwen、DeepSeek 等）注意力层的核心架构。它们之间的差异主要在超参数选择和少量实现细节上，整体框架是一致的。
