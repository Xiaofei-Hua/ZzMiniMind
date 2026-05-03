# MiniMind 项目全面解析

> 逐行、逐模块、逐公式，完整梳理 MiniMind 大语言模型的实现原理。

---

## 目录

1. [项目整体架构](#1-项目整体架构)
2. [配置模块：ZzMindConfig](#2-配置模块zzmindconfig)
3. [归一化：RMSNorm](#3-归一化rmsnorm)
4. [位置编码：precompute_freqs](#4-位置编码precompute_freqs)
5. [位置编码应用：apply_rotary_pos_emb](#5-位置编码应用apply_rotary_pos_emb)
6. [GQA 辅助：repeat_kv](#6-gqa-辅助repeat_kv)
7. [注意力：Attention](#7-注意力attention)
8. [前馈网络：FeedForward](#8-前馈网络feedforward)
9. [MoE 门控：MoEGate](#9-moe-门控moegate)
10. [MoE 前馈：MoEFeedForward](#10-moe-前馈moefeedforward)
11. [Transformer 块：ZzMindBlock](#11-transformer-块zzmindblock)
12. [模型主体：ZzMindModel](#12-模型主体zzmindmodel)
13. [语言模型：ZzMindForCausalLM](#13-语言模型zzmindforcausallm)
14. [完整数据流图](#14-完整数据流图)
15. [训练与推理的区别](#15-训练与推理的区别)

---

## 1. 项目整体架构

MiniMind 是一个基于 Decoder-only 架构的大语言模型，参数量约 2600 万（Dense 配置），适合学习和实验。

**输入输出**：
- **输入**：Token ID 序列 `(B, S)`，`B` 为 batch size，`S` 为序列长度
- **输出**：每个位置对应词表上每个词的概率分布 `(B, S, V)`，`V` 为词表大小

整体结构如下：

```
输入 Token IDs: (B, S)
    │
    ▼
┌─────────────────────────────────────┐
│  Embedding: vocab_size → hidden_size │
│  输出: (B, S, D)                     │
└──────────────┬──────────────────────┘
               │
               ▼
        ┌──────────────┐
        │              │
        │  N 个 Block  │  重复 num_hidden_layers 次
        │              │
        │  ┌──────────┐│
        │  │ RMSNorm  ││
        │  │ Attention││
        │  │ 残差连接  ││
        │  ├──────────┤│
        │  │ RMSNorm  ││
        │  │ FFN/MoE  ││
        │  │ 残差连接  ││
        │  └──────────┘│
        │              │
        └──────────────┘
               │
               ▼
        ┌──────────────┐
        │   RMSNorm    │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │   LM Head    │  Linear(hidden_size → vocab_size)
        │  (和 Embedding 共享权重) │
        └──────┬───────┘
               │
               ▼
        输出 Logits: (B, S, vocab_size)
```

**关键设计选择**：

| 设计 | 选择 | 原因 |
|------|------|------|
| 架构 | Decoder-only | 自回归语言生成的标准架构 |
| 归一化 | Pre-LN (RMSNorm) | 训练更稳定，LLaMA 系列标配 |
| 注意力 | GQA + RoPE | 减少 KV Cache 显存，外推性好 |
| FFN | SwiGLU | 表达能力强，LLaMA2+ 标配 |
| MoE | 可选 | 稀疏激活，大幅增加参数而不增加计算 |
| 权重共享 | Embedding 与 LM Head 共享 | 减少参数量，理论上有益 |

---

## 2. 配置模块：ZzMindConfig

### 2.1 代码

```python
class ZzMindConfig(PretrainedConfig):
    model_type = "zzmind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
```

### 2.2 参数详解

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `hidden_size` | 512 | 模型隐藏维度 $d_{model}$ |
| `num_hidden_layers` | 8 | Transformer Block 层数 |
| `num_attention_heads` | 8 | 注意力头数 $H$ |
| `num_key_value_heads` | 2 | KV 头数 $H_{kv}$，GQA 的关键 |
| `intermediate_size` | None | FFN 中间维度，None 时自动计算为 $\frac{8}{3} \times hidden\_size$ |
| `max_position_embeddings` | 32768 | 最大序列长度 |
| `vocab_size` | 6400 | 词表大小 |
| `rope_theta` | 1000000 | RoPE 频率基数 |
| `use_moe` | False | 是否使用 MoE |
| `n_routed_experts` | 4 | 路由专家数量 |
| `num_experts_per_tok` | 2 | 每个 token 激活的专家数 |
| `inference_rope_scaling` | False | 推理时是否启用 YaRN 长度扩展 |

### 2.3 `rope_scaling` 的初始化逻辑

```python
self.rope_scaling = (
    {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 2048,
        "attention_factor": 1.0,
        "type": "yarn",
    }
    if self.inference_rope_scaling
    else None
)
```

当 `inference_rope_scaling=True` 时，`rope_scaling` 是一个配置字典，包含 YaRN 扩展所需的超参数；否则为 `None`，不启用扩展。

### 2.4 核心超参数关系

$$
head\_dim = \frac{hidden\_size}{num\_attention\_heads} = \frac{512}{8} = 64
$$

$$
n\_rep = \frac{num\_attention\_heads}{num\_key\_value\_heads} = \frac{8}{2} = 4
$$

$$
intermediate\_size = \frac{8}{3} \times hidden\_size = \frac{8}{3} \times 512 \approx 1365 \rightarrow \text{向上取整到 64 倍数} = 1408
$$

---

## 3. 归一化：RMSNorm

### 3.1 为什么用 RMSNorm 而不是 LayerNorm？

标准 LayerNorm：

$$
LayerNorm(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

RMSNorm 去掉了减去均值的操作：

$$
RMSNorm(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

**原因**：
- 在大模型中，减去均值的操作对训练稳定性的贡献很小
- 去掉均值计算可以减少一次统计量计算，略微加速
- LLaMA 系列验证：RMSNorm 效果不劣于 LayerNorm

### 3.2 代码解析

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1E-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x.pow(2).mean(-1, keepdim=True): 计算每个样本的均方值
        # torch.rsqrt: 计算平方根的倒数 = 1 / sqrt(...)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # x.float(): 先转 float32 计算，提高数值稳定性
        # type_as(x): 再转回原始精度（fp16/bf16）
        return self.weight * self._norm(x.float()).type_as(x)
```

**为什么先转 float32？**

在低精度（fp16/bfloat16）下直接计算 `rsqrt(x.pow(2).mean())` 可能出现数值不稳定：
- fp16 的动态范围有限，平方后容易溢出或下溢
- rsqrt（平方根倒数）在接近 0 的值上精度损失大

先转 float32 计算归一化，再转回原始精度，是 LLaMA 等模型的标准做法。

**数学公式**：

$$
RMS(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

$$
RMSNorm(x) = \frac{x}{RMS(x)} \odot \gamma
$$

其中 $\gamma$ 是可学习的缩放参数 `self.weight`。

---

## 4. 位置编码：precompute_freqs

### 4.1 标准 RoPE 频率计算

```python
freqs = 1 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
```

注意：`torch.arange(0, dim, 2)` 本身生成长度为 `dim//2` 的序列，`[: (dim // 2)]` 切片在此场景下是多余的（已包含全部元素），但代码保留此切片是一种防御性编程习惯。

**数学推导**：

$$
\theta_j = \frac{1}{\theta_{base}^{2j/d}}, \quad j = 0, 1, \ldots, \frac{d}{2}-1
$$

### 4.2 YaRN 长度扩展（可选）

当 `inference_rope_scaling=True` 时启用 YaRN，用于推理时扩展序列长度。

**核心思想**：对低频维度的旋转频率进行插值缩放，同时保持高频维度不变。

```python
# 波长到维度索引的映射
inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

# 高频区和低频区的切分点
low = floor(inv_dim(beta_fast))    # 高于此频率的不缩放
high = ceil(inv_dim(beta_slow))    # 低于此频率的全量缩放

# 线性过渡 (Ramp)
ramp = clamp((j - low) / (high - low), 0, 1)

# 频率融合
freqs' = freqs * ((1 - ramp) + ramp / factor)
```

**公式**：

$$
\theta_j' = \theta_j \times \left((1 - \gamma_j) + \frac{\gamma_j}{s}\right)
$$

其中 $\gamma_j$ 是混合因子，$s$ 是扩展倍数。

### 4.3 生成 cos 和 sin

```python
t = torch.arange(end, device=freqs.device)
freqs = torch.outer(t, freqs).float()

freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
```

**`attn_factor`（注意力温度补偿）**：

YaRN 扩展序列长度后，注意力分布会变平缓（熵增加）。`attn_factor` 是一个缩放系数（默认 1.0），用于让注意力重新"聚焦"。当 `rope_scaling` 不为 None 时从配置中读取。

**形状**：

- `freqs`: $(E, d/2)$ — 每个位置、每对维度的旋转角度
- `freqs_cos/freqs_sin`: $(E, d)$ — 复制两份后的 cos/sin 查找表

---

## 5. 位置编码应用：apply_rotary_pos_emb

### 5.1 rotate_half

```python
def rotate_half(x):
    d = x.shape[-1]
    return torch.cat((-x[..., d//2:], x[..., :d//2]), dim=-1)
```

**作用**：将向量后半截取负后拼到前面。

```
输入:  [a, b, c, d, e, f, g, h]
输出:  [-e, -f, -g, -h, a, b, c, d]
```

**配对方式**：半分配对 $(x_i, x_{i+d/2})$。

### 5.2 RoPE 应用

```python
q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
```

**`unsqueeze_dim=1` 的含义**：

- `q` 的形状是 `(B, S, H, d)`
- `cos` 的形状是 `(S, d)`
- `cos.unsqueeze(1)` → `(S, 1, d)`
- 广播后与 `(B, S, H, d)` 逐元素相乘

在 `dim=1` 插入大小为 1 的维度，使得 `S` 维度对齐，`1` 广播到 `H`。

**逐元素展开**（第 j 对）：

$$
q'_{j} = q_{j} \cos(pos \cdot \theta_j) - q_{j+d/2} \sin(pos \cdot \theta_j)
$$

$$
q'_{j+d/2} = q_{j+d/2} \cos(pos \cdot \theta_j) + q_{j} \sin(pos \cdot \theta_j)
$$

**关键性质**：

$$
q'_m \cdot k'_n = f(m - n)
$$

点积只和相对位置有关，与绝对位置无关。

---

## 6. GQA 辅助：repeat_kv

### 6.1 GQA 回顾

GQA（Grouped Query Attention）：多个 Q head 共享一组 KV head。

```
Q heads: 8 个
KV heads: 2 个
n_rep = 4  — 每个 KV head 被 4 个 Q head 共享
```

### 6.2 实现

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]                                    # (B, S, H_kv, 1, d)
        .expand(batch_size, seq_len, num_key_value_heads, n_rep, head_dim)  # (B, S, H_kv, R, d)
        .reshape(batch_size, seq_len, num_key_value_heads * n_rep, head_dim)  # (B, S, H, d)
    )
```

**关键点**：`expand` 是视图操作，**不实际复制内存**，只是改变 stride 让多个 head 看到同一份数据。

---

## 7. 注意力：Attention

### 7.1 整体结构

```python
class Attention(nn.Module):
    def __init__(self, args: ZzMindConfig):
        # Q/K/V/O 四个线性投影
        self.q_proj = nn.Linear(hidden_size, H * d, bias=False)
        self.k_proj = nn.Linear(hidden_size, H_kv * d, bias=False)
        self.v_proj = nn.Linear(hidden_size, H_kv * d, bias=False)
        self.o_proj = nn.Linear(H * d, hidden_size, bias=False)
```

### 7.2 Forward 流程

**步骤 1：线性投影**

```python
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
# xq: (B, S, H*d)    → view → (B, S, H, d)
# xk: (B, S, H_kv*d) → view → (B, S, H_kv, d)
# xv: (B, S, H_kv*d) → view → (B, S, H_kv, d)
```

**步骤 2：应用 RoPE**

```python
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
# 输出: (B, S, H, d) 和 (B, S, H_kv, d)
```

**步骤 3：KV Cache**

```python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)
past_kv = (xk, xv) if use_cache else None
```

**`use_cache` 的作用**：
- `use_cache=True`（推理时）：返回 `(xk, xv)` 作为下一轮的 `past_key_value`
- `use_cache=False`（训练时）：返回 `None`，不缓存 KV（因为训练时一次性处理整个序列，不需要缓存）

**步骤 4：调整维度**

```python
xq = xq.transpose(1, 2)                    # (B, H, S, d)
xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # (B, H, S_total, d)
xv = repeat_kv(xv, self.n_rep).transpose(1, 2)  # (B, H, S_total, d)
```

**步骤 5：注意力计算**

**路径 A：Flash Attention（训练时）**

```python
if (
    self.flash and                    # 1. 环境支持（PyTorch ≥ 2.0 + CUDA）
    (sql > 1) and                     # 2. 序列长度 > 1（单token不需要注意力）
    (past_key_value is None) and      # 3. 不使用 KV Cache（Flash Attn 要求 Q/K 长度相同）
    (attention_mask is None or        # 4. 没有自定义 padding mask
     torch.all(attention_mask == 1))
):
    output = F.scaled_dot_product_attention(
        xq, xk, xv,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=True,
    )
```

**为什么需要这四个条件？**

| 条件 | 原因 |
|------|------|
| `self.flash` | PyTorch 2.0+ 提供 `scaled_dot_product_attention`，自动选择最优内核（Flash Attention / Memory-Efficient Attention / Math） |
| `sql > 1` | 序列长度为 1 时，注意力退化为 identity，不需要计算 |
| `past_key_value is None` | Flash Attention 要求 Q 和 K 的序列长度相同。使用 KV Cache 时，K 的长度 > Q 的长度 |
| `attention_mask` 全为 1 | Flash Attention 的 `is_causal=True` 只处理因果掩码，不处理自定义 padding 掩码 |

**路径 B：手动计算（推理/KV Cache 时）**

```python
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
# scores: (B, H, S, S_total)

# 因果掩码
scores[:, :, :, -sql:] += torch.triu(
    torch.full((sql, sql), float("-inf"), device=scores.device),
    diagonal=1
)

# Padding 掩码
if attention_mask is not None:
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = (1.0 - extended_attention_mask) * 1E-9
    scores = scores + extended_attention_mask

# Softmax + Dropout + 加权求和
# scores.float(): softmax 对精度敏感，先转 float32 计算再转回
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
scores = self.attn_dropout(scores)
output = scores @ xv    # (B, H, S, d)
```

**因果掩码的详细解释**：

`scores[:, :, :, -sql:]` 的含义：

| 场景 | `S_total` | `-sql:` 切片 |
|------|----------|-------------|
| **训练时**（无 KV Cache）| `S_total = S` | 覆盖全部列，完整的 `S × S` 因果掩码 |
| **推理时**（有 KV Cache）| `S_total = S_cache + S` | 只覆盖最后 `S` 列，当前输入的 `S × S` 掩码 |

`scores` 形状为 `(B, H, S, S_total)`。推理时前面 `S_cache` 列是历史 token（已处理过），只需在最后 `S` 列加三角掩码；训练时 `S_total = S`，`-sql:` 即全部列。

**Padding Mask 的注意事项**：

```python
extended_attention_mask = (1.0 - extended_attention_mask) * 1E-9
```

⚠️ 这里 `1E-9` 是一个很小的正数。当 `attention_mask=0`（padding）时，结果是 `1E-9`，加到 score 上几乎不影响 softmax，**padding 位置的 token 仍然会被注意到**。

正确的做法应该是产生一个很大的负数（如 `-1E9`），使 padding 位置在 softmax 后权重趋近于 0：

```python
extended_attention_mask = (extended_attention_mask - 1.0) * 1E9
# 当 mask=1 时: (1-1)*1E9 = 0（不影响）
# 当 mask=0 时: (0-1)*1E9 = -1E9（softmax 后趋近 0）
```

**步骤 6：输出投影**

```python
output = output.transpose(1, 2).reshape(bsz, sql, -1)  # (B, S, H*d)
output = self.resid_dropout(self.o_proj(output))        # (B, S, hidden_size)
```

### 7.3 缩放点积注意力的数学

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}} + Mask\right) V
$$

**为什么要除以 $\sqrt{d_k}$？**

当 $d_k$ 很大时，点积的方差为 $d_k$。softmax 对大的输入值敏感，会导致梯度消失。除以 $\sqrt{d_k}$ 将方差归一化为 1。

**因果掩码**：

$$
Mask_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}
$$

确保每个位置只能看到自己和之前的 token。

---

## 8. 前馈网络：FeedForward

### 8.1 SwiGLU 结构

```python
class FeedForward(nn.Module):
    def __init__(self, config: ZzMindConfig):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]   # SiLU(x) = x * sigmoid(x)

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
```

**数学公式**：

$$
FFN(x) = W_{down}(SiLU(x W_{gate}) \odot (x W_{up}))
$$

其中：

$$
SiLU(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

### 8.2 为什么 SwiGLU 比标准 FFN 好？

标准 FFN：

$$
FFN(x) = ReLU(x W_1) W_2
$$

SwiGLU 引入了门控机制：

```
gate = SiLU(x W_gate)   ← 门控信号（0 到 1 之间的连续值）
up   = x W_up           ← 值信号
gated = gate * up       ← 门控筛选后的值
output = W_down(gated)  ← 降维投影
```

**优势**：
- 门控可以**连续调节**每个维度的通过程度，比 ReLU 的二元开关更灵活
- 两条路径同时学习，表达能力更强
- 大模型实验验证：SwiGLU 效果优于 ReLU/GELU

### 8.3 intermediate_size 的计算

```python
intermediate_size = int(hidden_size * 8 / 3)
intermediate_size = 64 * ((intermediate_size + 63) // 64)
```

**公式**：

$$
d_{inter} = 64 \times \left\lceil \frac{8 \times hidden\_size}{3 \times 64} \right\rceil
$$

**为什么是 8/3？**

SwiGLU 有三条投影路径（gate、up、down），参数量比标准 FFN 多。为了控制总参数量，中间维度取 $\frac{8}{3} \approx 2.67$ 倍，而不是标准 FFN 的 4 倍。

向上对齐到 64 的倍数是为了 GPU 计算的内存对齐效率。

---

## 9. MoE 门控：MoEGate

### 9.1 核心思想

MoE（Mixture of Experts）：每个 token 只激活一小部分专家 FFN，而不是所有专家。

```
输入 token
    │
    ▼
┌──────────────┐
│  门控网络     │  计算每个专家的"适合程度"
│  score = [0.7, 0.2, 0.05, 0.05]
└──────┬───────┘
       │
       ▼
    Top-K 选择 (K=2)
       │
┌──────┴──────┐
│ 专家 0 (0.7) │
│ 专家 1 (0.2) │
│ 专家 2,3 (忽略)│
└─────────────┘
       │
       ▼
output = 0.7 * expert_0(x) + 0.2 * expert_1(x)
```

### 9.2 代码解析

**门控得分计算**：

```python
hidden_states = hidden_states.view(-1, h)   # (B*S, hidden_size)
logits = F.linear(hidden_states, self.weight, None)   # (B*S, n_experts)
scores = logits.softmax(dim=-1)              # (B*S, n_experts)
```

**数学**：

$$
s_i = \frac{e^{z_i}}{\sum_{j=1}^{E} e^{z_j}}
$$

其中 $z = x W_{gating}^T$，$W_{gating} \in \mathbb{R}^{E \times d}$。

**Top-K 选择**：

```python
topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
# topk_weight: (B*S, K)
# topk_idx:    (B*S, K)
```

**Top-K 归一化**：

```python
if self.top_k > 1 and self.norm_topk_prob:
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
```

**负载均衡辅助损失**：

```python
if self.training and self.alpha > 0.0:
    if self.seq_aux:
        # 序列级辅助损失
        ce = torch.zeros(bsz, n_routed_experts)
        ce.scatter_add_(1, topk_idx_for_aux_loss, ones)
        ce.div_(sql * aux_topk / n_routed_experts)
        aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * alpha
    else:
        # Token 级辅助损失
        ce = mask_ce.float().mean(0)
        Pi = scores_for_aux.mean(0)
        fi = ce * n_routed_experts
        aux_loss = (Pi * fi).sum() * alpha
```

**数学公式**：

**Token 级**：

$$
\mathcal{L}_{aux} = \alpha \cdot \sum_{i=1}^{E} P_i \cdot f_i
$$

其中 $P_i$ 是专家 $i$ 的路由概率均值，$f_i = \frac{n_i}{T} \cdot E$ 是负载因子。

**序列级**：

$$
\mathcal{L}_{aux} = \alpha \cdot \frac{1}{B} \sum_{b=1}^{B} \sum_{i=1}^{E} c_{b,i} \cdot \bar{s}_{b,i}
$$

其中 $c_{b,i}$ 是第 $b$ 个序列中专家 $i$ 的归一化负载，$\bar{s}_{b,i}$ 是平均路由概率。

**为什么需要辅助损失？**

没有约束时，模型会倾向于把所有 token 路由到同一个"最好"的专家（**路由坍缩**）。辅助损失强制 token 均匀分布在各专家上。

---

## 10. MoE 前馈：MoEFeedForward

### 10.1 整体结构

```python
class MoEFeedForward(nn.Module):
    def __init__(self, config: ZzMindConfig):
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(n_routed)])
        self.gate = MoEGate(config)
        if n_shared > 0:
            self.shared_experts = nn.ModuleList([FeedForward(config) for _ in range(n_shared)])
```

**共享专家（Shared Experts）**：所有 token 都会经过的固定专家，不受门控影响。提供基础能力，让路由专家专注于特定领域的知识。

### 10.2 训练 Forward

```python
def forward(self, x):
    identity = x
    topk_idx, topk_weight, aux_loss = self.gate(x)
    x = x.view(-1, x.shape[-1])     # (B*S, hidden_size)

    # 每个 token 复制 K 份，分别送给 K 个专家
    x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
    # x: (B*S*K, hidden_size)

    # 创建输出缓冲区
    y = torch.empty_like(x)

    # 遍历每个专家，处理分配给它的 token
    for i, expert in enumerate(self.experts):
        # 取出分配给专家 i 的所有 token（可能为空张量）
        expert_out = expert(x[flat_topk_idx == i])
        if expert_out.shape[0] > 0:
            # 有 token 分配给专家 i，正常写入输出
            y[flat_topk_idx == i] = expert_out.to(y.dtype)
        else:
            # ⚠️ 梯度 trick：没有 token 分配给专家 i 时，
            # expert_out 是空张量 shape [0, hidden_size]，
            # 通过 "0 * sum(parameters)" 创建虚拟梯度路径，
            # 确保该专家的参数仍参与反向传播（梯度为 0）
            y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                p.sum() for p in expert.parameters()
            )

    # 加权求和
    y = (y.view(B*S, K, hidden_size) * topk_weight.unsqueeze(-1)).sum(dim=1)
    # y: (B*S, hidden_size)
    y = y.view(*orig_shape)

    # 加上共享专家的输出
    # 共享专家与路由专家是"加性"关系：y = routed_expert_output + shared_expert_output
    # 共享专家提供基础能力，所有 token 都必须经过
    if self.config.n_shared_experts > 0:
        for expert in self.shared_experts:
            y = y + expert(identity)

    self.aux_loss = aux_loss
    return y
```

### 10.3 推理 Forward (moe_infer)

推理时使用更高效的实现：

```python
@torch.no_grad()
def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
    expert_cache = torch.zeros_like(x)

    # 按专家索引排序 token
    idxs = flat_expert_indices.argsort()
    tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
    token_idxs = idxs // self.config.num_experts_per_tok

    # 逐个专家处理
    for i, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
        if start_idx == end_idx:
            continue

        expert = self.experts[i]
        exp_token_idx = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idx]

        # 批量处理
        expert_out = expert(expert_tokens)
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

        # 散点累加回结果
        expert_cache.scatter_add_(
            0,
            exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
            expert_out
        )

    return expert_cache
```

**训练和推理的区别**：

| | 训练 | 推理 |
|---|---|---|
| 实现 | `repeat_interleave` + 逐专家处理 | `argsort` + 批量处理 |
| 梯度 | 需要梯度 | `@torch.no_grad()` |
| 目的 | 支持梯度回传 | 更高效，减少内存分配 |
| token 分配 | 每个 token 复制 K 份 | 按专家排序后批量处理 |

---

## 11. Transformer 块：ZzMindBlock

### 11.1 结构

```python
class ZzMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: ZzMindConfig):
        self.self_attention = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MoEFeedForward(config)
```

### 11.2 Forward

```python
def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    res = hidden_states

    # Pre-Norm + Attention + 残差
    hidden_states, present_key_value = self.self_attention(
        self.input_layernorm(hidden_states),
        position_embeddings,
        past_key_value,
        use_cache,
        attention_mask,
    )
    hidden_states = res + hidden_states

    # Pre-Norm + FFN/MoE + 残差
    hidden_states = hidden_states + self.mlp(
        self.post_attention_layernorm(hidden_states)
    )

    return hidden_states, present_key_value
```

**Pre-Norm vs Post-Norm**：

```
Pre-Norm (本项目):  Norm → SubLayer → +残差
                    训练更稳定，深层网络不易梯度消失

Post-Norm (原始 Transformer):  SubLayer → Norm → +残差
                              训练不稳定，需要 warmup
```

**数学表达**：

$$
x_{l+1} = x_l + Attention(RMSNorm(x_l))
$$

$$
x_{l+1}' = x_{l+1} + FFN(RMSNorm(x_{l+1}))
$$

---

## 12. 模型主体：ZzMindModel

### 12.1 初始化

```python
class ZzMindModel(nn.Module):
    def __init__(self, config: ZzMindConfig):
        self.embd_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([ZzMindBlock(l, config) for l in range(num_layers)])
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # 预计算 RoPE
        freqs_cos, freqs_sin = precompute_freqs(
            dim=head_dim,
            end=max_position_embeddings,
            rope_base=rope_theta,
            rope_scaling=rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

`register_buffer` 将 cos/sin 注册为模型的 buffer（不是参数，不需要梯度），`persistent=False` 表示不保存到 checkpoint 中（因为可以随时重新计算）。

### 12.2 Forward

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
    batch_size, seq_len = input_ids.shape

    # 处理 past_key_values
    # past_key_values 类型: List[Tuple[K, V]]，每层一个 (K, V) 元组
    # hasattr(past_key_values, "layers")：兼容某些库返回的特殊对象格式
    if hasattr(past_key_values, "layers"):
        past_key_values = None

    past_key_values = past_key_values or [None] * len(self.layers)

    # start_pos: 已缓存序列的长度
    # 第 1 次调用时 past_key_values[0] 为 None → start_pos = 0
    # 第 2 次调用时 past_key_values[0][0].shape[1] = S_prompt → start_pos = S_prompt
    start_pos = (
        past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
    )

    # Embedding + dropout
    hidden_states = self.dropout(self.embd_tokens(input_ids))
    # (B, S, hidden_size)

    # 取出对应位置的 RoPE
    # 第 1 次: freqs_cos[0:S_prompt]
    # 第 2 次: freqs_cos[S_prompt:S_prompt+1]（只取新 token 的位置编码）
    position_embeddings = (
        self.freqs_cos[start_pos : start_pos + seq_len],
        self.freqs_sin[start_pos : start_pos + seq_len],
    )

    # 逐层传播
    presents = []
    for layer, past_key_value in zip(self.layers, past_key_values):
        hidden_states, present = layer(
            hidden_states,
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        presents.append(present)

    # 最终归一化
    hidden_states = self.norm(hidden_states)

    # 收集所有层的 MoE 辅助损失
    aux_loss = sum([layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, MoEFeedForward)],
                   hidden_states.new_zeros(1).squeeze())

    return hidden_states, presents, aux_loss
```

---

## 13. 语言模型：ZzMindForCausalLM

### 13.1 结构

```python
class ZzMindForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config: ZzMindConfig):
        self.model = ZzMindModel(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.model.embd_tokens.weight = self.lm_head.weight  # 权重共享
```

**权重共享（Tie Weights）**：Embedding 和 LM Head 共享同一份权重矩阵。

**原因**：
- 减少参数量（vocab_size × hidden_size 是很大的矩阵）
- 理论上 Embedding 和输出投影是对偶操作：Embedding 将 token ID → 向量，LM Head 将向量 → token 概率，共享权重有归纳偏置
- GPT-2、LLaMA 等主流模型都采用此设计

**注意**：这里通过直接赋值 `self.model.embd_tokens.weight = self.lm_head.weight` 实现共享。PyTorch 的 nn.Parameter 是引用类型，所以两个层指向同一份内存。

### 13.2 Forward

```python
def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False, logits_to_keep=0):
    # 1. 经过模型主体
    hidden_states, past_key_values, aux_loss = self.model(...)

    # 2. 计算 logits
    # logits_to_keep: 控制计算哪些位置的 logits，提高效率
    #   - int 类型:
    #       logits_to_keep=0  →  计算全部位置（训练时）
    #       logits_to_keep=1  →  只算最后一个位置（推理时预测下一个 token）
    #   - torch.Tensor 类型: 自定义索引，计算指定位置的 logits
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    # logits: (B, S, vocab_size)

    # 3. 计算损失（训练时）
    loss = None
    if labels is not None:
        # ignore_index=-100: labels 中值为 -100 的位置不参与损失计算（用于 padding）
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)
    output.aux_loss = aux_loss
    return output
```

**aux_loss 的处理**：

MoE 的负载均衡辅助损失 `aux_loss` 被附加到输出对象上。训练时，外部训练循环需要将主损失和辅助损失相加：

```python
total_loss = outputs.loss + outputs.aux_loss
```

如果模型没有启用 MoE（`use_moe=False`），则 `aux_loss = 0`。

**为什么 shift？**

语言模型的目标是预测下一个 token：

```
输入:    ["我", "喜欢", "吃", "苹果"]
目标:    ["喜欢", "吃", "苹果", "</s>"]

即：用 logits[..., :-1, :] 预测 labels[..., 1:]
     位置 0 的预测 → 位置 1 的标签
     位置 1 的预测 → 位置 2 的标签
     位置 2 的预测 → 位置 3 的标签
```

---

## 14. 完整数据流图

### 14.1 训练时的数据流

```
input_ids: (B, S)
    │
    ▼
Embedding: (B, S) → (B, S, D=512)
    │
    ▼ Dropout
(B, S, 512)
    │
    ├─── ZzMindBlock × 8 ───┐
    │                        │
    │   RMSNorm ──→ Attention ──→ +残差
    │      │          │
    │      │     ┌────┴────┐
    │      │     │ q_proj  │ (B,S,512)→(B,S,8,64)
    │      │     │ k_proj  │ (B,S,512)→(B,S,2,64)
    │      │     │ v_proj  │ (B,S,512)→(B,S,2,64)
    │      │     │ RoPE    │ (B,S,H,64)
    │      │     │ FlashAttn│ (B,H,S,S) → (B,H,S,64)
    │      │     │ o_proj  │ (B,S,512)
    │      │     └────┬────┘
    │      │          │
    │   RMSNorm ──→ FFN/MoE ──→ +残差
    │      │          │
    │      │     ┌────┴────┐
    │      │     │ Dense   │ gate/up/down 标准 SwiGLU
    │      │     │   或    │
    │      │     │ MoE     │ gate → topk_idx/weight → 选 K 个专家
    │      │     │         │   experts[i](x[mask_i]) → 加权求和
    │      │     │         │   + shared_experts(identity)
    │      │     │         │ → (B,S,512) + aux_loss
    │      │     └────┬────┘
    │      │          │
    │      └──────────┘
    │            │
    └────────────┘
          │
    RMSNorm
          │
    LM Head: (B, S, 512) → (B, S, 6400)
          │
    CrossEntropy Loss
```

### 14.2 推理时的数据流（带 KV Cache）

```
第 1 步（prompt）:
input_ids: (B, S_prompt)
    │
    ▼
Embedding → Block × 8 → LM Head → logits[:, -1, :] → next_token
    │                           │
    └───── KV Cache ────────────┘   (缓存每层的 K, V)

第 2 步（生成第 1 个新 token）:
input_ids: (B, 1)   ← 只输入新 token！
    │
    ▼
Embedding → Block × 8 → LM Head → logits[:, -1, :] → next_token
    │                           │
    └──→ cat(KV_cache, new_KV) ─┘   (拼接旧缓存 + 新计算的 K, V)

第 3~N 步: 同上，每次只处理 1 个 token
```

**KV Cache 节省的计算量**：

- 第 $t$ 步标准计算：$O(t^2 \cdot d)$
- 第 $t$ 步用 KV Cache：$O(t \cdot d)$
- 总序列长度 $N$：从 $O(N^3)$ 降到 $O(N^2)$

---

## 15. 训练与推理的区别

| 方面 | 训练 | 推理 |
|------|------|------|
| **输入** | 完整序列 (B, S) | 第 1 步：prompt；之后每次 1 个 token |
| **KV Cache** | 不用 | 必须启用 |
| **注意力** | Flash Attention | 手动计算（带 Cache） |
| **因果掩码** | `is_causal=True` | 手动加三角掩码 |
| **MoE** | `repeat_interleave` 实现 | `moe_infer` 批量实现 |
| **梯度** | 需要 | `@torch.no_grad()` |
| **logits_to_keep** | 0（计算全部位置） | 1（只算最后一个位置） |
| **aux_loss** | 计算并加到主损失上 | 为 0 |
| **损失** | CrossEntropy | 不需要 |
| **输出** | loss + logits | logits（取最后一个位置的） |

---

## 参数数量估算

**Attention 每层（GQA）**：

| 投影 | 形状 | 参数量 |
|------|------|--------|
| q_proj | $(D, H \times d) = (512, 512)$ | 262K |
| k_proj | $(D, H_{kv} \times d) = (512, 128)$ | 65.5K |
| v_proj | $(D, H_{kv} \times d) = (512, 128)$ | 65.5K |
| o_proj | $(H \times d, D) = (512, 512)$ | 262K |
| **总计** | $2D^2 + 2D \times H_{kv} \times d$ | **655K** |

注意：GQA 下不是 MHA 的 $4D^2$，而是约 $2.5D^2$（因为 $H_{kv} < H$）。

**FFN 每层**：

| 投影 | 形状 | 参数量 |
|------|------|--------|
| gate_proj | $(D, D_{inter}) = (512, 1408)$ | 721K |
| up_proj | $(D, D_{inter}) = (512, 1408)$ | 721K |
| down_proj | $(D_{inter}, D) = (1408, 512)$ | 721K |
| **总计** | $3 \times D \times D_{inter}$ | **2.16M** |

**每层总计（Dense）**：655K + 2.16M = **2.82M**

**MoE 每层**：4 个路由专家 × 2.16M + gate (4 × 512) + 1 个共享专家 × 2.16M = **约 10.8M**

注意：gate 的权重 `W_gating` 形状为 `(n_routed_experts, hidden_size) = (4, 512)`，参数量仅 2048，相对于专家网络可忽略。

**Embedding**：$V \times D = 6400 \times 512 =$ **3.28M**

**8 层 Dense 模型总计**：3.28M + 8 × 2.82M = **约 2590 万参数**

| 模型 | 参数量 | 说明 |
|------|--------|------|
| **MiniMind (Dense)** | ~26M | 本项目 |
| **MiniMind (MoE)** | ~90M | 启用 MoE 后 |
| GPT-2 | 1.5B | |
| LLaMA-7B | 7B | |
| LLaMA-70B | 70B | |

这是一个非常小的模型（26M Dense），适合学习和实验。
