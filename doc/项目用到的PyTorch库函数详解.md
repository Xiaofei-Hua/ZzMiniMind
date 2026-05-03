# MiniMind 项目用到的 PyTorch 库函数详解

> 梳理项目中每一个用到的库函数：它是什么、怎么用、在什么场景下用、在本项目中具体做了什么。

---

## 目录

1. [torch 基础张量操作](#1-torch-基础张量操作)
2. [torch 数学函数](#2-torch-数学函数)
3. [torch 张量变形与索引](#3-torch-张量变形与索引)
4. [torch.nn 模块](#4-torchnn-模块)
5. [torch.nn.functional 函数式 API](#5-torchnnfunctional-函数式-api)
6. [torch.nn.init 初始化](#6-torchnninit-初始化)
7. [math 标准库](#7-math-标准库)
8. [transformers 库](#8-transformers-库)
9. [Python 内置与 typing](#9-python-内置与-typing)

---

## 1. torch 基础张量操作

### 1.1 `torch.arange(start, end, step)` — 生成等差序列

**作用**：生成一个一维张量，包含从 `start` 开始到 `end` 结束（不包含）、步长为 `step` 的序列。

**签名**：
```python
torch.arange(start=0, end, step=1, *, dtype=None, device=None)
```

**本项目用例**：
```python
# 生成维度索引 [0, 2, 4, ..., d-2]
torch.arange(0, dim, 2)   # 形状: (d//2,)

# 生成位置索引 [0, 1, 2, ..., end-1]
torch.arange(end, device=freqs.device)   # 形状: (end,)
```

**场景**：生成索引、位置编码中的位置编号、维度对编号等。

---

### 1.2 `torch.ones(*size)` — 全 1 张量

**作用**：创建一个指定形状的张量，所有元素为 1。

**本项目用例**：
```python
# 创建 RMSNorm 的可学习权重，初始值为 1
torch.ones(dim)   # 形状: (dim,)
```

**场景**：初始化权重（如 LayerNorm/RMSNorm 的 gamma），创建掩码等。

---

### 1.3 `torch.empty(*size)` — 未初始化张量

**作用**：创建一个指定形状的张量，**不初始化元素值**（内容是内存中的随机值，不可读）。

**签名**：
```python
torch.empty(*size, dtype=None, device=None)
```

**本项目用例**：
```python
# MoE 门控权重初始化（未初始化，后续调用 reset_parameters）
torch.empty((n_routed_experts, gating_dim))

# MoE 训练时的输出缓冲区
y = torch.empty_like(x, dtype=x.dtype)
```

**与 `torch.zeros` 的区别**：`empty` 不填充任何值，分配内存后立即返回，速度更快。适用于马上要覆盖写入的场景（如上面的 `y` 缓冲区）。

---

### 1.4 `torch.zeros(*size)` — 全 0 张量

**作用**：创建一个指定形状的张量，所有元素为 0。

**本项目用例**：
```python
# MoE 推理时的累加缓存
expert_cache = torch.zeros_like(x)

# 辅助损失统计：每个序列中每个专家的负载
torch.zeros(bsz, n_routed_experts, device=hidden_states.device)
```

**场景**：累加操作的起始值、统计计数器的初始值。

---

### 1.5 `torch.full(size, fill_value)` — 填充指定值

**作用**：创建一个指定形状的张量，所有元素为 `fill_value`。

**本项目用例**：
```python
# 创建因果掩码的上三角部分（对角线上方全为 -inf）
torch.full((sql, sql), float("-inf"), device=scores.device)
```

**场景**：创建掩码矩阵、初始化特定值的张量。

---

### 1.6 `torch.cat(tensors, dim)` — 张量拼接

**作用**：沿指定维度拼接多个张量。

**签名**：
```python
torch.cat(tensors, dim=0)
```

**本项目用例**：
```python
# KV Cache：将历史 K/V 与新计算的 K/V 沿序列维度拼接
xk = torch.cat([past_key_value[0], xk], dim=1)

# RoPE：将 cos/sin 复制两份沿最后一维拼接
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)

# rotate_half：将后半截取负后与前半拼接
rotate_half(x) = torch.cat((-x[..., d//2:], x[..., :d//2]), dim=-1)
```

**注意**：拼接的维度上，各张量的其他维度大小必须相同。

---

## 2. torch 数学函数

### 2.1 `torch.rsqrt(x)` — 平方根倒数

**作用**：计算 $\frac{1}{\sqrt{x}}$，即 $x^{-0.5}$。

**数学**：
```
rsqrt(x) = 1 / sqrt(x)
```

**本项目用例**：
```python
# RMSNorm 的核心计算
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
# 等价于: x / sqrt(mean(x^2) + eps)
```

**为什么用 `rsqrt` 而不是 `1/sqrt`？**

- **精度**：`rsqrt` 是硬件原生指令（在 CUDA 上），比先 `sqrt` 再除法精度更高
- **速度**：单条指令完成，不需要中间存储

---

### 2.2 `torch.pow(x, exponent)` — 幂运算

**作用**：计算 $x^{exponent}$，逐元素。

**本项目用例**：
```python
# RMSNorm：计算 x^2
x.pow(2)   # 等价于 x ** 2，也等价于 x * x

# RoPE 频率计算：base ^ (2i/d)
rope_base ** (torch.arange(0, dim, 2).float() / dim)
```

---

### 2.3 `torch.mean(x, dim, keepdim)` — 求均值

**作用**：沿指定维度求均值。

**参数**：
- `dim`：求均值的维度
- `keepdim=True`：保持维度（结果形状中该维度为 1，便于广播）

**本项目用例**：
```python
# RMSNorm：求每个样本在所有维度上的均方值
x.pow(2).mean(-1, keepdim=True)   # 形状: (B, S, 1)
# keepdim=True 使结果可以与 x (B, S, D) 广播
```

**广播的关键**：如果不加 `keepdim=True`，`(B, S, D)` → `mean(-1)` 会变成 `(B, S)`，无法与 `(B, S, D)` 做逐元素运算。`keepdim=True` 保持 `(B, S, 1)`，利用 PyTorch 广播机制自动扩展到 `(B, S, D)`。

---

### 2.4 `torch.cos(x)` / `torch.sin(x)` — 三角函数

**作用**：逐元素计算余弦/正弦。

**本项目用例**：
```python
# RoPE：根据旋转角度计算 cos 和 sin
freqs = torch.outer(t, freqs).float()   # (E, d/2)
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
```

**注意**：输入是弧度制，不是角度制。

---

### 2.5 `torch.clamp(x, min, max)` — 截断

**作用**：将张量元素限制在 `[min, max]` 范围内，超出部分截断到边界。

**本项目用例**：
```python
# YaRN：将 Ramp 因子限制在 [0, 1]
ramp = torch.clamp(
    (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
    0,
    1,
)
```

**场景**：限制数值范围、确保梯度稳定、实现分段函数。

---

### 2.6 `torch.outer(a, b)` — 外积

**作用**：计算两个一维向量的外积。

**数学**：
```
outer(a, b)[i, j] = a[i] * b[j]
```

**本项目用例**：
```python
# RoPE：将位置索引与频率相乘，得到每个位置的旋转角度
# t: (E,), freqs: (d/2,)
freqs = torch.outer(t, freqs)   # (E, d/2)
```

**结果**：`freqs[pos, j] = pos * θ_j`，即位置 pos、第 j 个维度对的旋转角度。

---

## 3. torch 张量变形与索引

### 3.1 `tensor.view(*shape)` — 重塑形状

**作用**：改变张量的形状，**不复制数据**（只是改变张量的解读方式）。

**本项目用例**：
```python
# Attention：将 (B, S, H*d) 重塑为 (B, S, H, d)
xq = xq.view(bsz, sql, self.n_local_heads, self.head_dim)

# 输出：将 (B, H, S, d) 重塑回 (B, S, H*d)
output = output.transpose(1, 2).reshape(bsz, sql, -1)
```

**与 `reshape` 的区别**：
- `view`：要求数据在内存中连续，返回视图（共享内存）
- `reshape`：不要求连续，必要时复制数据

**规则**：新形状的元素总数必须与原形状相同。

---

### 3.2 `tensor.transpose(dim0, dim1)` — 交换维度

**作用**：交换两个维度。

**本项目用例**：
```python
# Attention：将 (B, S, H, d) 的 S 和 H 维度交换，便于矩阵乘法
xq = xq.transpose(1, 2)   # (B, H, S, d)

# 输出：将 (B, H, S, d) 交换回 (B, S, H, d)
output = output.transpose(1, 2)
```

**为什么需要 transpose？**

注意力计算需要 `Q @ K^T`，要求 Q 和 K 的最后两个维度是 `(seq, d)` 和 `(d, seq)`。而多头拆分后的形状是 `(B, S, H, d)`，矩阵乘法是作用于最后两个维度的。所以需要先 transpose 成 `(B, H, S, d)`，这样最后两个维度就是 `(S, d)`，可以直接做矩阵乘法。

---

### 3.3 `tensor.unsqueeze(dim)` — 插入维度

**作用**：在指定位置插入一个大小为 1 的新维度。

**本项目用例**：
```python
# RoPE：为 cos/sin 插入广播维度
# cos: (S, d) → unsqueeze(1) → (S, 1, d)
# 然后与 q: (B, S, H, d) 广播相乘
q_embed = q * cos.unsqueeze(unsqueeze_dim)
```

**广播机制**：`(S, 1, d)` 与 `(B, S, H, d)` 相乘时：
- 第 0 维：1 → B（复制 B 份）
- 第 1 维：S 匹配 S
- 第 2 维：1 → H（复制 H 份）
- 第 3 维：d 匹配 d

---

### 3.4 `tensor.expand(*sizes)` — 广播扩展

**作用**：扩展张量到指定形状，**不复制数据**（只是改变 stride/步长）。

**本项目用例**：
```python
# GQA repeat_kv：将 KV head 从 H_kv 扩展到 H
torch.zeros(1, 3, 2, 4, 64).expand(1, 3, 2, 4, 64)   # 形状不变
x[:, :, :, None, :].expand(B, S, H_kv, R, d)         # 从 (B,S,H_kv,1,d) 扩展到 (B,S,H_kv,R,d)
```

**与 `repeat` 的区别**：
- `expand`：视图操作，不分配新内存，只是改变 stride 让同一个数据被"看到"多次
- `repeat`：实际复制数据，分配新内存

**限制**：只能扩展大小为 1 的维度到更大的大小，不能改变非 1 维度的大小。

---

### 3.5 `tensor.contiguous()` — 内存连续化

**作用**：确保张量在内存中是连续存储的。如果张量不连续（如 transpose、permute、expand 后的结果），会复制数据到连续的内存区域。

**本项目用例**：
```python
# 交叉熵损失计算前确保内存连续
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
```

**为什么需要 `contiguous`？**

某些操作（如 `view`）要求张量内存连续。`transpose`、`permute`、`narrow` 等操作会改变 stride 使张量不连续：
```python
x = torch.randn(2, 3)
y = x.transpose(0, 1)   # y 不连续
y.view(-1)              # 报错！RuntimeError: view size is not compatible
y.contiguous().view(-1) # 正确
```

---

### 3.6 `tensor.type_as(other)` — 类型转换

**作用**：将张量转换为与 `other` 张量相同的数据类型。

**本项目用例**：
```python
# RMSNorm：计算完 float32 后转回原始精度
self.weight * self._norm(x.float()).type_as(x)
# x 可能是 fp16/bf16，计算时用 float32 保证精度，最后转回
```

**等价于**：`x.to(other.dtype)`

---

### 3.7 `tensor.mul_(other)` — 原地乘法

**作用**：`x.mul_(y)` 等价于 `x = x * y`，但**原地修改**，不分配新内存。

**本项目用例**：
```python
# MoE 推理：原地加权
expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
# 等价于: expert_out = expert_out * weights，但原地操作
```

**带下划线的方法（in-place）**：PyTorch 中方法名带 `_` 的都是原地操作（如 `add_`、`mul_`、`div_`），不创建新张量，直接修改自身。节省内存但会丢失原始值。

---

### 3.8 `tensor.scatter_add_(dim, index, src)` — 散点累加

**作用**：根据 `index` 将 `src` 的元素累加到 `self` 的指定位置。

**签名**：
```python
self.scatter_add_(dim, index, src)
# self[dim][index[i][j]] += src[i][j]
```

**本项目用例**：
```python
# MoE 推理：将各专家的输出按原始 token 位置累加回结果
expert_cache.scatter_add_(
    0,
    exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
    expert_out
)
```

**直观理解**：`index` 告诉你要把 `src` 的每个元素放到 `self` 的哪个位置，`dim` 告诉你在哪个维度上放。

---

### 3.9 `tensor.argsort()` — 排序索引

**作用**：返回将张量排序后的索引序列。

**本项目用例**：
```python
# MoE 推理：按专家索引排序 token，使相同专家的 token 聚集在一起
idxs = flat_expert_indices.argsort()
# 结果: [0, 0, 0, 1, 1, 2, 2, 2, ...]
```

**场景**：分组操作（如将相同路由目标的 token 聚集后批量处理）。

---

### 3.10 `tensor.bincount()` — 统计频次

**作用**：统计一维整数张量中每个值出现的次数。

**本项目用例**：
```python
# MoE 推理：统计每个专家被分配到的 token 数量
tokens_per_expert = flat_expert_indices.bincount()
# 结果: [专家0的token数, 专家1的token数, ...]
```

**限制**：输入必须是一维非负整数张量。

---

### 3.11 `tensor.repeat_interleave(repeats, dim)` — 重复元素

**作用**：沿指定维度重复张量的每个元素。

**本项目用例**：
```python
# MoE 训练：每个 token 复制 K 份（因为有 K 个专家）
x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
# (B*S, D) → (B*S*K, D)
```

**与 `repeat` 的区别**：
- `repeat`：重复整个张量
- `repeat_interleave`：重复每个元素

```python
torch.tensor([1, 2]).repeat(2)        # [1, 2, 1, 2]
torch.tensor([1, 2]).repeat_interleave(2)   # [1, 1, 2, 2]
```

---

## 4. torch.nn 模块

### 4.1 `nn.Parameter(tensor)` — 可学习参数

**作用**：将张量包装为模型的可学习参数，会被 `model.parameters()` 收集，参与梯度计算和优化器更新。

**本项目用例**：
```python
# RMSNorm 的可学习缩放权重
self.weight = nn.Parameter(torch.ones(dim))

# MoE 门控权重
self.weight = nn.Parameter(torch.empty((n_routed_experts, gating_dim)))
```

**与 `nn.Buffer` 的区别**：
- `nn.Parameter`：参与梯度计算，会被优化器更新
- `nn.Buffer`（通过 `register_buffer` 注册）：不计算梯度，不更新，但会随模型保存/加载

---

### 4.2 `nn.Embedding(num_embeddings, embedding_dim)` — 词嵌入层

**作用**：将离散的整数索引（token ID）映射为连续的密集向量。

**本项目用例**：
```python
# 将 token ID 映射为 hidden_size 维向量
self.embd_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
# 输入: (B, S) 的整数张量，输出: (B, S, hidden_size) 的浮点张量
```

**原理**：内部是一个 `(vocab_size, hidden_size)` 的可学习权重矩阵。输入 `id` 时，相当于 `weight[id]` 查找。

---

### 4.3 `nn.Linear(in_features, out_features, bias=True)` — 线性层

**作用**：实现线性变换 $y = xW^T + b$。

**本项目用例**：
```python
# q_proj: 将 hidden_size 映射到 H*d
self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

# lm_head: 将 hidden_size 映射到 vocab_size
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
```

**参数**：
- `in_features`：输入维度
- `out_features`：输出维度
- `bias`：是否加偏置项（本项目所有 Linear 都设 `bias=False`，减少参数量）

**权重形状**：`W` 的形状是 `(out_features, in_features)`，即 `(输出维度, 输入维度)`。

---

### 4.4 `nn.Dropout(p)` — Dropout 层

**作用**：以概率 `p` 随机将输入元素置为 0，用于防止过拟合。

**本项目用例**：
```python
self.attn_dropout = nn.Dropout(args.dropout)   # 对注意力权重 dropout
self.resid_dropout = nn.Dropout(args.dropout)   # 对残差输出 dropout
self.dropout = nn.Dropout(config.dropout)        # 对 Embedding dropout
```

**行为**：
- **训练时**（`model.train()`）：以概率 `p` 随机置 0，剩余元素乘以 `1/(1-p)` 保持期望不变
- **推理时**（`model.eval()`）：不 dropout，直接原样通过

---

### 4.5 `nn.Module` / `nn.ModuleList` — 模块容器

**`nn.Module`**：所有神经网络模块的基类。自定义层必须继承它。

**`nn.ModuleList`**：一个列表容器，里面的模块会被自动注册到模型中（会被 `model.parameters()` 收集）。

**本项目用例**：
```python
# 存储 N 个 Transformer Block
self.layers = nn.ModuleList([ZzMindBlock(l, config) for l in range(num_layers)])

# 存储多个专家 FFN
self.experts = nn.ModuleList([FeedForward(config) for _ in range(n_routed)])
```

**为什么不用 Python list？**

```python
# ❌ 错误：普通 list 中的模块不会被 PyTorch 识别
self.layers = [ZzMindBlock(l, config) for l in range(num_layers)]

# ✅ 正确：ModuleList 中的模块会自动注册
self.layers = nn.ModuleList([ZzMindBlock(l, config) for l in range(num_layers)])
```

普通 `list` 中的模块不会被 `model.parameters()` 收集，也不会随模型保存/加载。`nn.ModuleList` 解决了这个问题。

---

### 4.6 `model.register_buffer(name, tensor, persistent=True)` — 注册 Buffer

**作用**：将张量注册为模型的 buffer（非参数），会随模型保存/加载。

**本项目用例**：
```python
# 注册预计算的 RoPE cos/sin 为 buffer（不保存到 checkpoint）
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

**与 `nn.Parameter` 的区别**：
| | Parameter | Buffer |
|---|---|---|
| 计算梯度 | 是 | 否 |
| 优化器更新 | 是 | 否 |
| 保存到 checkpoint | 是 | `persistent=True` 时保存 |
| 用途 | 可学习权重 | 预计算常量、统计量等 |

---

### 4.7 `@torch.no_grad()` — 禁用梯度计算

**作用**：装饰器，标记函数内部不计算梯度，节省内存和计算。

**本项目用例**：
```python
@torch.no_grad()
def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
    # 推理阶段不需要梯度
    ...
```

**与 `torch.set_grad_enabled(False)` 的区别**：
- `@torch.no_grad()`：只在函数内部禁用，不影响外部
- `torch.set_grad_enabled(False)`：全局设置，影响后续所有操作

---

## 5. torch.nn.functional 函数式 API

### 5.1 `F.scaled_dot_product_attention(query, key, value, ...)` — Flash Attention

**作用**：高效计算缩放点积注意力，自动选择最优实现（Flash Attention / Memory-Efficient Attention / Math）。

**签名**：
```python
F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None
)
```

**本项目用例**：
```python
output = F.scaled_dot_product_attention(
    xq, xk, xv,
    dropout_p=self.dropout if self.training else 0.0,
    is_causal=True,
)
```

**`is_causal=True`**：自动应用因果掩码（上三角为 -inf），无需手动构造掩码矩阵。

**`scale`**：默认是 `1/sqrt(d_k)`，可自定义。

---

### 5.2 `F.softmax(x, dim)` — Softmax

**作用**：沿指定维度计算 softmax，将数值转换为概率分布。

**数学**：
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

**本项目用例**：
```python
# 注意力权重归一化
scores = F.softmax(scores.float(), dim=-1)

# MoE 门控得分
scores = logits.softmax(dim=-1)
```

**注意**：`tensor.softmax(dim)` 和 `F.softmax(tensor, dim)` 是等价的，前者是实例方法，后者是函数式 API。

---

### 5.3 `F.linear(input, weight, bias=None)` — 线性变换

**作用**：实现 $y = xW^T + b$，与 `nn.Linear` 的前向计算等价。

**本项目用例**：
```python
# MoE 门控：直接用函数式 API 做线性变换
logits = F.linear(hidden_states, self.weight, None)
# 等价于: hidden_states @ self.weight.T
```

**与 `nn.Linear` 的区别**：
- `nn.Linear`：封装了权重和偏置，是模块
- `F.linear`：只有计算逻辑，需要手动传入权重和偏置，是函数

---

### 5.4 `F.cross_entropy(input, target, ...)` — 交叉熵损失

**作用**：计算分类任务的交叉熵损失，内部自动包含 `log_softmax`。

**签名**：
```python
F.cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean')
```

**本项目用例**：
```python
loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),   # (B*S, vocab_size)
    shift_labels.view(-1),                # (B*S,)
    ignore_index=-100,
)
```

**注意**：`input` 是未归一化的 logits（不需要先 softmax），`target` 是类别索引（整数）。

**`ignore_index=-100`**：`target` 中值为 `-100` 的位置不参与损失计算，用于处理 padding。

---

### 5.5 `F.one_hot(x, num_classes)` — One-Hot 编码

**作用**：将整数索引转换为 one-hot 编码。

**本项目用例**：
```python
# MoE 辅助损失：将专家索引转为 one-hot
mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=n_routed_experts)
# 结果: (N, n_experts)，每行只有一个 1
```

---

## 6. torch.nn.init 初始化

### 6.1 `init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')` — Kaiming 初始化

**作用**：使用 Kaiming（He）初始化方法初始化张量，适合 ReLU 及其变体激活函数。

**本项目用例**：
```python
# MoE 门控权重初始化
init.kaiming_uniform_(self.weight, a=math.sqrt(5))
```

**原理**：从均匀分布 $U(-bound, bound)$ 采样，其中：
```
bound = sqrt(6 / ((1 + a^2) * fan_in))
```

- `fan_in`：输入维度
- `a`：激活函数的负斜率（LeakyReLU 参数，默认 0 对应 ReLU）

**为什么用 `a=math.sqrt(5)`？**

这是 PyTorch `nn.Linear` 默认的初始化参数。`math.sqrt(5)` 对应于 `nn.Linear` 中 `a=math.sqrt(5)` 的默认值，保持一致性。

---

## 7. math 标准库

### 7.1 `math.sqrt(x)` — 平方根

**本项目用例**：
```python
# Attention：缩放点积注意力的缩放因子
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

# Kaiming 初始化参数
init.kaiming_uniform_(self.weight, a=math.sqrt(5))
```

### 7.2 `math.floor(x)` / `math.ceil(x)` — 向下/向上取整

**本项目用例**：
```python
# YaRN：计算高频/低频切分点
low = max(math.floor(inv_dim(beta_fast)), 0)
high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
```

### 7.3 `math.log(x)` — 自然对数

**本项目用例**：
```python
# YaRN：波长到维度索引的映射
inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
```

---

## 8. transformers 库

### 8.1 `ACT2FN` — 激活函数字典

**作用**：将字符串名称映射到对应的激活函数。

**本项目用例**：
```python
from transformers.activations import ACT2FN

self.act_fn = ACT2FN[config.hidden_act]   # "silu" → nn.SiLU()
```

**常用映射**：
| 字符串 | 对应函数 |
|--------|---------|
| `"silu"` | `nn.SiLU()` |
| `"gelu"` | `nn.GELU()` |
| `"relu"` | `nn.ReLU()` |
| `"swish"` | `nn.SiLU()` |

### 8.2 `PretrainedConfig` — 配置基类

**作用**：Hugging Face Transformers 的配置基类，提供序列化、反序列化等功能。

**本项目用例**：
```python
class ZzMindConfig(PretrainedConfig):
    model_type = "zzmind"
```

### 8.3 `PreTrainedModel` / `GenerationMixin` — 模型基类

**作用**：
- `PreTrainedModel`：提供 `from_pretrained`、`save_pretrained` 等标准接口
- `GenerationMixin`：提供 `generate()` 方法，支持贪心搜索、束搜索、温度采样等

**本项目用例**：
```python
class ZzMindForCausalLM(PreTrainedModel, GenerationMixin):
    ...
```

### 8.4 `CausalLMOutputWithPast` — 输出结构

**作用**：标准化的模型输出结构，包含 loss、logits、past_key_values 等字段。

**本项目用例**：
```python
output = CausalLMOutputWithPast(
    loss=loss,
    logits=logits,
    past_key_values=past_key_values,
    hidden_states=hidden_states,
)
output.aux_loss = aux_loss   # 附加自定义字段
```

---

## 9. Python 内置与 typing

### 9.1 `typing` 类型注解

**本项目用到的类型**：

| 类型 | 含义 | 示例 |
|------|------|------|
| `Optional[T]` | `T` 或 `None` | `Optional[Tensor]` |
| `Tuple[T1, T2]` | 长度为 2 的元组 | `Tuple[Tensor, Tensor]` |
| `List[T]` | 列表 | `List[Tuple[Tensor, Tensor]]` |
| `Union[T1, T2]` | T1 或 T2 | `Union[int, Tensor]` |

**作用**：
- 代码可读性：明确参数和返回值的类型
- IDE 支持：自动补全、类型检查
- 文档生成：自动生成 API 文档

**注意**：Python 的类型注解**不运行时检查**，只是提示。真正运行时不影响程序行为。

---

## 速查表：本项目核心操作一览

| 操作 | 函数/方法 | 本项目场景 |
|------|----------|-----------|
| 生成索引序列 | `torch.arange` | RoPE 维度索引、位置索引 |
| 拼接 | `torch.cat` | KV Cache、RoPE cos/sin、rotate_half |
| 外积 | `torch.outer` | RoPE 旋转角度矩阵 |
| 重塑 | `tensor.view` | 多头拆分、输出合并 |
| 交换维度 | `tensor.transpose` | (B,S,H,d) ↔ (B,H,S,d) |
| 广播 | `tensor.unsqueeze` + 自动广播 | RoPE 应用到 Q/K |
| 视图扩展 | `tensor.expand` | GQA repeat_kv（不复制内存） |
| 排序 | `tensor.argsort` | MoE 推理 token 分组 |
| 频次统计 | `tensor.bincount` | MoE 专家负载统计 |
| 散点累加 | `tensor.scatter_add_` | MoE 推理结果汇总 |
| 元素重复 | `tensor.repeat_interleave` | MoE 训练 token 复制 |
| 线性变换 | `nn.Linear` / `F.linear` | Q/K/V/O 投影、门控得分 |
| 注意力 | `F.scaled_dot_product_attention` | Flash Attention 快速路径 |
| 归一化 | `F.softmax` | 注意力权重、门控得分 |
| 损失 | `F.cross_entropy` | 语言模型训练损失 |
| Dropout | `nn.Dropout` | 防止过拟合 |
| 初始化 | `init.kaiming_uniform_` | MoE 门控权重 |
