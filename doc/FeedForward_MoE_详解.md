# FeedForward & MoEGate 模块全面解析

> 面向对象：有算法竞赛基础、刚学完 Transformer 理论的同学。
> 目标：从数学原理到代码实现，不跳步，完整理解 `FeedForward` 和 `MoEGate` 的每一行。

---

## 目录

1. [先修知识：为什么需要 FeedForward 层](#1-先修知识为什么需要-feedforward-层)
2. [从标准 FFN 到 SwiGLU 激活函数](#2-从标准-ffn-到-swiglu-激活函数)
3. [FeedForward 逐行解析](#3-feedforward-逐行解析)
4. [MoE (Mixture of Experts) 背景知识](#4-moe-mixture-of-experts-背景知识)
5. [MoE 的门控路由机制](#5-moe-的门控路由机制)
6. [负载均衡辅助损失 (Auxiliary Loss)](#6-负载均衡辅助损失-auxiliary-loss)
7. [MoEGate 逐行解析](#7-moegate-逐行解析)
8. [代码 Bug 分析](#8-代码-bug-分析)
9. [数据流全图](#9-数据流全图)
10. [关键公式速查表](#10-关键公式速查表)

---

## 1. 先修知识：为什么需要 FeedForward 层

### 1.1 注意力层已经能提取特征了，为什么还需要 FFN？

在 Transformer 的每一层中，结构是：`Self-Attention → Add&Norm → FFN → Add&Norm`。两者各司其职：

| 层 | 功能 | 类比 |
|---|---|---|
| **Attention** | "关联"——找不同 token 之间的关系 | 阅读时做笔记、联想 |
| **FFN** | "变换"——对单个 token 的特征做非线性变换 | 对笔记做深加工、提炼 |

Attention 做的是**跨位置**的信息交互（不同位置的 token 之间交换信息），而 FFN 做的是**单个位置内的特征变换**（不与其他位置交互，只在每个 token 的特征维度上做非线性映射）。

### 1.2 FFN 的数学本质

FFN 就是一个两层的全连接网络：

$$
\text{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2
$$

其中：
- $\sigma$ 是非线性激活函数（如 ReLU、GELU）
- $W_1$ 将维度从 $d$ 扩展到 $d_{inter}$（中间维度更大）
- $W_2$ 将维度从 $d_{inter}$ 压缩回 $d$

为什么先扩展再压缩？

> **直觉**：从 512 维映射到 2048 维，等于给模型更多的"表达空间"，再压缩回 512 维时，可以选择性地保留最重要的信息。
>
> **数学原因**：更高的中间维度可以近似更复杂的函数。两层网络 + 非线性 = 万能近似器。

---

## 2. 从标准 FFN 到 SwiGLU 激活函数

### 2.1 标准 FFN (ReLU/GELU)

最早的 Transformer 论文中，FFN 用的是 ReLU：

```python
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

后来大家发现 GELU (高斯误差线性单元) 效果更好：

```python
GELU(x) = x * Φ(x)   # Φ 是标准正态分布的 CDF
```

### 2.2 GLU (Gated Linear Unit) 门控线性单元

GLU 的灵感来自 LSTM 的门控机制，核心思想是：**用门控来控制信息流**。

标准的 GLU：

$$
\text{GLU}(a, b) = a \odot \sigma(b)
$$

其中 $a$ 和 $b$ 是同一个输入经过不同线性变换得到的：

$$
\text{GLU}(x) = (x W) \odot \sigma(x V)
$$

- $xW$ 是"值"部分
- $\sigma(xV)$ 是"门"部分（0 到 1 之间的权重）
- $\odot$ 是逐元素相乘

> **直觉**：门控决定每个维度保留多少信息。如果门接近 0，该维度就被关闭；接近 1，就完全保留。这比单纯的 ReLU（要么全开要么全关）更灵活。

### 2.3 SwiGLU (Swish + GLU)

SwiGLU 是 GLU 的变体，把 sigmoid 门控换成了 Swish (也叫 SiLU) 激活函数：

$$
\text{SwiGLU}(x) = (x W) \odot \text{SiLU}(x V)
$$

其中：

$$
\text{SiLU}(x) = x \odot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

SiLU 的特点：
- 非单调（不是一直递增）
- 平滑（处处可导）
- 负半轴不完全截断，而是给很小的值
- 在大模型中被证明比 ReLU/GELU 效果更好

### 2.4 SwiGLU FFN 的结构

```python
# 门控路径
门控信号 = SiLU( gate_proj(x) )      # x → W_gate → SiLU → gate
# 值路径
值信号   = up_proj(x)                # x → W_up
# 相乘后降维
输出     = down_proj( 门控信号 * 值信号 )
```

三个线性层的维度关系：
- `gate_proj` : `(hidden_size, intermediate_size)` — 产生门控信号
- `up_proj`   : `(hidden_size, intermediate_size)` — 产生值信号
- `down_proj` : `(intermediate_size, hidden_size)` — 压缩回原始维度

### 2.5 intermediate_size 的选择

标准 Transformer 用 $d_{ff} = 4 \times d_{model}$。

SwiGLU 因为有两条路径同时参与计算，参数更多。为了控制参数量，通常取：

$$
d_{inter} = \frac{8}{3} \times d_{hidden}
$$

然后向上对齐到 64 的倍数（方便 GPU 并行）：

```python
intermediate_size = int(config.hidden_size * 8 / 3)
intermediate_size = 64 * ((intermediate_size + 63) // 64)  # 向上取整到64的倍数
```

---

## 3. FeedForward 逐行解析

```python
class FeedForward(nn.Module):
    def __init__(self, config: ZzConfig):
        super().__init__()

        # ---------- intermediate_size 自动计算 ----------
        if config.intermediate_size is None:
            # 先按 8/3 倍计算
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 向上取整到 64 的倍数（GPU kernel 优化，整数倍效率最高）
            config.intermediate_size = 64 * ((intermediate_size + 63) // 64)

        # ---------- gate_proj: 产生门控信号 ----------
        # 输入: hidden_size (512), 输出: intermediate_size (~1365)
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,     # 大模型惯例：不加 bias，减少参数量
        )

        # ---------- down_proj: 降维投影 ----------
        # 输入: intermediate_size, 输出: hidden_size
        self.down_proj = nn.Linear(
            config.intermediate_size, 
            config.hidden_size, 
            bias=False,
        )

        # ---------- up_proj: 产生值信号 ----------
        # 输入: hidden_size, 输出: intermediate_size
        self.up_proj = nn.Linear(
            config.hidden_size, 
            config.intermediate_size, 
            bias=False
        )

        # ---------- Dropout ----------
        self.dropout = nn.Dropout(config.dropout)

        # ---------- 激活函数 ----------
        # ACT2FN 是 transformers 库预定义的字典，"silu" 对应 nn.SiLU()
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # x 形状: (bsz, sql, hidden_size) 例: (2, 10, 512)

        # 门控路径: x → gate_proj → SiLU 激活
        gate = self.act_fn(self.gate_proj(x))    # (bsz, sql, intermediate_size)
        # gate_proj(x): (bsz, sql, intermediate_size)
        # act_fn (SiLU): 逐元素应用 SiLU(x) = x * sigmoid(x)

        # 值路径: x → up_proj
        up = self.up_proj(x)                      # (bsz, sql, intermediate_size)

        # 门控机制: 门控信号 * 值信号，逐元素相乘
        gated = gate * up                         # (bsz, sql, intermediate_size)
        # 这里实现的是: SiLU(x W_gate) ⊙ (x W_up)

        # 降维回原始维度
        output = self.down_proj(gated)            # (bsz, sql, hidden_size)

        # Dropout
        return self.dropout(output)
```

**形状变化总结**：

```
输入:     (bsz, sql, 512)
    │
    ├─ gate_proj ──→ (bsz, sql, 1365) ──→ SiLU ──→ gate: (bsz, sql, 1365)
    └─ up_proj   ──→ (bsz, sql, 1365) ──────────────→ up:   (bsz, sql, 1365)
                                                        │
                                                    gate * up: (bsz, sql, 1365)
                                                        │
                                                  down_proj ──→ (bsz, sql, 512)
                                                        │
                                                     dropout ──→ 输出
```

---

## 4. MoE (Mixture of Experts) 背景知识

### 4.1 问题：为什么需要 MoE？

传统 Transformer 每一层只有一个 FFN。模型参数量受限于计算量（FLOPs）—— 参数量越大，前向传播需要的计算越多。

MoE 的灵感：

> **让模型"变大"但不"变胖"** —— 每一层有 N 个专家 FFN，但每次只激活其中 K 个。总参数量 = N × 单个 FFN 参数量，但前向计算量只 = K × 单个 FFN 计算量。

### 4.2 MoE 的核心思想

```
┌─────────────────────────────────────┐
│            输入 token x             │
│              (512维)                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│         门控网络 (Gating)            │
│    计算每个专家的"适合程度"           │
│    score = [0.7, 0.2, 0.05, 0.05]   │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────┬──────────────┬────────────────┐
│   选 top-2   │              │                │
│              │              │                │
│    专家0     │    专家1     │    专家2/3     │
│  (0.7权重)   │  (0.2权重)   │   (被忽略)      │
│              │              │                │
│   ffn_0(x)   │   ffn_1(x)   │                │
└──────────────┴──────────────┴────────────────┘
               │
               ▼
        output = 0.7 × ffn_0(x) + 0.2 × ffn_1(x)
```

### 4.3 MoE 的三个核心问题

**问题 1：路由决策怎么做？**
- 每个 token 去哪些专家？
- 每个专家的权重是多少？

**问题 2：负载均衡**
- 所有 token 都只路由到专家 0，其他专家闲置怎么办？
- 需要让 token 均匀分布在各个专家上

**问题 3：通信开销**
- token 可能被路由到不同 GPU 上的专家
- 需要高效的 all-to-all 通信

本项目目前只实现了**问题 1 和问题 2**（路由和负载均衡），没有实现多 GPU 专家并行。

---

## 5. MoE 的门控路由机制

### 5.1 路由决策：线性层 + Softmax

```python
# 输入: x (bsz, sql, hidden_size)
logits = x @ W_gating.T   # (bsz * sql, n_experts)
scores = softmax(logits)   # 每个 token 对每个专家的得分
```

$W_{gating}$ 是一个可学习的权重矩阵，形状为 `(n_experts, hidden_size)`。它学的是：**什么样的输入特征应该分给哪个专家**。

### 5.2 Top-K 选择

每个 token 只选得分最高的 K 个专家（默认 K=2）：

```python
topk_weight, topk_idx = torch.topk(scores, k=K)
```

- `topk_idx`: 选中的专家索引，形状 `(bsz * sql, K)`
- `topk_weight`: 对应的得分，形状 `(bsz * sql, K)`

### 5.3 归一化 Top-K 概率

选出来的 K 个权重需要重新归一化，使它们的和为 1：

```python
# 原始得分: [0.7, 0.2] → 归一化后: [0.778, 0.222]
topk_weight = topk_weight / topk_weight.sum()
```

这样加权和不会使输出大小剧烈波动。

### 5.4 输出计算（在 MoE 层中，本代码中 MoEGate 只负责路由决策）

```python
# 对 token 0: 选中了专家 3 和 7，权重 [0.6, 0.4]
output_token0 = 0.6 * expert_3(input_token0) + 0.4 * expert_7(input_token0)
```

**注意**：MoEGate 类只负责计算路由（选哪些专家、权重多少），实际调用专家 FFN 是在外部 MoE 层完成的。本项目代码中 MoEGate 是独立的，完整的 MoE 层需要配合 `MoELayer` 使用。

---

## 6. 负载均衡辅助损失 (Auxiliary Loss)

### 6.1 为什么需要负载均衡？

如果没有约束，模型会倾向于把所有 token 路由到同一个或少数几个"最好的"专家，其他专家变成"摆设"。这叫做 **路由坍缩 (routing collapse)**。

### 6.2 负载均衡损失的设计

核心思想：让每个专家处理的 token 数量大致相等。

#### 方法 A：Token-level Aux Loss (非 seq_aux)

对整个 batch 统计：

- $f_i$ = 专家 $i$ 被选中的 token 比例
- $P_i$ = 所有 token 对专家 $i$ 的路由概率均值

目标：$f_i \cdot P_i$ 要尽量小（如果专家 $i$ 同时被很多 token 选中且概率高，说明负载不均衡）

$$
\mathcal{L}_{aux} = \alpha \sum_{i=1}^{E} P_i \cdot f_i
$$

其中 $f_i = \frac{n_i}{T} \times E$，$n_i$ 是选中专家 $i$ 的 token 数，$T$ 是总 token 数，$E$ 是专家总数。

#### 方法 B：Sequence-level Aux Loss (seq_aux=True)

在方法 A 的基础上，对每个序列（batch 中的每个样本）分别计算负载均衡：

```python
# 对 batch 中的每个序列单独计算
ce = zeros(bsz, n_experts)
# 统计每个序列中每个专家被选中的次数
ce.scatter_add_(...)
# 归一化
ce.div_(sql * aux_topk / n_routed_experts)
# 与路由概率做内积
aux_loss = (ce * scores.mean(dim=1)).sum() * alpha
```

**区别**：

| 模式 | 粒度 | 适用场景 |
|------|------|---------|
| `seq_aux=False` | 整个 batch | 短序列、batch 大小适中 |
| `seq_aux=True` | 每个序列 | 长序列、batch 中序列长度差异大 |

序列级别更细粒度，避免因为 batch 中某个长序列全部路由到少数专家导致的偏差。

### 6.3 辅助损失的物理意义

$$\mathcal{L}_{total} = \mathcal{L}_{main} + \alpha \cdot \mathcal{L}_{aux}$$

- $\mathcal{L}_{main}$：主任务损失（语言模型的交叉熵损失）
- $\mathcal{L}_{aux}$：负载均衡损失
- $\alpha$：平衡系数（默认 0.01），通常很小，确保不影响主任务

---

## 7. MoEGate 逐行解析

```python
class MoEGate(nn.Module):
    def __init__(self, config: ZzConfig):
        super().__init__()
        self.config = config

        # ---------- 路由参数 ----------
        self.top_k = config.num_experts_per_tok     # 每个 token 选几个专家 (默认 2)
        self.n_routed_experts = config.n_routed_experts  # 总专家数 (默认 4)

        self.scoring_func = config.scoring_func     # 得分函数 (默认 "softmax")
        self.alpha = config.aux_loss_alpha          # 辅助损失系数 (默认 0.01)
        self.seq_aux = config.seq_aux               # 是否用序列级辅助损失

        self.norm_topk_porb = config.norm_topk_prob  # ⚠️ 拼写错误！应该是 norm_topk_prob
                                                     # 是否对 topk 概率做归一化

        # ---------- 门控权重 ----------
        # 形状: (n_experts, hidden_size)
        # 每个专家对应一个 hidden_size 维的权重向量
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )

        self.reset_parameters()

    # ⚠️ BUG 注意：以下两个方法缩进错误！
    # 它们目前被定义在 __init__ 内部，不会成为类方法
    # 详见第 8 节

    def reset_parameters(self) -> None:
        # Kaiming 初始化：适合 ReLU/SiLU 等激活函数
        # a = sqrt(5) 是 PyTorch Linear 默认的初始化参数
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # hidden_states 形状: (bsz, sql, hidden_size) 例: (2, 10, 512)
        bsz, sql, h = hidden_states.shape

        # reshape 成 (bsz * sql, hidden_size)，把 batch 和序列合并
        hidden_states = hidden_states.view(-1, h)    # (20, 512)

        # ---------- 计算路由得分 ----------
        # 门控线性层: logits = hidden_states @ weight.T
        # 每个 token 对每个专家有一个分数
        logits = F.linear(hidden_states, self.weight, None)  # (20, 4)

        # Softmax 归一化
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)          # (20, 4)，每行之和为 1
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # ---------- Top-K 选择 ----------
        # 对每个 token 选 K 个得分最高的专家
        # topk_weight: (20, 2) — 选中的 K 个专家的分数
        # topk_idx:    (20, 2) — 选中的 K 个专家的索引
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # ---------- Top-K 概率归一化 ----------
        # 让选中的 K 个权重之和为 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1E-20  # 防除零
            topk_weight = topk_weight / denominator
            # 例: [0.6, 0.3] → [0.667, 0.333]

        # ---------- 负载均衡辅助损失 ----------
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores                   # (20, 4)
            aux_topk = self.top_k                     # 2
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # (2, 20) ← 注意形状变化

            if self.seq_aux:
                # ===== 序列级辅助损失 =====
                # scores 还原回 (bsz, sql, n_experts)
                scores_for_seq_aux = scores_for_aux.view(bsz, sql, -1)  # (2, 10, 4)

                # ce: 每个序列中每个专家被选中的次数，归一化后
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # scatter_add_: 按索引累加
                # 将 torch.ones(bsz, sql * aux_topk) 加到 ce 的对应位置
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,           # (2, 20)，专家索引
                    torch.ones(bsz, sql * aux_topk, device=hidden_states.device),
                ).div_(sql * aux_topk / self.n_routed_experts)
                # ce 形状: (2, 4)，每行是该序列中各专家的负载

                # 与路由概率做内积，求平均
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
                # scores_for_seq_aux.mean(dim=1): (2, 4)，每序列的平均路由概率
                # ce * ...: (2, 4) 逐元素相乘
                # .sum(dim=1): (2,) 每序列的负载均衡损失
                # .mean(): 标量，batch 平均
                # * alpha: 缩放系数

            else:
                # ===== Token 级辅助损失 =====
                # one_hot 编码: 把专家索引变成 one-hot
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1),   # (40,)
                    num_classes=self.n_routed_experts # 4
                )                                     # (40, 4)

                # 计算每个专家被选中的 token 比例
                ce = mask_ce.float().mean(0)          # (4,)
                # 所有 token 对每个专家的平均路由概率
                Pi = scores_for_aux.mean(0)           # (4,)
                # fi = ce * E，归一化因子
                fi = ce * self.n_routed_experts       # (4,)
                # 负载不均衡程度 = Σ Pi * fi
                aux_loss = (Pi * fi).sum() * self.alpha

        else:
            # 推理时或 alpha=0 时不计算辅助损失
            aux_loss = scores.new_zeros(1).squeeze()   # 0 标量张量

        # 返回三个值
        return topk_idx, topk_weight, aux_loss
```

**返回值的含义**：

| 返回值 | 形状 | 含义 |
|--------|------|------|
| `topk_idx` | `(bsz * sql, K)` | 每个 token 选中的 K 个专家的索引 |
| `topk_weight` | `(bsz * sql, K)` | 对应的归一化权重 |
| `aux_loss` | 标量 | 负载均衡辅助损失（训练时） |

---

## 8. 代码 Bug 分析

### 8.1 MoEGate 缩进 Bug（严重）

**问题**：`reset_parameters` 和 `forward` 方法被缩进在了 `__init__` 内部。

```python
class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        ...
        self.reset_parameters()   # ← 第 341 行

        def reset_parameters(self):   # ← 第 343 行，缩进错误！
            ...

        def forward(self, hidden_states):   # ← 第 346 行，缩进错误！
            ...
```

**影响**：
1. `self.reset_parameters()` 在第 341 行执行时，`reset_parameters` 函数还未定义（它在 343 行才定义）
2. 即使把调用移到后面，这两个函数也只是 `__init__` 内部的**局部函数**，不会成为类方法
3. `self.weight` 不会被正确初始化（保持 `torch.empty` 分配的未初始化值）
4. 外部调用 `moegate(hidden_states)` 时，会报错 `'MoEGate' object has no attribute 'forward'`

**修复方案**：

```python
class MoEGate(nn.Module):
    def __init__(self, config: ZzConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob   # 修正拼写
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:    # ← 取消缩进，成为类方法
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):       # ← 取消缩进，成为类方法
        ...
```

### 8.2 `norm_topk_porb` 拼写错误

```python
self.norm_topk_porb = config.norm_topk_prob   # porb → prob
```

建议统一修正为 `norm_topk_prob`。

### 8.3 与 MoEGate 配合的完整 MoE 层

本项目目前只有 `MoEGate`（门控），缺少 `MoELayer`（整合门控和专家的完整层）。一个完整的 MoE 层需要类似：

```python
class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = MoEGate(config)
        # 创建 N 个专家 FFN
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)
        ])

    def forward(self, x):
        # 1. 门控决策
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # 2. 按专家聚合 token（关键步骤，涉及 all-to-all 通信）
        # 3. 每个专家处理分配到的 token
        # 4. 按权重加权求和
        # 5. 返回输出和辅助损失
        ...
        return output, aux_loss
```

---

## 9. 数据流全图

### 9.1 FeedForward 数据流

```
输入 x: (bsz, sql, 512)
    │
    ├── gate_proj ──→ (bsz, sql, 1365) ──→ SiLU ──→ gate: (bsz, sql, 1365)
    │                                                        │
    │    SiLU(z) = z * sigmoid(z)                            │
    │    负值区域 → 小正值（非完全截断）                       │
    │    正值区域 → 近似线性通过                             │
    │                                                        ▼
    ├─ up_proj ──→ (bsz, sql, 1365) ─────────────────────→ up: (bsz, sql, 1365)
    │                                                        │
    │                                                    gate * up
    │                                                    (逐元素乘)
    │                                                        │
    │                                                        ▼
    │                                                  gated: (bsz, sql, 1365)
    │                                                        │
    └────────────────────────────────────────────────── down_proj ──→ (bsz, sql, 512)
                                                                │
                                                            dropout
                                                                │
                                                            输出
```

### 9.2 MoEGate 数据流

```
输入 hidden_states: (bsz, sql, 512)
    │
    ▼ view
(bsz * sql, 512)  例: (20, 512)
    │
    ├── @ weight.T ──→ logits: (20, 4)
    │   weight: (4, 512)
    │
    ├── softmax(dim=-1) ──→ scores: (20, 4)
    │   每行之和为 1
    │
    ├── topk(K=2) ──→ topk_weight: (20, 2)
    │               └─ topk_idx:    (20, 2)
    │
    ├── [可选] 归一化 topk_weight
    │   [0.7, 0.2] → [0.778, 0.222]
    │
    └── [训练时] 计算 aux_loss
            │
            ├── seq_aux=True:
            │   对 batch 中每个序列分别统计专家负载
            │   ce: (bsz, n_experts) ← 各专家负载比例
            │   aux_loss = mean(ce * mean_scores) * alpha
            │
            └── seq_aux=False:
                对整个 batch 统计
                ce: (n_experts,)
                aux_loss = sum(Pi * fi) * alpha

返回: (topk_idx, topk_weight, aux_loss)
```

### 9.3 完整的 MoE Transformer 层（概念图）

```
输入 x
    │
    ├── RMSNorm ──→ Attention ──→ + ──→ x'
    │                              (残差连接)
    │
    ├── RMSNorm ──→ MoELayer ──→ + ──→ 输出
    │       │
    │       ├── MoEGate: 决定每个 token 去哪些专家
    │       │
    │       └── 选中的专家 FFN 计算 → 加权求和
    │
    └── 辅助损失 aux_loss（反向传播时加到主损失上）
```

---

## 10. 关键公式速查表

| 公式 | 含义 |
|------|------|
| $\text{SwiGLU}(x) = \text{SiLU}(x W_{gate}) \odot (x W_{up})$ | SwiGLU 门控前馈网络 |
| $\text{SiLU}(x) = x \cdot \sigma(x)$ | SiLU 激活函数 |
| $d_{inter} = \frac{8}{3} d_{hidden}$ | SwiGLU FFN 的中间维度 |
| $\text{scores} = \text{softmax}(x W_{gating}^T)$ | 门控路由得分 |
| $\text{topk_weight} = \frac{\text{topk_weight}}{\sum \text{topk_weight}}$ | Top-K 归一化 |
| $\mathcal{L}_{aux}^{seq} = \alpha \cdot \frac{1}{B} \sum_{b=1}^{B} \sum_{i=1}^{E} c_{b,i} \cdot \bar{s}_{b,i}$ | 序列级辅助损失 |
| $\mathcal{L}_{aux}^{token} = \alpha \cdot \sum_{i=1}^{E} P_i \cdot f_i$ | Token 级辅助损失 |
| $f_i = \frac{n_i}{T} \cdot E$ | 专家负载因子 |

---

## 总结

| 模块 | 核心作用 | 关键知识点 |
|------|---------|-----------|
| **FeedForward** | 单个 token 的非线性特征变换 | SwiGLU 门控机制、中间维度 8/3 倍 |
| **MoEGate** | 决定 token 分配给哪些专家 | Top-K 路由、Softmax 得分、负载均衡 |

**FeedForward 和 Attention 的对比**：

| | Attention | FeedForward |
|---|---|---|
| 交互范围 | 跨位置（不同 token 之间） | 单个位置内（独立处理每个 token） |
| 参数量 | 大（Q/K/V/O 四个投影） | 更大（中间维度扩展） |
| 计算量 | $O(n^2 \cdot d)$ | $O(n \cdot d \cdot d_{inter})$ |
| 作用 | "看别处" | "深加工" |

**MoE 的核心价值**：在不增加前向计算量的前提下大幅增加模型总参数量，实现"稀疏激活"。当前大模型竞赛中，MoE 是扩大模型规模的主流手段之一（如 DeepSeek-V3、Mixtral 等）。
