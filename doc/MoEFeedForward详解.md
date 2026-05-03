# MoEFeedForward 全面详解

## 一、前置知识：什么是 Mixture of Experts (MoE)

传统 Transformer 的每个 FeedForward 层对**所有 token** 使用**同一组参数**进行计算。MoE 的核心思想是：**不同 token 由不同"专家"处理**，通过一个"门控"机制动态选择。

类比：把 FeedForward 想象成一个科室，普通模式是所有病人都看同一个医生；MoE 模式是有一个分诊台（Gate），根据病情把病人分给不同专科医生（Expert），每位医生专精不同领域。

---

## 二、整体架构概览

```
输入 x (bsz, seq_len, hidden_size)
        │
        ├─── [路由分支] ──────────────────────────────────────┐
        │     x → MoEGate → topk_idx, topk_weight, aux_loss   │
        │     (决定每个 token 去哪些专家)                        │
        │                                                      │
        │     x 被 reshape 为 (bsz*seq_len, hidden_size)        │
        │              │                                        │
        │     ┌────────┼────────┬────────┐                     │
        │     ▼        ▼        ▼        ▼                     │
        │  Expert_0  Expert_1  Expert_2  Expert_3  (路由专家)    │
        │     │        │        │        │                     │
        │     └────────┴────────┴────────┘                     │
        │              │                                        │
        │     加权求和 (topk_weight)                             │
        │     y_routed (bsz, seq_len, hidden_size)              │
        │                                                      │
        ├─── [共享分支] ──────────────────────────────────────┐│
        │     identity (原始输入)                                ││
        │              │                                        ││
        │     ┌────────┴────────┐                              ││
        │     ▼                 ▼                              ││
        │  Shared_0  Shared_1  ...  (共享专家，所有token都经过)   ││
        │     │                 │                              ││
        │     └────────┬────────┘                              ││
        │     y_shared (bsz, seq_len, hidden_size)              ││
        │                                                      ││
        └──────────────────────────────────────────────────────┘│
                                                                 │
        y = y_routed + y_shared ←────────────────────────────────┘
        │
        输出 (bsz, seq_len, hidden_size)
```

**两条路径并行计算后相加**：
- **路由专家 (routed experts)**：token 被门控分配到 top-k 个专家
- **共享专家 (shared experts)**：所有 token 都经过，提供通用知识

---

## 三、配置参数说明

定义在 `ZzMindConfig` 中的 MoE 相关参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `use_moe` | `False` | 是否启用 MoE |
| `n_routed_experts` | `4` | 路由专家总数 |
| `num_experts_per_tok` | `2` | 每个 token 激活的专家数 (top-k) |
| `n_shared_experts` | `1` | 共享专家数量 |
| `scoring_func` | `"softmax"` | 门控评分函数 |
| `aux_loss_alpha` | `0.01` | 负载均衡辅助损失系数 |
| `seq_aux` | `True` | 是否使用序列级辅助损失 |
| `norm_topk_prob` | `True` | 是否归一化 top-k 权重 |
| `hidden_size` | `512` | 隐藏层维度 |

---

## 四、基础组件：FeedForward (单个专家)

每个专家本质上就是一个标准的 **SwiGLU FeedForward**：

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        # 三层线性变换
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiLU  # Swish 激活函数

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)  # SwiGLU
        return self.down_proj(gated)
```

### 数据变化过程 (hidden_size=512, intermediate_size=1376)

```
输入:  x  shape: (..., 512)
                │
        ┌───────┴───────┐
        ▼               ▼
  gate_proj(x)    up_proj(x)
  (..., 1376)     (..., 1376)
        │               │
  SiLU激活            │
        │               │
        └─── × ─────────┘   ← 逐元素相乘 (SwiGLU)
                │
          (..., 1376)
                │
          down_proj
                │
          (..., 512)          ← 回到原始维度
```

**为什么用 SwiGLU 而不是简单 ReLU？**
- gate 分支通过 SiLU 激活后与 up 分支相乘，实现了**门控机制**
- 信息流动更加平滑，梯度更好，已被 LLaMA、Mistral 等主流模型验证

---

## 五、MoEGate：门控路由机制

门控层是 MoE 的核心——它决定每个 token 应该被哪些专家处理。

### 5.1 初始化

```python
class MoEGate(nn.Module):
    def __init__(self, config):
        self.weight = nn.Parameter(
            torch.empty((n_routed_experts, hidden_size))
        )
        # Kaiming 均匀初始化
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
```

`weight` 的形状是 `(n_routed_experts, hidden_size)` = `(4, 512)`，它本质上是一个**线性投影矩阵**，将 token 的隐藏表示映射为对每个专家的得分。

### 5.2 前向传播流程

```python
def forward(self, hidden_states):
    # hidden_states: (bsz, seq_len, hidden_size)
    bsz, seq_len, h = hidden_states.shape

    # Step 1: 展平 token 维度
    hidden_states = hidden_states.view(-1, h)  # (N, 512), N = bsz * seq_len

    # Step 2: 计算每个 token 对每个专家的原始得分
    logits = F.linear(hidden_states, self.weight, None)
    # logits: (N, n_routed_experts) = (N, 4)

    # Step 3: Softmax 归一化
    scores = logits.softmax(dim=-1)  # (N, 4), 每行和为 1

    # Step 4: 选取 top-k 个专家
    topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
    # topk_weight: (N, 2)  — 选中专家的权重
    # topk_idx:    (N, 2)  — 选中专家的索引 (0~3)

    # Step 5: 归一化 top-k 权重 (使权重和为 1)
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

    # Step 6: 计算辅助损失 (负载均衡)
    aux_loss = ...

    return topk_idx, topk_weight, aux_loss
```

### 5.3 数据变化示例

假设 batch=1, seq_len=3, hidden_size=512, n_routed_experts=4, top_k=2

```
hidden_states: (1, 3, 512)
        │ view(-1, 512)
        ▼
         (3, 512)
        │ F.linear(..., weight(4, 512))
        ▼
logits:  (3, 4)         ← 每个 token 对 4 个专家的原始得分
        │ softmax(dim=-1)
        ▼
scores:  (3, 4)         ← 概率分布，如:
         [[0.10, 0.05, 0.70, 0.15],    ← token 0 最倾向 Expert 2
          [0.60, 0.10, 0.05, 0.25],    ← token 1 最倾向 Expert 0
          [0.05, 0.55, 0.10, 0.30]]    ← token 2 最倾向 Expert 1
        │ topk(k=2)
        ▼
topk_idx:    (3, 2)     ← 选中的专家索引
         [[2, 3],        ← token 0 → Expert 2, 3
          [0, 3],        ← token 1 → Expert 0, 3
          [1, 3]]        ← token 2 → Expert 1, 3

topk_weight: (3, 2)     ← 对应权重 (归一化前)
         [[0.70, 0.15],
          [0.60, 0.25],
          [0.55, 0.30]]
        │ 归一化 (使每行和为1)
        ▼
         [[0.824, 0.176],
          [0.706, 0.294],
          [0.647, 0.353]]
```

### 5.4 辅助损失 (Auxiliary Loss)

**为什么需要辅助损失？** 纯粹靠 softmax 路由容易出现**负载不均衡**——大多数 token 都涌向少数几个专家，其他专家闲置。辅助损失通过惩罚不均匀的专家利用率来缓解这个问题。

#### 非序列级 (seq_aux=False) 的情况：

```
aux_loss = α × Σ_i (P_i × f_i)

其中:
- P_i = 所有 token 对专家 i 的平均得分 (softmax 输出的均值)
- f_i = 专家 i 被选中的频率 × 总专家数
- α  = aux_loss_alpha (默认 0.01)
```

直觉：当所有专家被均匀使用时，`f_i` 都接近 1，乘积最小；当负载不均时，某些 `f_i` 很大，损失增大。

---

## 六、MoEFeedForward：核心前向传播

### 6.1 初始化

```python
class MoEFeedForward(nn.Module):
    def __init__(self, config):
        # 路由专家: n_routed_experts 个独立的 FeedForward
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)  # 4 个
        ])

        # 门控层
        self.gate = MoEGate(config)

        # 共享专家: 所有 token 都会经过
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config) for _ in range(config.n_shared_experts)  # 1 个
            ])
```

**参数量对比**：
- 普通 FeedForward: `3 × 512 × 1376` ≈ 2.1M 参数
- MoEFeedForward: `(4 + 1) × 2.1M` + 门控参数 ≈ 10.6M 参数（但每次只激活 2 个路由专家 + 1 个共享专家）

---

### 6.2 训练阶段 forward

```python
def forward(self, x):
    identity = x                          # 保存原始输入给共享专家
    orig_shape = x.shape                  # (bsz, seq_len, hidden_size)

    # Step 1: 门控路由
    topk_idx, topk_weight, aux_loss = self.gate(x)
    # topk_idx:    (N, top_k) = (N, 2)
    # topk_weight: (N, top_k) = (N, 2)

    # Step 2: 展平
    x = x.view(-1, x.shape[-1])           # (N, 512)
    flat_topk_idx = topk_idx.view(-1)     # (N * top_k,) = (N*2,)

    # Step 3: 重复每个 token top_k 次
    x = x.repeat_interleave(top_k, dim=0) # (N*top_k, 512) = (N*2, 512)

    # Step 4: 逐专家处理
    y = torch.empty_like(x)               # (N*2, 512) 空张量
    for i, expert in enumerate(self.experts):
        mask = (flat_topk_idx == i)        # 布尔掩码: 哪些位置分配给专家 i
        expert_out = expert(x[mask])       # 只处理分配给该专家的 token
        y[mask] = expert_out               # 放回对应位置

    # Step 5: 加权求和
    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    # y: (N, 512)

    # Step 6: 恢复形状
    y = y.view(*orig_shape)               # (bsz, seq_len, 512)
```

### 6.3 训练阶段数据流详解

继续用之前的例子：bsz=1, seq_len=3, hidden_size=512, top_k=2

```
x: (1, 3, 512)  identity = x 的副本
        │
        │ view(-1, 512)
        ▼
x: (3, 512)           flat_topk_idx = [2, 3, 0, 3, 1, 3]
        │                         (展开后的专家索引)
        │ repeat_interleave(2, dim=0)
        ▼
x: (6, 512)           ← 每个 token 复制了 2 份
  行0: token_0 的副本 1 → 去 Expert 2
  行1: token_0 的副本 2 → 去 Expert 3
  行2: token_1 的副本 1 → 去 Expert 0
  行3: token_1 的副本 2 → 去 Expert 3
  行4: token_2 的副本 1 → 去 Expert 1
  行5: token_2 的副本 2 → 去 Expert 3

逐专家处理:
  Expert 0: 处理 x[行2]           → y[行2] = Expert_0(token_1)
  Expert 1: 处理 x[行4]           → y[行4] = Expert_1(token_2)
  Expert 2: 处理 x[行0]           → y[行0] = Expert_2(token_0)
  Expert 3: 处理 x[行1,行3,行5]   → y[行1] = Expert_3(token_0)
                                     y[行3] = Expert_3(token_1)
                                     y[行5] = Expert_3(token_2)

y: (6, 512)
        │
        │ view(3, 2, 512)       ← 恢复为 (N, top_k, hidden)
        ▼
y: (3, 2, 512)
        │
        │ × topk_weight.unsqueeze(-1)   ← (3, 2, 1) 广播相乘
        ▼
y: (3, 2, 512)           ← 每个专家输出已乘以对应权重
        │
        │ .sum(dim=1)                  ← 两个专家的结果加权求和
        ▼
y: (3, 512)              ← 每个 token 的最终路由输出
        │
        │ view(1, 3, 512)
        ▼
y: (1, 3, 512)
```

**加权求和的数学表达**：

对于第 j 个 token：
```
y_j = Σ_{k=1}^{top_k}  weight_jk × Expert_{idx_jk}(x_j)
```

例如 token_0：
```
y_0 = 0.824 × Expert_2(token_0) + 0.176 × Expert_3(token_0)
```

---

### 6.4 推理阶段 forward (moe_infer)

推理时使用 `@torch.no_grad()` 装饰的优化方法，避免重复计算，使用排序+分桶提高效率：

```python
@torch.no_grad()
def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
    expert_cache = torch.zeros_like(x)     # (N, 512) 零张量

    # Step 1: 对专家索引排序 (把同一专家的 token 分到一起)
    idxs = flat_expert_indices.argsort()
    # 例: flat_expert_indices = [2, 3, 0, 3, 1, 3]
    # argsort → [2, 4, 0, 1, 3, 5]
    #           即先 Expert 0(token_1), 再 Expert 1(token_2), 再 Expert 2(token_0), 最后 Expert 3(token_0,1,2)

    # Step 2: 统计每个专家分到的 token 数量的累计值
    tokens_per_expert = flat_expert_indices.bincount().cumsum(0)
    # bincount: [1, 1, 1, 3]  → Expert 0 有 1 个, Expert 1 有 1 个, Expert 2 有 1 个, Expert 3 有 3 个
    # cumsum:   [1, 2, 3, 6]  → Expert 0 在 [0,1), Expert 1 在 [1,2), Expert 2 在 [2,3), Expert 3 在 [3,6)

    # Step 3: 计算原始 token 索引
    token_idxs = idxs // top_k
    # 排序后的位置映射回原始 token: [1, 2, 0, 0, 1, 2]

    # Step 4: 逐专家批量处理
    for i, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
        if start_idx == end_idx:
            continue  # 该专家没有被分配到任何 token

        expert = self.experts[i]
        exp_token_idx = token_idxs[start_idx:end_idx]   # 该专家要处理的 token 索引
        expert_tokens = x[exp_token_idx]                 # 取出对应的 token 数据
        expert_out = expert(expert_tokens)               # 一次性批量处理
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])  # 乘以权重
        expert_cache.scatter_add_(0, ...)                # 散点加到结果中

    return expert_cache
```

### 6.5 推理阶段数据流详解

```
输入:
  x:                    (3, 512)        [token_0, token_1, token_2]
  flat_expert_indices:  [2, 3, 0, 3, 1, 3]
  flat_expert_weights:  [0.824, 0.176, 0.706, 0.294, 0.647, 0.353]

expert_cache: (3, 512) 全零

排序:
  idxs = argsort([2, 3, 0, 3, 1, 3]) = [2, 4, 0, 1, 3, 5]
  token_idxs = [2, 4, 0, 1, 3, 5] // 2 = [1, 2, 0, 0, 1, 2]

分桶:
  Expert 0: tokens_per_expert[0]=1, range [0, 1)
    token_idxs[0:1] = [1]          → x[1] = token_1
    out = Expert_0(token_1) × weight[idxs[0]] = Expert_0(token_1) × 0.706
    expert_cache[1] += out

  Expert 1: tokens_per_expert[1]=2, range [1, 2)
    token_idxs[1:2] = [2]          → x[2] = token_2
    out = Expert_1(token_2) × weight[idxs[1]] = Expert_1(token_2) × 0.647
    expert_cache[2] += out

  Expert 2: tokens_per_expert[2]=3, range [2, 3)
    token_idxs[2:3] = [0]          → x[0] = token_0
    out = Expert_2(token_0) × weight[idxs[2]] = Expert_2(token_0) × 0.824
    expert_cache[0] += out

  Expert 3: tokens_per_expert[3]=6, range [3, 6)
    token_idxs[3:6] = [0, 1, 2]    → x[[0,1,2]] = [token_0, token_1, token_2]
    out = Expert_3([t0,t1,t2]) × [weight[idxs[3]], weight[idxs[4]], weight[idxs[5]]]
        = [Expert_3(t0)×0.176, Expert_3(t1)×0.294, Expert_3(t2)×0.353]
    expert_cache[0] += Expert_3(t0)×0.176
    expert_cache[1] += Expert_3(t1)×0.294
    expert_cache[2] += Expert_3(t2)×0.353

最终结果:
  expert_cache[0] = 0.824 × Expert_2(token_0) + 0.176 × Expert_3(token_0)
  expert_cache[1] = 0.706 × Expert_0(token_1) + 0.294 × Expert_3(token_1)
  expert_cache[2] = 0.647 × Expert_1(token_2) + 0.353 × Expert_3(token_2)
```

### 6.6 训练 vs 推理的区别

| 特性 | 训练 (`forward`) | 推理 (`moe_infer`) |
|------|-----------------|-------------------|
| 梯度 | 需要反向传播 | `@torch.no_grad()` |
| token 组织 | `repeat_interleave` 复制 | 排序分桶，原位索引 |
| 专家调度 | 遍历专家，布尔掩码筛选 | 按专家分组批量处理 |
| 结果合并 | view + 乘法 + sum | `scatter_add_` 累加 |
| 效率 | 简单直观，适合 autograd | 更高效，避免内存复制 |

---

### 6.7 共享专家分支

```python
if self.config.n_shared_experts > 0:
    for expert in self.shared_experts:
        y = y + expert(identity)
```

共享专家对**所有 token** 使用**同一组参数**计算，结果直接加到路由专家的输出上。

**为什么需要共享专家？**
- 路由专家负责捕获特定领域的知识
- 共享专家负责捕获通用知识（语法、常见模式等）
- 避免所有基础能力都被分散到不同路由专家中

---

## 七、完整数据流总览

```
输入: x (1, 3, 512)
 │
 ├─── MoEGate ─────────────────────────────────────────────────
 │    x → flatten → F.linear → softmax → topk
 │    得到: topk_idx (3,2), topk_weight (3,2), aux_loss
 │
 ├─── 路由分支 ────────────────────────────────────────────────
 │    x.view(-1, 512) → (3, 512)
 │    repeat_interleave(2) → (6, 512)
 │    Expert_0(token_1)  → 0.706 × out
 │    Expert_1(token_2)  → 0.647 × out
 │    Expert_2(token_0)  → 0.824 × out
 │    Expert_3(token_0, token_1, token_2) → 加权
 │    加权求和 → (3, 512) → view → (1, 3, 512)
 │
 ├─── 共享分支 ────────────────────────────────────────────────
 │    identity (1, 3, 512) → SharedExpert_0 → (1, 3, 512)
 │
 └─── 合并 ────────────────────────────────────────────────────
      y = y_routed + y_shared
      self.aux_loss = aux_loss  (用于训练时的负载均衡优化)
      return y (1, 3, 512)
```

---

## 八、关键设计总结

### 8.1 为什么 MoE 有效

| 特性 | 普通 FeedForward | MoE FeedForward |
|------|-----------------|-----------------|
| 参数量 | ~2.1M | ~10.6M (5个专家) |
| **激活参数量** | ~2.1M | ~4.2M (2路由+1共享) |
| 计算量 | 全量 | 与激活专家数成正比 |
| 专业化 | 通用 | 路由专家各有所长 |

**核心优势**：参数量大但计算量小——模型容量大幅增加，而每次推理只多激活少量参数。

### 8.2 负载均衡的重要性

如果不加辅助损失，可能出现 **routing collapse**（路由坍塌）：
- 所有 token 都涌向同一两个专家
- 其他专家得不到训练，参数退化
- 模型退化为更小的普通模型

辅助损失 `aux_loss` 在训练时被加到总损失中，鼓励专家被均匀使用。

### 8.3 scatter_add_ 的巧妙

推理阶段使用 `scatter_add_` 而非训练阶段的 view+sum，原因是：
- 推理时不需要梯度，可以自由使用原地操作
- `scatter_add_` 直接在目标位置累加，无需中间张量
- 配合排序分桶，每个专家只需一次前向计算（而非逐 token）
