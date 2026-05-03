# RoPE 位置编码符号化流程解析

> 不用具体数字，只用**符号和张量形状**，完整展示 RoPE 从参数到输出的每一步变换。

---

## 目录

1. [符号定义](#1-符号定义)
2. [整体流程概览](#2-整体流程概览)
3. [函数一：precompute_freqs_cis](#3-函数一precompute_freqs_cis)
4. [函数二：apply_rotary_pos_emb](#4-函数二apply_rotary_pos_emb)
5. [函数三：rotate_half](#5-函数三rotate_half)
6. [函数四：repeat_kv（辅助理解）](#6-函数四repeat_kv辅助理解)
7. [完整数据流图（从 Attention.forward 视角）](#7-完整数据流图从-attentionforward-视角)
8. [形状变化总表](#8-形状变化总表)

---

## 1. 符号定义

| 符号 | 含义 | 本项目默认值 |
|------|------|-------------|
| $B$ | batch_size（批次大小） | 2 |
| $S$ | seq_len（序列长度） | 当前输入长度 |
| $S_{cache}$ | KV Cache 中已缓存的序列长度 | 推理时递增 |
| $D$ | hidden_size（模型隐藏维度） | 512 |
| $H$ | num_attention_heads（注意力头数） | 8 |
| $H_{kv}$ | num_key_value_heads（KV 头数） | 2 |
| $R$ | n_rep = H / H_kv（KV 复制倍数） | 4 |
| $d$ | head_dim = D / H（每个头的维度） | 64 |
| $E$ | end = max_position_embeddings（最大位置） | 32768 |
| $\theta_{base}$ | rope_base（频率基数） | 1,000,000 |

---

## 2. 整体流程概览

```
RoPE 发生在 Attention 层内部，只作用于 Q 和 K，不作用于 V。

                    ┌────────────────────────────┐
                    │  预计算阶段（只需做一次）    │
                    │  precompute_freqs_cis()    │
                    │  输入: (d, E, θ_base)      │
                    │  输出: (freqs_cos, freqs_sin) │
                    │       形状: (E, d)         │
                    └──────────────┬─────────────┘
                                   │
                                   │ 传入 Attention
                                   ▼
                    ┌────────────────────────────┐
                    │  Attention.forward()        │
                    │                             │
                    │  xq: (B, S, H, d) ──→      │
                    │  xk: (B, S, H_kv, d) ──→   │
                    │                             │
                    │  apply_rotary_pos_emb()     │
                    │  输出: xq', xk'            │
                    └────────────────────────────┘
```

**关键**：`precompute_freqs_cis` 只在**模型初始化/首次推理**时调用一次，生成一个大的查找表。之后每个 forward 只需要**切片取出当前位置对应的 cos/sin**。

---

## 3. 函数一：precompute_freqs_cis

### 函数签名

```python
def precompute_freqs_cis(
    dim: int,                          # d = 64
    end: int = int(32 * 1024),         # E = 32768
    rope_base: float = 1E6,            # θ_base = 1,000,000
    rope_scaling: Optional[dict] = None  # YaRN 扩展参数
):
    return freqs_cos, freqs_sin        # 两者形状都是 (E, d)
```

### 输入参数

| 参数 | 符号 | 含义 |
|------|------|------|
| `dim` | $d$ | 每个注意力头的维度 |
| `end` | $E$ | 预计算的最大位置数 |
| `rope_base` | $\theta_{base}$ | 频率基数，控制频率衰减速度 |
| `rope_scaling` | - | YaRN 扩展参数（可选） |

### 内部变量逐行解析

#### 步骤 A：生成维度索引

```python
# torch.arange(0, dim, 2)
# 生成: [0, 2, 4, ..., d-2]
# 长度: d // 2

i = torch.arange(0, d, 2)      # 形状: (d/2,)
```

含义：将 d 维分成 d/2 对，每对两个维度。

#### 步骤 B：计算基础频率

```python
freqs = 1 / (rope_base ** (i.float() / dim))
```

**符号化**：

$$
\theta_j = \frac{1}{\theta_{base}^{2j/d}}, \quad j = 0, 1, 2, \ldots, \frac{d}{2}-1
$$

**结果**：

```
freqs: (d/2,)
     = [θ₀, θ₁, θ₂, ..., θ_{d/2-1}]

其中 θ₀ > θ₁ > θ₂ > ... > θ_{d/2-1}
     （频率从高到低单调递减）
```

含义：每个维度对有一个基础旋转频率。高频维度对（小 j）转得快，低频维度对（大 j）转得慢。

#### 步骤 C：生成位置索引

```python
t = torch.arange(end)          # 形状: (E,)
     = [0, 1, 2, ..., E-1]
```

含义：所有可能的位置编号。

#### 步骤 D：外积计算旋转角度

```python
freqs = torch.outer(t, freqs)   # 形状: (E, d/2)
```

**符号化**：

$$
\text{freqs}[p, j] = p \times \theta_j, \quad p = 0, 1, \ldots, E-1; \quad j = 0, 1, \ldots, \frac{d}{2}-1
$$

**结果**：

```
freqs: (E, d/2)

     ╱  θ₀    θ₁    θ₂   ...  θ_{d/2-1}  ╲
    │   0      0      0   ...     0      │  ← pos=0
    │   θ₀    θ₁    θ₂   ...   θ_{d/2-1} │  ← pos=1
    │  2θ₀   2θ₁   2θ₂   ...  2θ_{d/2-1}│  ← pos=2
    │   ⋮      ⋮      ⋮   ...     ⋮      │
    │ E·θ₀  E·θ₁  E·θ₂  ... E·θ_{d/2-1} │  ← pos=E-1
     ╲                                    ╱
```

含义：`freqs[p, j]` 表示位置 p、第 j 个维度对的旋转角度。

#### 步骤 E：计算 cos 和 sin

```python
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)   # (E, d)
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)   # (E, d)
```

**分解**：

```
# torch.cos(freqs): 形状 (E, d/2)
# cat(..., dim=-1): 沿最后一维拼接两份

# freqs_cos[p] = [cos(p·θ₀), cos(p·θ₁), ..., cos(p·θ_{d/2-1}),   ← 前半
#                 cos(p·θ₀), cos(p·θ₁), ..., cos(p·θ_{d/2-1})]   ← 后半
#               = [cos(p·θ₀), cos(p·θ₁), ..., cos(p·θ₀), cos(p·θ₁), ...]
#               共 d 个元素

# freqs_sin[p] 同理
```

**结果**：

```
freqs_cos: (E, d)
           每行 = [cos(角度_0), cos(角度_1), ..., cos(角度_{d/2-1}),
                   cos(角度_0), cos(角度_1), ..., cos(角度_{d/2-1})]

freqs_sin: (E, d)
           每行 = [sin(角度_0), sin(角度_1), ..., sin(角度_{d/2-1}),
                   sin(角度_0), sin(角度_1), ..., sin(角度_{d/2-1})]
```

为什么要复制两份？

因为 `rotate_half` 会把向量的后半截取负后移到前面。为了让逐元素乘法正确配对，cos/sin 也需要对应复制。

具体配对关系：

```
原始向量 x = [x₀, x₁, x₂, x₃, ..., x_{d/2-1}, x_{d/2}, x_{d/2+1}, ..., x_{d-1}]
                    │  │        │             │        │              │
                    │  │        │             │        └─ 第 d/2-1 对的另一半
                    │  │        │             └─ 第 d/2-1 对的起始
                    │  │        └─ 第 2 对的起始
                    │  └─ 第 1 对的起始
                    └─ 第 0 对的起始

⚠️ 注意：配对是 (xᵢ, x_{i+d/2})，不是相邻的 (x_{2i}, x_{2i+1})

cos = [c₀, c₁, c₂, c₃, ..., c_{d/2-1}, c₀, c₁, c₂, c₃, ...]
       │  │        │             │       │  │        │
       │  │        │             │       │  │        └─ x_{d-1} 的 cos (配 x_{d/2-1})
       │  │        │             │       │  └─ x_{d/2+1} 的 cos (配 x₁)
       │  │        │             │       └─ x_{d/2} 的 cos (配 x₀)
       │  │        │             └─ x_{d/2-1} 的 cos
       │  │        └─ x₂ 的 cos
       │  └─ x₁ 的 cos
       └─ x₀ 的 cos
```

#### 输出

```python
return freqs_cos, freqs_sin
```

| 返回值 | 形状 | 含义 |
|--------|------|------|
| `freqs_cos` | $(E, d)$ | 每个位置、每个维度的 cos 值 |
| `freqs_sin` | $(E, d)$ | 每个位置、每个维度的 sin 值 |

---

## 4. 函数二：apply_rotary_pos_emb

### 函数签名

```python
def apply_rotary_pos_emb(
    q,                        # (B, S, H, d)  或 (B, S, H_kv, d)
    k,                        # (B, S, H_kv, d)
    cos,                      # (S, d) 或 (E, d) 的切片
    sin,                      # (S, d) 或 (E, d) 的切片
    position_idx=None,        # 可选的自定义位置索引
    unsqueeze_dim=1           # 在哪个维度插入新轴
):
    return q_embed, k_embed   # 形状与输入相同
```

### 输入参数

| 参数 | 形状 | 含义 |
|------|------|------|
| `q` | $(B, S, H, d)$ | 多头拆分后的 Q |
| `k` | $(B, S, H_{kv}, d)$ | 多头拆分后的 K |
| `cos` | $(S, d)$ 或 $(E, d)$ | 位置对应的 cos 值 |
| `sin` | $(S, d)$ 或 $(E, d)$ | 位置对应的 sin 值 |
| `position_idx` | `None` 或 tensor | 自定义位置（用于特殊场景） |
| `unsqueeze_dim` | int | 插入广播维度的位置 |

### 内部执行流程

#### 步骤 A：unsqueeze 扩展维度

```python
cos.unsqueeze(unsqueeze_dim)    # unsqueeze_dim = 1
sin.unsqueeze(unsqueeze_dim)    # unsqueeze_dim = 1
```

**变换**：

```
cos: (S, d) ──unsqueeze(dim=1)──→ (S, 1, d)
sin: (S, d) ──unsqueeze(dim=1)──→ (S, 1, d)
```

含义：在第 1 维插入大小为 1 的维度，使得形状与 Q/K 兼容，可以通过**广播机制**进行逐元素乘法。

#### 步骤 B：rotate_half 变换

```python
def rotate_half(x):
    # x: (B, S, H, d)
    
    # x[..., d//2 :] 取后半截: (B, S, H, d/2)
    # -x[..., d//2 :] 取负:    (B, S, H, d/2)
    # x[..., : d//2] 取前半截: (B, S, H, d/2)
    
    # cat(..., dim=-1) 沿最后一维拼接
    return torch.cat(
        (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]),
        dim=-1
    )                                    # (B, S, H, d)
```

**符号化**：

```
输入 x = [x₀, x₁, ..., x_{d/2-1}, x_{d/2}, x_{d/2+1}, ..., x_{d-1}]
            └──── 前半截 ────┘  └────────── 后半截 ──────────┘
            第0对的起始        第0对的另一半

rotate_half(x) = [-x_{d/2}, -x_{d/2+1}, ..., -x_{d-1}, x₀, x₁, ..., x_{d/2-1}]
                    └──── 后半截取负 ────┘  └──── 前半截不变 ────┘
                    第0对取负后放前面      原前半截接在后面

⚠️ 配对方式：(xᵢ, x_{i+d/2})，即前半截第 i 个与后半截第 i 个配对
```

**结果**：

```
rotate_half(q): (B, S, H, d)
rotate_half(k): (B, S, H_kv, d)
```

含义：将向量后半截取负后拼到前面。这是为了用向量化操作实现二维旋转矩阵。

#### 步骤 C：应用 RoPE 公式

```python
q_embed = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
k_embed = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
```

**逐元素展开**（以 q 为例）：

```
q_embed[b, s, h, i] = q[b, s, h, i] * cos[s, i] + rotate_half(q)[b, s, h, i] * sin[s, i]

其中 b ∈ [0, B), s ∈ [0, S), h ∈ [0, H), i ∈ [0, d)
```

**等价于二维旋转**（对第 j 对维度，j = i % (d/2)，注意配对是 (xⱼ, x_{j+d/2})）：

```
当 i < d/2 时（前半截）:
    q_embed[..., i] = q[..., i] * cos(s, i) - q[..., i + d/2] * sin(s, i)

当 i >= d/2 时（后半截）:
    q_embed[..., i] = q[..., i] * cos(s, i - d/2) + q[..., i - d/2] * sin(s, i - d/2)
```

即第 j 对 (qⱼ, q_{j+d/2}) 的旋转：
- q'ⱼ      = qⱼ * cos(θⱼ) - q_{j+d/2} * sin(θⱼ)
- q'_{j+d/2} = q_{j+d/2} * cos(θⱼ) + qⱼ * sin(θⱼ)

**广播过程详解**：

```
q:           (B, S, H, d)
cos:         (S, 1, d)     ← unsqueeze 后
             │  │  │
             │  │  └─ 匹配 d
             │  └──── 广播: 1 → H（复制 H 份）
             └─────── 匹配 S

结果:        (B, S, H, d)   ← 逐元素相乘
```

#### 输出

```python
return q_embed, k_embed
```

| 返回值 | 形状 | 含义 |
|--------|------|------|
| `q_embed` | $(B, S, H, d)$ | 施加 RoPE 后的 Q |
| `k_embed` | $(B, S, H_{kv}, d)$ | 施加 RoPE 后的 K |

---

## 5. 函数三：rotate_half

### 单独拆解

```python
def rotate_half(x):
    # 假设 d = 8，展示每一步的形状变化
    # 配对方式: (x₀,x₄), (x₁,x₅), (x₂,x₆), (x₃,x₇)
    
    # x:                      (B, S, H, 8)
    #                         [0, 1, 2, 3, 4, 5, 6, 7]
    #                         │───── 前半截 ─────││───── 后半截 ─────│
    #                         │ 第0对 ││ 第3对  ││ 第0对 ││ 第3对  │
    
    # x[..., 4 :]            (B, S, H, 4)
    #                         [4, 5, 6, 7]  ← 后半截（各对的另一半）
    
    # -x[..., 4 :]           (B, S, H, 4)
    #                         [-4, -5, -6, -7]
    
    # x[..., : 4]            (B, S, H, 4)
    #                         [0, 1, 2, 3]  ← 前半截（各对的起始）
    
    # cat([...], dim=-1)     (B, S, H, 8)
    #                         [-4, -5, -6, -7, 0, 1, 2, 3]
    #                         │后半截取负  ││  原前半截   │
    #                         │移到前面    ││  接到后面   │
    
    return torch.cat(
        (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]),
        dim=-1
    )
```

### 为什么这样就能实现二维旋转？

二维旋转矩阵：

```
[x']   [cos(θ)  -sin(θ)] [x]
[y'] = [sin(θ)   cos(θ)] [y]

即:
x' = x * cos(θ) - y * sin(θ)
y' = x * sin(θ) + y * cos(θ)
```

把 d 维向量分成两半，前半截和后半截对应位置组成一对：
- 第 j 对 = (xⱼ, x_{j+d/2})，其中 j = 0, 1, ..., d/2-1

rotate_half 的巧妙之处：

```
原始:     [x₀, x₁, x₂, x₃, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]
rotate:   [-x_{d/2}, -x_{d/2+1}, ..., -x_{d-1}, x₀, x₁, ..., x_{d/2-1}]

现在逐元素看:
位置 j (前半截): 原始值 xⱼ, rotate_half 值 -x_{j+d/2}
         → xⱼ * cos(θⱼ) + (-x_{j+d/2}) * sin(θⱼ) = xⱼ*cos(θⱼ) - x_{j+d/2}*sin(θⱼ)
         这正好是第 j 对的前半部分旋转公式！

位置 j+d/2 (后半截): 原始值 x_{j+d/2}, rotate_half 值 xⱼ
         → x_{j+d/2} * cos(θⱼ) + xⱼ * sin(θⱼ)
         这正好是第 j 对的后半部分旋转公式！
```

由于 cos 和 sin 都是复制了两份的 `[cos(θ₀), cos(θ₁), ..., cos(θ_{d/2-1}), cos(θ₀), cos(θ₁), ...]`，所以：
- 前半截位置 j 乘到 cos(θⱼ) 和 sin(θⱼ)
- 后半截位置 j+d/2 也乘到 cos(θⱼ) 和 sin(θⱼ)
正好对应第 j 对的旋转！

---

## 6. 函数四：repeat_kv（辅助理解）

### 为什么需要 repeat_kv？

GQA 配置下：

```
Q 有 H = 8 个头
K/V 只有 H_kv = 2 个头

要做注意力: Q @ K^T，要求 Q 和 K 的头数相同

所以需要把 K/V 的每个头复制 R = H/H_kv = 4 份
```

### 函数签名

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: (B, S, H_kv, d) = (B, S, 2, 64)
    # n_rep: R = 4
```

### 内部变换

```python
# 步骤 1: 插入新维度
x[:, :, :, None, :]      # (B, S, H_kv, 1, d)
                          #     = (B, S, 2, 1, 64)

# 步骤 2: expand 广播（不实际复制内存）
.expand(B, S, H_kv, R, d)  # (B, S, H_kv, R, d)
                            #     = (B, S, 2, 4, 64)

# 步骤 3: reshape 合并
.reshape(B, S, H_kv * R, d) # (B, S, H, d)
                              #     = (B, S, 8, 64)
```

### 内存效率

`expand` 是**视图操作**，不实际复制数据。它只是改变 tensor 的 stride（步长），让同一个内存块被多个 head "看到"。

```
原始内存: [kv0_data, kv1_data]    ← 2 个 head

expand 后:
  head0 看到的内存: kv0_data（stride 为 0 的维度）
  head1 看到的内存: kv0_data
  head2 看到的内存: kv0_data
  head3 看到的内存: kv0_data
  head4 看到的内存: kv1_data
  head5 看到的内存: kv1_data
  head6 看到的内存: kv1_data
  head7 看到的内存: kv1_data
```

---

## 7. 完整数据流图（从 Attention.forward 视角）

```
输入 x: (B, S, D) = (2, 10, 512)
    │
    ├── q_proj ──→ (B, S, H*d) = (2, 10, 512)
    │                │
    │                ▼ view
    │             (B, S, H, d) = (2, 10, 8, 64)  ← xq
    │                │
    │                ▼
    │         apply_rotary_pos_emb(xq, xk, cos, sin)
    │                │
    │         ┌──────┴──────┐
    │         │ 内部过程:   │
    │         │             │
    │         │ cos: (S, d) ──unsqueeze(1)──→ (S, 1, d)
    │         │ sin: (S, d) ──unsqueeze(1)──→ (S, 1, d)
    │         │             │
    │         │ rotate_half(xq): (B, S, H, d)
    │         │ rotate_half(xk): (B, S, H_kv, d)
    │         │             │
    │         │ q_embed = xq * cos + rotate_half(xq) * sin
    │         │ k_embed = xk * cos + rotate_half(xk) * sin
    │         │             │
    │         └──────┬──────┘
    │                │
    │                ▼
    │         (B, S, H, d) = (2, 10, 8, 64)  ← xq_embed
    │
    ├── k_proj ──→ (B, S, H_kv*d) = (2, 10, 128)
    │                │
    │                ▼ view
    │             (B, S, H_kv, d) = (2, 10, 2, 64)  ← xk
    │                │
    │                ▼ (同上，进入 apply_rotary_pos_emb)
    │             (B, S, H_kv, d) = (2, 10, 2, 64)  ← xk_embed
    │                │
    │                ▼ + past_key_value (推理时)
    │             (B, S_total, H_kv, d)  ← S_total = S_cache + S
    │                │
    │                ▼ repeat_kv(R=4)
    │             (B, S_total, H, d) = (2, S_total, 8, 64)
    │                │
    │                ▼ transpose(1, 2)
    │             (B, H, S_total, d) = (2, 8, S_total, 64)
    │
    └── v_proj ──→ (B, S, H_kv*d) = (2, 10, 128)
                     │
                     ▼ view
                  (B, S, H_kv, d) = (2, 10, 2, 64)  ← xv
                     │
                     ▼ + past_key_value (推理时)
                  (B, S_total, H_kv, d)
                     │
                     ▼ repeat_kv(R=4)
                  (B, S_total, H, d)
                     │
                     ▼ transpose(1, 2)
                  (B, H, S_total, d) = (2, 8, S_total, 64)


Q: (B, H, S, d) = (2, 8, 10, 64)
K: (B, H, S_total, d) = (2, 8, S_total, 64)   ← S_total ≥ S（有 cache 时）
V: (B, H, S_total, d) = (2, 8, S_total, 64)
    │
    ├── Flash Attention 路径（训练时）
    │   F.scaled_dot_product_attention(Q, K, V)
    │   输出: (B, H, S, d) = (2, 8, 10, 64)
    │
    └── 手动计算路径（推理/KV Cache 时）
        Q @ K^T: (B, H, S, S_total)
        + mask
        softmax
        @ V
        输出: (B, H, S, d) = (2, 8, 10, 64)
                │
                ▼ transpose(1, 2)
             (B, S, H, d) = (2, 10, 8, 64)
                │
                ▼ reshape
             (B, S, H*d) = (2, 10, 512)
                │
                ▼ o_proj
             (B, S, D) = (2, 10, 512)
```

---

## 8. 形状变化总表

### precompute_freqs_cis

| 变量 | 输入形状 | 输出形状 | 操作 |
|------|---------|---------|------|
| `torch.arange(0, dim, 2)` | - | $(d/2,)$ | 生成维度索引 |
| `freqs`（基础频率） | - | $(d/2,)$ | 计算 θ_j |
| `torch.arange(end)` | - | $(E,)$ | 生成位置索引 |
| `torch.outer(t, freqs)` | $(E,), (d/2,)$ | $(E, d/2)$ | 外积 |
| `torch.cos(freqs)` | $(E, d/2)$ | $(E, d/2)$ | 逐元素 cos |
| `torch.cat([..., ...])` | $(E, d/2), (E, d/2)$ | $(E, d)$ | 沿最后一维拼接 |
| **最终返回** | - | $(E, d), (E, d)$ | freqs_cos, freqs_sin |

### apply_rotary_pos_emb

| 变量 | 输入形状 | 输出形状 | 操作 |
|------|---------|---------|------|
| `q` | $(B, S, H, d)$ | - | 输入 Q |
| `k` | $(B, S, H_{kv}, d)$ | - | 输入 K |
| `cos` | $(S, d)$ | $(S, 1, d)$ | unsqueeze(1) |
| `sin` | $(S, d)$ | $(S, 1, d)$ | unsqueeze(1) |
| `rotate_half(q)` | $(B, S, H, d)$ | $(B, S, H, d)$ | 后半截取负前移 |
| `rotate_half(k)` | $(B, S, H_{kv}, d)$ | $(B, S, H_{kv}, d)$ | 同上 |
| `q * cos` | $(B, S, H, d) \times (S, 1, d)$ | $(B, S, H, d)$ | 广播乘法 |
| `rotate_half(q) * sin` | $(B, S, H, d) \times (S, 1, d)$ | $(B, S, H, d)$ | 广播乘法 |
| `q_embed` | - | $(B, S, H, d)$ | 相加 |
| `k_embed` | - | $(B, S, H_{kv}, d)$ | 相加 |

### repeat_kv

| 变量 | 输入形状 | 输出形状 | 操作 |
|------|---------|---------|------|
| `x` | $(B, S, H_{kv}, d)$ | - | 输入 K 或 V |
| `x[:, :, :, None, :]` | $(B, S, H_{kv}, d)$ | $(B, S, H_{kv}, 1, d)$ | 插入维度 |
| `.expand(...)` | $(B, S, H_{kv}, 1, d)$ | $(B, S, H_{kv}, R, d)$ | 广播扩展 |
| `.reshape(...)` | $(B, S, H_{kv}, R, d)$ | $(B, S, H_{kv} \cdot R, d)$ | 合并维度 |
| **最终** | - | $(B, S, H, d)$ | 等于 $(B, S, H_{kv} \cdot R, d)$ |

---

## 总结

| 函数 | 核心作用 | 输入 → 输出 |
|------|---------|------------|
| `precompute_freqs_cis` | **预计算**所有位置的 cos/sin 查找表 | $(d, E) \rightarrow (E, d), (E, d)$ |
| `apply_rotary_pos_emb` | **应用**旋转位置编码到 Q/K | $(B, S, H/H_{kv}, d) \rightarrow$ 同形状 |
| `rotate_half` | **辅助**：向量后半截取负前移 | $(..., d) \rightarrow (..., d)$ |
| `repeat_kv` | **辅助**：复制 KV head 匹配 Q head 数 | $(B, S, H_{kv}, d) \rightarrow (B, S, H, d)$ |

RoPE 的本质：**用向量化操作高效实现高维旋转，且旋转后的点积只与相对位置有关**。
