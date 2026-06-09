# train.py 详解（一）：GPT 模型架构

> **Agent 可修改文件**：这是 Agent 唯一可以编辑的文件。本文档分析 GPT 模型的架构设计。

---

## 一、整体架构概览

```
GPT 模型 (train.py:124-291)
│
├── 配置: GPTConfig (line 32-39)
│   ├── sequence_len: 2048
│   ├── vocab_size: 32768 (配置默认, 实际由 tokenizer 决定)
│   ├── n_layer: 12 (配置默认, 实际运行时 8)
│   ├── n_head, n_kv_head, n_embd: 由 DEPTH 和 ASPECT_RATIO 推导
│   └── window_pattern: "SSSL"
│
├── Token Embedding (line 130)
│   └── wte: nn.Embedding(vocab_size, n_embd)
│
├── Transformer Blocks × N (line 112-121)
│   │
│   ├── [Attention Sub-layer]
│   │   ├── Pre-RMSNorm
│   │   ├── Causal Self-Attention (with FlashAttn3 + Sliding Window)
│   │   │   ├── Query/Key/Value 投影
│   │   │   ├── Rotary Embedding 应用到 Q, K
│   │   │   ├── Q, K 各自做 RMSNorm
│   │   │   ├── Value Embedding 门控混合 (ResFormer)
│   │   │   └── Flash Attention 3 (causal + sliding window)
│   │   └── 残差连接
│   │
│   └── [MLP Sub-layer]
│       ├── Pre-RMSNorm
│       ├── Linear(n_embd → 4*n_embd)
│       ├── ReLU² 激活
│       ├── Linear(4*n_embd → n_embd)
│       └── 残差连接
│
├── 每一层的可学习标量
│   ├── resid_lambdas: 控制残差路径强度 (初始 1.0)
│   └── x0_lambdas: 控制 x0 shortcut 强度 (初始 0.1)
│
├── Value Embeddings (交替层)
│   └── ve_i: nn.Embedding(vocab_size, n_kv_head * head_dim)
│
├── Final RMSNorm
├── LM Head: Linear(n_embd → vocab_size, bias=False)
└── Logit Softcap: 15 * tanh(logits / 15)
```

---

## 二、配置类与模型构建

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
```

**运行时模型构建**（[train.py:469-477](file:///workspace/train.py#L469-L477)）：

```python
ASPECT_RATIO = 64
HEAD_DIM = 128

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO           # 8 * 64 = 512
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM  # 对齐到 128
    num_heads = model_dim // HEAD_DIM         # 512 // 128 = 4
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=tokenizer.get_vocab_size(),  # 实际 tokenizer 词表大小
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,                    # 无 GQA，K=H
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
```

**设计要点**：
- `ASPECT_RATIO` 控制深度与宽度的平衡
- `HEAD_DIM` 固定 128 是 Flash Attention 的高效尺寸
- `n_kv_head == n_head` 意味着没有 Grouped Query Attention（GQA）

---

## 三、RMSNorm（归一化）

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

RMSNorm = Root Mean Square Layer Normalization。公式：

```
y = x / RMS(x) * gamma
```

其中 `RMS(x) = sqrt(mean(x²))`，`gamma` 是可学习缩放参数（此处使用 F.rms_norm 的默认行为）。

**与 LayerNorm 的区别**：没有均值减法，只有方差归一化，计算更快且更稳定。

---

## 四、Rotary Position Embedding (RoPE)

```python
def apply_rotary_emb(x, cos, sin):
    """对 4D tensor [B, T, H, D] 应用 RoPE"""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]      # 分成两半
    y1 = x1 * cos + x2 * sin             # 旋转实部
    y2 = -x1 * sin + x2 * cos            # 旋转虚部
    return torch.cat([y1, y2], 3)
```

**预计算 RoPE 缓存**：

```python
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))  # 频率倒数
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)   # [seq_len, head_dim/2]
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.bfloat16(), sin.bfloat16()
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # [1, T, 1, D/2]
    return cos, sin
```

**关键观察**：
- RoPE 只应用于 Q 和 K（不应用于 V）
- `rotary_seq_len = config.sequence_len * 10`（预计算 10 倍长度，允许更长推理）
- 存储为 `register_buffer`（state_dict 的一部分，但不需要梯度）

---

## 五、Causal Self-Attention 详解

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // self.n_head
        
        # 投影矩阵 (无 bias)
        self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # ResFormer: Value Embedding 门控
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(32, n_kv_head, bias=False) if has_ve(layer_idx, n_layer) else None
    
    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        
        # Step 1: Q, K, V 投影
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # Step 2: Value Embedding 残差混合 (ResFormer)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            # 用输入的前 32 个通道预测每个 KV 头的门控值
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))  # [B, T, n_kv_head]
            # 初始 gate ≈ 1.0 (因 ve_gate 初始化为 0, sigmoid(0)=0.5, 2*0.5=1.0)
            v = v + gate.unsqueeze(-1) * ve
        
        # Step 3: 应用 RoPE 到 Q, K
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        
        # Step 4: Q, K 各自做 RMSNorm (!)
        q, k = norm(q), norm(k)
        
        # Step 5: Flash Attention 3
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        
        # Step 6: 输出投影
        y = self.c_proj(y)
        return y
```

### 5.1 ResFormer Value Embedding 机制

这是此实现的一个核心创新（来自 ResFormer 论文）：

```
常规注意力:
  V = Linear(x)
  
ResFormer:
  ve = Embedding(token_id)    # 全局 token 级的值表示
  gate = sigmoid(Linear(x[:32]))  # 输入依赖的门控
  V = V + gate * ve           # 残差式混合
```

**直觉**：注意力中的常规 V 是上下文相关的（依赖 x），而 VE 是原始 token 的"全局"值表示。门控让模型可以动态混合两者。

**交替层机制**：

```python
def has_ve(layer_idx, n_layer):
    """交替层有 VE，最后一层一定有"""
    return layer_idx % 2 == (n_layer - 1) % 2
```

例如 n_layer=8 时：layers 1, 3, 5, 7 有 VE（从 0 开始计数）。

### 5.2 Flash Attention 3 的选择

```python
from kernels import get_kernel

cap = torch.cuda.get_device_capability()
# Hopper (sm90): varunneal 的 FA3 实现
# 其他架构: kernels-community 的 FA3 实现
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn"
fa3 = get_kernel(repo).flash_attn_interface
```

**窗口大小**由 `window_pattern` 决定（见下一节）。

---

## 六、Sliding Window Attention（滑动窗口）

```python
def _compute_window_sizes(self, config):
    """'SSSL' 模式: 短窗口关注局部，长窗口整合全局"""
    pattern = config.window_pattern.upper()  # "SSSL"
    long_window = config.sequence_len         # 2048 (全上下文)
    short_window = long_window // 2           # 1024 (半上下文)
    char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
    
    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    
    # 强制最后一层为 L (全上下文)
    window_sizes[-1] = (long_window, 0)
    return window_sizes
```

**对于 n_layer=8, pattern="SSSL"**：

| Layer | 字符 | 窗口大小 | 说明 |
|-------|------|---------|------|
| 0 | S | 1024 | 浅层：局部模式 (语法、短语) |
| 1 | S | 1024 |  |
| 2 | S | 1024 |  |
| 3 | L | 2048 |  |
| 4 | S | 1024 |  |
| 5 | S | 1024 |  |
| 6 | S | 1024 |  |
| 7 | L | 2048 | 最后一层：强制全上下文 |

**设计直觉**：
- 浅层处理局部模式（不需要全上下文）
- 深层整合全局信息（需要全上下文）
- 滑动窗口减少注意力计算量：O(T×w) vs O(T²)
- 最后一层强制全上下文：确保输出 token 可以看到全部历史

---

## 七、MLP 层

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU² (不是 ReGLU)
        x = self.c_proj(x)
        return x
```

**MLP 类型对比**：

| 类型 | 公式 | 激活 | 参数量 |
|-----|------|------|--------|
| 标准 MLP | σ(xW₁)W₂ | ReLU/GELU | 8d² |
| GPT-2 MLP | GELU(xW₁)W₂ | GELU | 8d² |
| ReGLU (PaLM) | GELU(xW₁) ⊙ xW₂ | GELU | 16d² |
| **此项目 (ReLU²)** | **ReLU(xW₁)² W₂** | **ReLU 平方** | **8d²** |

**ReLU² 的优势**（Meta 论文发现）：
- 比 GELU 稍快（无 erf 计算）
- 在某些配置下实际效果更好
- 保持与标准 MLP 相同的参数/计算量

---

## 八、Block 结构（Pre-LN Residual）

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
    
    def forward(self, x, ve, cos_sin, window_size):
        # Pre-LN: 先归一化，再子层，再残差
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x
```

标准的 Pre-LN Transformer block，无 Parallel Block / Parallel Attention 等变体。

---

## 九、GPT 主模型的 Forward 流程

```python
def forward(self, idx, targets=None, reduction='mean'):
    B, T = idx.size()
    assert T <= self.cos.size(1)  # 确保不超过预计算的 RoPE 长度
    
    # Step 1: Token Embedding + Normalization
    x = self.transformer.wte(idx)   # [B, T, n_embd]
    x = norm(x)                      # 嵌入后立即归一化 (!)
    x0 = x                           # 保存原始嵌入 (用于 x0 shortcut)
    
    # Step 2: Transformer Blocks
    for i, block in enumerate(self.transformer.h):
        # x0 shortcut: 混合输入嵌入与当前层表示
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        
        # Value Embedding (交替层)
        ve = self.value_embeddings[str(i)](idx) if str(i) in self.value_embeddings else None
        
        # Block forward
        x = block(x, ve, cos_sin, self.window_sizes[i])
    
    # Step 3: Final RMSNorm
    x = norm(x)
    
    # Step 4: LM Head + Logit Softcap
    logits = self.lm_head(x)        # [B, T, vocab_size]
    logits = logits.float()         # 提升精度到 float32
    softcap = 15
    logits = softcap * torch.tanh(logits / softcap)  # 限制在 (-15, +15)
    
    # Step 5: Loss (如果有 targets)
    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # [B*T, vocab_size]
            targets.view(-1),                   # [B*T]
            ignore_index=-1,
            reduction=reduction
        )
        return loss
    return logits
```

### 9.1 x0 Shortcut 机制

这是一个重要的架构创新：

```
常规深层 Transformer:
  x_{l+1} = x_l + F_l(x_l)
  
x0 Shortcut:
  x_{l+1} = resid_lambda_l × x_l + x0_lambda_l × x_0 + F_l(x_l)
```

**直觉**：每一层都可以直接"参考"原始 token embedding。对于深层网络，这提供了一个梯度高速通道，让深层可以直接从输入提取信息。

**初始化**：
- `resid_lambdas = 1.0`（初始为标准残差）
- `x0_lambdas = 0.1`（初始 x0 路径很弱，让网络先学习常规路径）

### 9.2 Logit Softcap

```python
logits = softcap * torch.tanh(logits / softcap)  # softcap = 15
```

**作用**：限制 logits 的绝对值范围在 (-15, +15) 之间。

**为什么重要**：
- 防止极端大的 logits 导致数值不稳定（softmax overflow）
- 限制单个 token 的"置信度"，有助于训练稳定性
- 在推理时防止模型输出过于自信的预测

---

## 十、初始化策略（init_weights）

```python
@torch.no_grad()
def init_weights(self):
    """精心设计的初始化，确保初始状态稳定"""
    
    # 1. Token Embedding: 标准正态
    torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
    
    # 2. LM Head: 极小正态 (初始输出接近均匀分布)
    torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
    
    # 3. Transformer 矩阵: 均匀分布 [-s, s]
    #    s = sqrt(3) / sqrt(n_embd) → 方差 = s²/3 = 1/n_embd
    n_embd = self.config.n_embd
    s = 3**0.5 * n_embd**-0.5
    for block in self.transformer.h:
        torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
        torch.nn.init.zeros_(block.attn.c_proj.weight)   # 投影初始为 0
        torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
        torch.nn.init.zeros_(block.mlp.c_proj.weight)     # 投影初始为 0
    
    # 4. 标量参数
    self.resid_lambdas.fill_(1.0)   # 初始为标准残差
    self.x0_lambdas.fill_(0.1)      # 初始 x0 路径很弱
    
    # 5. Value Embeddings: 与 wte 相同的均匀初始化
    for ve in self.value_embeddings.values():
        torch.nn.init.uniform_(ve.weight, -s, s)
    
    # 6. VE Gate: 初始为 0 → sigmoid(0)=0.5 → gate=1.0 → 恒等映射
    for block in self.transformer.h:
        if block.attn.ve_gate is not None:
            torch.nn.init.zeros_(block.attn.ve_gate.weight)
    
    # 7. 类型转换: Embeddings → bf16 (节省内存)
    self.transformer.wte.to(dtype=torch.bfloat16)
    for ve in self.value_embeddings.values():
        ve.to(dtype=torch.bfloat16)
```

**初始化关键思想**：
- **残差路径初始为 0**（`c_proj` 和 `mlp.c_proj` 初始化为 0）→ 初始状态下 block 输出 ≈ 输入
- **LM head 极小** → 初始输出 logits ≈ 0 → 概率 ≈ 均匀 → 初始 loss ≈ log(vocab_size)
- **ResFormer gate 初始为恒等** → `v = v + 1.0 * ve` 自然过渡

---

## 十一、FLOPs 估算

```python
def estimate_flops(self):
    """估算每个 token 的 FLOPs（前向 + 反向）"""
    
    # 1. 矩阵运算: 6 × 参数量 (2x forward + 4x backward)
    nparams = sum(p.numel() for p in self.parameters())
    # 排除 embedding/value_embedding/标量参数 (这些不参与主要 matmul)
    nparams_exclude = (wte.numel() + value_embeds_numel + 
                      resid_lambdas.numel() + x0_lambdas.numel())
    
    # 2. 注意力 FLOPs (按窗口大小单独计算)
    h = self.config.n_head
    q = self.config.n_embd // self.config.n_head
    t = self.config.sequence_len
    attn_flops = 0
    for window_size in self.window_sizes:
        window = window_size[0]
        effective_seq = t if window < 0 else min(window, t)
        attn_flops += 12 * h * q * effective_seq  # 12 = 2(Q·K) + 2(softmax·V) ... 
    
    return 6 * (nparams - nparams_exclude) + attn_flops
```

**用途**：计算 MFU（Model FLOPs Utilization）：

```
MFU = (FLOPs_per_token × tokens_per_step) / (step_time × peak_FLOPs)
```

H100 BF16 峰值 = 989.5 TFLOPs。此项目通常达到约 40% MFU。

---

## 十二、参数分类（用于优化器分组）

```python
def num_scaling_params(self):
    return {
        'wte': wte.numel(),                    # token embedding
        'value_embeds': value_embeds.numel(),  # ResFormer VE
        'lm_head': lm_head.numel(),            # 输出投影
        'transformer_matrices': transformer_matrices.numel(),  # 所有其他矩阵
        'scalars': scalars.numel(),            # resid_lambdas + x0_lambdas
        'total': total,
    }
```

这个分类用于设置不同的学习率和优化器类型（见优化器文档）。

---

## 十三、架构创新点总结

| 特性 | 位置 | 作用 |
|-----|------|------|
| Pre-Norm on Embedding | forward line 274 | 嵌入后立即归一化 |
| x0 Shortcut | forward line 276 | 每层可访问原始输入 |
| ResFormer VE | Attention + init | 值嵌入改进注意力 |
| Q/K RMSNorm | Attention line 91 | 归一化 Q/K 稳定训练 |
| Alternating Window | _compute_window_sizes | 效率/效果平衡 |
| ReLU² MLP | MLP forward | 现代激活函数 |
| Logit Softcap | forward line 285 | 防止极端 logits |
| Zero-init Proj | init_weights | 残差初始为恒等 |
| bf16 Embeddings | init_weights | 节省显存 |
