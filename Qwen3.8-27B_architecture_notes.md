# Qwen3.8-27B 本地模型结构说明

分析目录：`/home1/model/Qwen3.8-27B-w8a8`。生成日期：2026-09-22。

主图：`Qwen3.8-27B_architecture.svg`（矢量、内嵌中文字体）、`Qwen3.8-27B_architecture.png`（4800 × 5895）和 `Qwen3.8-27B_architecture.pdf`（PNG 的单页 PDF 版本）。七个模块的独立 PNG 在 `Qwen3.8-27B_architecture_assets/`。

## 依据与边界

模型名称沿用本地目录和 README。配置中的架构为 `Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`，`language_model_only=false`，任务为 `image-text-to-text`。该记录描述指定的本地检查点，不以架构类中的 “3_5” 推断目录名称错误，也不把其他同名模型的网上参数替代为本地参数。

直接读取了 config、量化说明、量化 YAML、权重索引，以及全部 10 个 safetensors 文件的 JSON 头。共核对 2079 个张量条目，包含 scale/offset。没有把约 32 GB 权重加载进内存，没有执行模型前向或精度测试。

模块数据流参考本机 Transformers 5.5.4 的 qwen3_5 实现，以及工作目录内 SGLang 的 qwen3_5 / qwen3_5_mtp 实现。模型目录未包含自己的 modeling 源码；配置记录的 transformers_version 为 5.8.0.dev0，而目录 README 的运行建议为 5.14.0。图示的形状和量化属性以实际检查点为准，具体融合算子、缓存布局和 MTP 调度由运行引擎决定。

## 1. 全局结构

`文本 IDs → Token Embedding` 与 `图像/视频 → Vision Encoder → Patch Merger` 在输入序列处合流：视觉特征替换图像/视频占位 token 的 embedding，而不是通过单独的 cross-attention 层注入。

随后是 `64 层混合注意力 Dense Decoder → RMSNorm → LM Head → logits`。

| 配置项 | 实际值 |
|---|---:|
| text hidden_size | 5120 |
| intermediate_size | 17408 |
| num_hidden_layers | 64 |
| vocab_size | 248320 |
| max_position_embeddings | 262144（256K，配置上限） |
| tie_word_embeddings | false |
| RMSNorm eps | 1e-6 |
| attention_bias / dropout | false / 0 |
| 架构形态 | Dense；权重中无 MoE experts/router |

主干层从 1 开始编号时：`[GDN, GDN, GDN, GQA] × 16`。全注意力在第 4、8、12、…、64 层；对应权重的零起始索引为 3、7、11、…、63。

Token Embedding 与 LM Head 分别有 `[248320,5120]` 的 BF16 权重，每个包含 1,271,398,400 个元素。两者不绑权。所有层的残差维度都是 5120；注意力内的 Q 或 V 总维度可以是 6144，因此不能把 `hidden_size / num_attention_heads` 当作这里的 head_dim。

## 2. Decoder 公共骨架与 FFN

每层为：

```text
h1 = h + TokenMixer(input_layernorm(h))
h2 = h1 + MLP(post_attention_layernorm(h1))
MLP(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

`gate_proj` 和 `up_proj` 的权重都是 `[17408,5120]`；`down_proj` 是 `[5120,17408]`。主干 64 层和 MTP 1 层均为 W8A8_DYNAMIC。语言主干的普通 RMSNorm 实现使用 `(1 + weight)` 的缩放参数化；GDN 的 gated RMSNorm 使用其自身的缩放权重。它们不是视觉模块的 LayerNorm。

## 3. Gated DeltaNet 线性注意力（48 层）

| 模块 | 权重形状 | 保存 dtype / 量化 |
|---|---|---|
| in_proj_qkv | [10240,5120] | INT8 / W8A8_DYNAMIC |
| in_proj_z | [6144,5120] | INT8 / W8A8_DYNAMIC |
| in_proj_a / in_proj_b | 各 [48,5120] | BF16 / FLOAT |
| conv1d | [10240,1,4] | BF16 / FLOAT |
| A_log / dt_bias | 各 [48] | BF16 / FLOAT |
| norm.weight | [128] | BF16 / FLOAT |
| out_proj | [5120,6144] | BF16 / FLOAT |

QKV 经 10240 通道、kernel=4 的因果 depthwise Conv1D 和 SiLU 后拆分：Q=2048、K=2048、V=6144。Q/K 各 16 头、每头 128；V 为 48 头、每头 128。Q/K 经 L2 归一化，并按头重复三份以匹配 48 个 value heads。递推实现还按 `1/sqrt(128)` 缩放 Q。

每个 value head 有 `beta=sigmoid(b)` 与 `g=-exp(A_log)*softplus(a+dt_bias)`。用列向量表达，单步逻辑可以写为：

```text
S_decay = exp(g_t) * S_previous
error   = v_t - transpose(S_decay) @ k_t
S_t     = S_decay + beta_t * outer(k_t, error)
y_t     = transpose(S_t) @ q_t
```

S 的逻辑形状为 `[B,48,128,128]`；q 已包括归一化/缩放。prefill 可以使用分块并行实现，decode 用递推状态；它不保存每个历史 token 的传统 K/V 张量。还需要短卷积历史缓存，物理存储布局视引擎而定。配置 `mamba_ssm_dtype=float32` 说明递推状态的预期精度，不代表 checkpoint 的 `A_log` 是 FP32（实际为 BF16）。

最后：每头 128 维的 RMSNorm → 乘 `Swish(z)` → 拼为 6144 维 → `out_proj` 回到 5120。

**门控语义核对：`output_gate_type=swish` 控制 GDN 输出归一化门控；它不控制下面全注意力中的 sigmoid gate。** SGLang 的配置文档和两个模块实现均支持该区分。

## 4. Gated GQA 全注意力（16 层）

| 模块 | 权重形状 | dtype |
|---|---|---|
| q_proj（包含 Q 与 gate） | [12288,5120] | INT8 |
| k_proj | [1024,5120] | INT8 |
| v_proj | [1024,5120] | INT8 |
| o_proj | [5120,6144] | INT8 |
| q_norm / k_norm | 各 [256] | BF16 |

q_proj 输出按头组织为 `24 × (256 + 256)`，分别得到 Q 与输出 gate。二者各 6144 维；不能把 12288 都当作 Q。K/V 各 4 头、每头 256，Q:KV 头比为 6:1。Q/K 都先做逐头 RMSNorm，然后在每头前 64 维应用部分 M-RoPE，另外 192 维不旋转。

RoPE：`partial_rotary_factor=0.25`，`rope_theta=10000000`，`mrope_interleaved=true`，`mrope_section=[11,11,10]`（32 个频率对，对应 64 个旋转维度）。

```text
attention = softmax(QK^T / sqrt(256) + causal_mask) @ V
output    = o_proj(flatten(attention) * sigmoid(gate))
```

这里 `sqrt(256)=16`。每层 K/V cache 的逻辑形状分别为 `[B,4,T,256]`，无需把重复后的 24 个头都存入缓存。图示的张量布局是逻辑布局，不限定后端的物理布局。

## 5. 视觉模块

| 项目 | 实际结构 |
|---|---|
| 输入 | RGB 图像 / 视频，in_channels=3 |
| patch embedding | Conv3D，kernel=stride=(2,16,16)，3→1152，含 bias |
| patch 权重 | [1152,3,2,16,16]，BF16 |
| 位置嵌入 | [2304,1152]，空间网格 48×48，插值到输入网格 |
| 视觉深度 | 27 个 VisionBlock |
| 注意力 | 非因果 MHA，16 heads × 72；Q/K 使用空间 RoPE |
| QKV / proj 权重 | [3456,1152] / [1152,1152]，INT8 |
| FFN | 1152→4304→1152，GELU(tanh approximation) |
| FFN fc1 / fc2 权重 | [4304,1152] INT8 / [1152,4304] BF16 |
| 归一化 | pre-LayerNorm(1152)，eps=1e-6，两段残差 |
| DeepStack | deepstack_visual_indexes=[]，无该注入分支 |

单个视觉块为 `x += MHA(LayerNorm(x))`，然后 `x += MLP(LayerNorm(x))`。注意力按输入的空间片段处理，图中“非因果”不表示所有视频帧必然进行一次全局时空注意力。

Merger 先对 1152 维做 LayerNorm，再把每个 2×2 空间组拼为 4608 维：`[N,1152] → [N/4,4608] → Linear(4608,4608) → GELU → Linear(4608,5120)`。两个线性层都有 bias，Merger 全部 BF16。最终视觉 token 数为 `grid_t * grid_h * grid_w / 4`，N 指合并前 patch 数，不能把 2304 个位置嵌入误读为固定视觉 token 数。

## 6. MTP 辅助分支

检查点中确实存在 `mtp.fc`、`mtp.pre_fc_norm_embedding`、`mtp.pre_fc_norm_hidden`、`mtp.layers.0.*` 和 `mtp.norm`，不是仅有配置字段。

逻辑结构：主干隐藏状态与后移一个 token 的 embedding 分别做 RMSNorm → 按 `[embedding, hidden]` 拼接为 10240 → `mtp.fc` 投影至 5120 → 1 个全注意力 Decoder → `mtp.norm` → 复用主干 LM Head。

`mtp.fc.weight=[5120,10240]`，BF16；MTP 的 attention/MLP 形状与主干全注意力层一致，投影权重为 INT8。`mtp_use_dedicated_embeddings=false`；不存在独立的 `mtp.embed_tokens` / `mtp.lm_head` 权重。复用 embedding/head 不表示主干 embedding 与 head 自身绑权。

MTP 是独立辅助模块，不能把主干写成 65 层。是否启用 speculative decoding、如何传递隐藏状态与位置等由运行引擎决定。现有 Transformers 通用主干实现可忽略 mtp 权重；本图根据本地 SGLang MTP 路径说明辅助模块连接关系，不声称已验证某个实际部署运行方式。

## 7. 量化与参数统计

量化 YAML：激活为 per-token INT8 symmetric minmax；权重为 per-channel INT8 symmetric minmax。保存说明为 `W8A8_DYNAMIC`；各量化权重附有 FP32 `weight_scale` / `weight_offset`。`FLOAT` 是量化类别而不是 FP32 同义词：大多数 FLOAT 权重实际为 BF16，视觉量化投影的 bias 为 FP32。

已对全部层验证：

- 文本主干/MTP 的 MLP 与全注意力 Q/K/V/O 投影：W8A8_DYNAMIC。
- GDN 的 in_proj_qkv / in_proj_z：W8A8_DYNAMIC；out_proj、in_proj_a/b、conv1d、norm、A_log、dt_bias：FLOAT / BF16。
- VisionBlock 的 qkv、proj、linear_fc1 权重：W8A8_DYNAMIC；linear_fc2：BF16。
- Patch embedding、视觉位置嵌入、Merger、文本 embedding / LM head、各 norm 权重：BF16。

| 部分 | 参数元素数（不含量化 scale/offset） |
|---|---:|
| 64 层 Decoder | 24,353,196,544 |
| Token Embedding | 1,271,398,400 |
| LM Head | 1,271,398,400 |
| 最终 RMSNorm | 5,120 |
| Vision Encoder + Merger | 460,730,096 |
| 主干小计（含视觉） | 27,356,728,560 |
| MTP | 424,699,392 |
| 全部模型参数 | 27,781,427,952 |

INT8 参数 23,466,457,088 个（84.47%）；BF16 参数 4,314,730,240 个；FP32 参数 240,624 个。上述计数包含 bias、norm 与状态系数，不含量化 scale/offset。

所有存储张量（含量化元数据）有效数据共 32,128,509,248 字节，即 32.128509 GB / 29.922006 GiB，与索引 metadata.total_size 一致。这个数字不是模型实际推理显存需求，不包含激活、KV/递推缓存、运行时工作区或文件头。

## 8. 复现与完整证据

```bash
python3 /home/zhujianwei/Qwen3.8-27B_architecture_assets/generate_architecture.py
```

需要 Pillow，以及脚本 FONT 指向的中文 TrueType 字体。脚本重新读取源目录的元数据、验证全部层的关键权重维度与 dtype，再输出图文件；不导入 torch、也不加载张量数据。SVG 内嵌原始中文字体，PNG/PDF 为同一场景的栅格输出。

- `Qwen3.8-27B_architecture_assets/verified_metadata.json`：配置、精确计数、输入文件 SHA-256、分片 JSON 头 SHA-256。
- `Qwen3.8-27B_architecture_assets/tensor_inventory.tsv`：全部 2079 个张量的名称、形状、dtype、量化类型及分片。
- `Qwen3.8-27B_architecture_assets/A_overview.png` 至 `G_quantization.png`：七个放大分图。

本图以静态结构分析为目的。没有执行前向测试，因此不对实际后端兼容性、吞吐或数值精度作测试结论。
