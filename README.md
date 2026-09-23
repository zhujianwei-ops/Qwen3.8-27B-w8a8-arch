# Qwen 3.8-27B W8A8 架构文档

本仓库包含 Qwen 3.8-27B 模型的 W8A8 量化架构的完整可视化文档和技术说明。

## 📋 目录结构

```
.
├── Qwen3.8-27B_architecture.pdf          # 完整架构图（PDF格式）
├── Qwen3.8-27B_architecture.png          # 完整架构图（PNG格式）
├── Qwen3.8-27B_architecture.svg          # 完整架构图（SVG格式，可编辑）
├── Qwen3.8-27B_architecture_notes.md     # 架构详细说明文档
├── Qwen3.8-27B_architecture_assets/      # 架构组件资源目录
│   ├── A_overview.png                    # 架构总览图
│   ├── B_decoder_mlp.png                 # 解码器MLP组件图
│   ├── C_gated_deltanet.png              # 门控DeltaNet组件图
│   ├── D_full_attention.png              # 全注意力机制组件图
│   ├── E_vision.png                      # 视觉模块组件图
│   ├── F_mtp.png                         # MTP组件图
│   ├── G_quantization.png                # 量化细节图
│   ├── generate_architecture.py          # 架构图生成脚本
│   ├── preview.png                       # 预览图
│   ├── tensor_inventory.tsv              # 张量清单
│   └── verified_metadata.json            # 已验证的元数据
└── README.md                             # 本说明文档
```

## 🎯 主要内容

### 架构图文件

- **PDF版本** (`Qwen3.8-27B_architecture.pdf`): 适合打印和分发的高质量文档
- **PNG版本** (`Qwen3.8-27B_architecture.png`): 适合在网页和演示中使用
- **SVG版本** (`Qwen3.8-27B_architecture.svg`): 矢量图格式，可无损缩放和编辑

### 文档说明

- **架构笔记** (`Qwen3.8-27B_architecture_notes.md`): 包含模型架构的详细技术说明、参数配置和实现细节

### 组件资源

`Qwen3.8-27B_architecture_assets/` 目录包含：

- **A-G系列组件图**: 按字母顺序组织的各个架构组件的详细图示
  - A: 整体架构概览
  - B: 解码器和MLP层
  - C: 门控DeltaNet注意力机制
  - D: 全注意力机制
  - E: 视觉处理模块
  - F: 多任务处理(MTP)
  - G: W8A8量化实现细节

- **生成工具**: Python脚本用于生成和更新架构图
- **数据文件**: 张量清单和元数据用于验证和参考

## 🚀 快速开始

1. **查看完整架构**: 打开 `Qwen3.8-27B_architecture.pdf` 或 PNG 版本
2. **阅读技术说明**: 查看 `Qwen3.8-27B_architecture_notes.md` 了解详细实现
3. **浏览组件细节**: 进入 `Qwen3.8-27B_architecture_assets/` 查看各个组件的详细图示

## 📝 关于 W8A8 量化

W8A8 表示权重（Weight）和激活（Activation）都采用8位整数量化：
- **W8**: 权重使用8位整数表示
- **A8**: 激活值使用8位整数表示

这种量化方案在保持模型性能的同时，显著降低了内存占用和计算开销。

## 🔧 生成架构图

如需重新生成架构图，可以使用资源目录中的脚本：

```bash
cd Qwen3.8-27B_architecture_assets/
python generate_architecture.py
```

## 📄 许可证

请遵循 Qwen 模型的相关许可证要求。

## 🤝 贡献

如发现文档中的错误或有改进建议，欢迎提交 Issue 或 Pull Request。
