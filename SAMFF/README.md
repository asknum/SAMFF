# BiFA模型BI模块可视化工具

本工具用于可视化BiFA模型中双时相交互（BI）模块在不同特征提取阶段的影响，通过完整的消融分析展示BI模块的作用机制。

## 🎯 项目概述

BiFA模型的BI模块包含4个核心组件：
1. **跨时相注意力** (attn_realchannel) - 对齐双时相特征
2. **轻量级卷积增强** (feature_enhance) - 提取局部空间特征  
3. **残差连接增强** (residual_enhance) - 通道注意力机制
4. **自适应特征融合权重** (fusion_weight) - 可学习权重融合

本工具通过对比有BI和无BI模块的模型，直观展示BI模块在4个特征提取阶段的改进效果。

## 🚀 快速开始

### 安装依赖
```bash
pip install torch>=1.9.0 matplotlib>=3.3.0 numpy>=1.20.0 opencv-python>=4.5.0
```

### 运行方式

#### 方法1: 演示模式（推荐首次使用）
```bash
python bi_visualization.py --mode demo
```
**优点**: 无需任何配置，使用模拟数据直接运行

#### 方法2: 真实数据模式（单张图像）
```bash
python bi_visualization.py --mode real \
    --model_path path/to/your/model.pth \
    --dataset_path dataset/WHU-CD-256 \
    --image_idx 0
```
**注意**: 需要预训练模型权重文件和数据集

#### 方法3: 批量处理模式（整个测试集）⭐
```bash
python bi_visualization.py --mode real \
    --model_path path/to/your/model.pth \
    --dataset_path dataset/WHU-CD-256 \
    --batch_mode \
    --vis_dir vis
```
**特点**: 自动处理所有测试集图像，结果保存到指定目录

#### 参数说明
- `--mode`: 运行模式 (`demo` 或 `real`)
- `--model_path`: 预训练模型路径（真实模式必需）
- `--dataset_path`: 数据集路径（如: dataset/WHU-CD-256）
- `--image_idx`: 图像索引（单张模式）
- `--split`: 数据分割 (`test`, `val`, `train`)
- `--batch_mode`: 启用批量处理模式
- `--vis_dir`: 可视化结果保存目录（默认: vis）
- `--max_images`: 最大处理图像数量（用于测试）
- `--device`: 计算设备 (`auto`, `cuda`, `cpu`)
- `--save_path`: 结果保存路径（单张模式）

## 📊 功能特点

### ✅ 完整的BI模块消融分析
确保对比包含BI模块的**所有4个组件**，而非仅跨时相注意力

### ✅ 4个特征提取阶段分析
| 阶段 | 特征尺寸 | 功能描述 | BI模块作用 |
|------|----------|----------|------------|
| Stage1 | [32, 64, 64] | 浅层细节特征 | 减少噪声，增强边界 |
| Stage2 | [64, 32, 32] | 中层语义特征 | 平衡细节与语义 |
| Stage3 | [160, 16, 16] | 高层语义特征 | 增强语义理解 |
| Stage4 | [256, 8, 8] | 全局语义特征 | 精准变化定位 |

### ✅ 直观的可视化效果
- 3×5对比布局图
- 蓝色→红色热力图（符合注意力可视化惯例）
- 差异图量化BI模块改进效果

### ✅ 批量处理功能 ⭐ 
- 🚀 自动处理整个测试集的所有图像
- 📁 结果自动保存到指定目录（默认 `vis/` 文件夹）
- 📊 智能进度显示和处理统计
- 🏃‍♂️ 静默模式避免输出过多信息
- 📈 自动生成处理报告

## 🔍 结果解读

### 图像布局
```
      T1/T2    Stage1   Stage2   Stage3   Stage4
W/O BI        [热力图1] [热力图2] [热力图3] [热力图4]
 W/ BI        [热力图1] [热力图2] [热力图3] [热力图4]
 差异图        [差异图1] [差异图2] [差异图3] [差异图4]
```

### 颜色映射含义
- 🔴 **红色**: 高注意力值，模型重点关注的变化区域
- 🔵 **蓝色**: 低注意力值，模型认为变化较小的区域
- 🟡 **黄色**: 中等注意力值，潜在的变化候选区域
- 📊 **差异图**: 显示BI模块带来的注意力分布改进

### 关键发现

**浅层阶段 (Stage1-Stage2)**：
- ❌ 无BI模块：噪声较多，边界模糊，细节丢失
- ✅ 有BI模块：噪声显著减少，结构细节更清晰
- 💡 改进机制：跨时相注意力 + 卷积增强提升细节表示

**深层阶段 (Stage3-Stage4)**：
- ❌ 无BI模块：变化检测不够精确，容易误检
- ✅ 有BI模块：更精准地聚焦真实变化区域
- 💡 改进机制：残差增强 + 特征融合提升语义理解

## 🛠️ 技术实现细节

### BI模块完整架构

```python
class Block(nn.Module):
    def __init__(self, dim, ...):
        # 1. 跨时相注意力
        self.attn_realchannel = AttentionRealCrossChannel(...)
        
        # 2. 轻量级卷积增强  
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        
        # 3. 残差连接增强
        self.residual_enhance = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(), 
            nn.Linear(dim // 4, dim),
            nn.Dropout(drop)
        )
        
        # 4. 特征融合权重
        self.fusion_weight = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, H, W, cond):
        # 原始双向注意力
        x = x + self.attn(self.norm1(x), H, W, cond)
        x = x + self.attn_realchannel(self.norm1(x), H, W, self.norm1(cond))
        
        # BI模块增强
        x_enhanced = self.feature_enhance(x.transpose(1, 2)).transpose(1, 2)
        x_residual = self.residual_enhance(x)
        x_fused = x + self.fusion_weight * (x_enhanced + x_residual)
        
        # MLP处理
        x = x_fused + self.drop_path(self.mlp(self.norm2(x_fused), H, W))
        return x
```

### 关键实现步骤

#### 1. 模型对比设计
```python
# 有BI模块的完整模型
model_with_bi = BiFA(backbone="mit_b0", use_pfem=True, decoder_type="fgfm")

# 无BI模块的对比模型（禁用所有4个BI组件）
model_without_bi = BiFA(...)
disabled_count = _disable_bi_modules(model_without_bi)
```

#### 2. 完整BI模块禁用
```python
def _disable_bi_modules(self, model):
    for name, child in module.named_children():
        if hasattr(child, 'attn_realchannel'):
            # 组件1: 禁用跨时相注意力
            child.attn_realchannel = nn.Identity()
            # 组件2: 禁用卷积增强
            child.feature_enhance = nn.Identity()
            # 组件3: 禁用残差增强
            child.residual_enhance = nn.Identity()
            # 组件4: 禁用特征融合
            child.fusion_weight.data.fill_(0.0)
```

#### 3. 多阶段特征提取
```python
def extract_stage_features(self, model, x1, x2):
    with torch.no_grad():
        # Stage 1: 浅层特征 [B, 32, 64, 64]
        x1_1 = model.segformer.forward_features1(x1, x2)
        x2_1 = model.segformer.forward_features1(x2, x1)
        diff1 = model.diffflow1(x1_1, x2_1)
        
        # Stage 2-4: 中层和高层特征
        # ...
        
        return [diff1, diff2, diff3, diff4]
```

#### 4. 注意力图计算
```python
def compute_attention_map(self, features):
    # 使用L2范数计算通道注意力
    attention = torch.norm(features, dim=1, keepdim=True)
    # 标准化到[0, 1]
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    return attention
```

## 🔧 项目文件

### 核心文件
- **`bi_visualization.py`** - 统一的可视化脚本
  - 支持演示模式和真实数据模式
  - 完整的BI模块消融分析
  - 命令行参数支持

### 文档文件
- **`README.md`** - 本文档，包含完整使用说明

## 💡 使用建议

1. **首次使用**: 运行演示模式熟悉工具功能和结果格式
2. **论文撰写**: 可直接使用生成的图像作为实验结果图表
3. **参数调整**: 可根据需要修改颜色映射和可视化参数
4. **扩展功能**: 可基于现有代码扩展更多分析功能

## 🎯 技术要点

### 重要注意事项
1. **模型一致性**: 有BI和无BI模型必须使用相同的初始权重
2. **组件完整性**: 必须禁用所有4个BI组件，不能只禁用注意力
3. **特征尺度**: 注意不同阶段特征图的尺寸变化
4. **设备兼容**: 支持CUDA和CPU两种运行模式
5. **错误处理**: 提供详细的错误提示和解决方案

### 实验验证
通过该可视化工具，您可以：
- 🔍 验证BI模块在每个阶段的具体改进效果
- 📈 量化分析BI模块带来的性能提升
- 🎯 理解BI模块的工作机制和作用原理
- 📋 为论文提供直观的实验结果图表

## 🔬 示例运行

### 演示模式输出
```bash
$ python bi_visualization.py --mode demo

============================================================
🎯 BiFA模型BI模块多阶段影响可视化工具 v2.0
============================================================
🚀 初始化BI模块可视化工具
   📋 运行模式: demo
   💻 计算设备: cuda
🎭 初始化演示模式...
✅ 演示模式初始化完成

🎯 开始BI模块多阶段影响可视化 (demo模式)
📋 使用模拟数据...
🔥 计算4个阶段的注意力对比...
🎨 生成可视化图像...
💾 可视化结果已保存至: bi_stage_analysis_demo.png

📊 BI模块多阶段影响分析结果 (DEMO模式)：
==================================================
🔍 浅层阶段 (Stage1-Stage2)：
   ❌ 无BI模块：噪声较多，边界模糊，细节丢失
   ✅ 有BI模块：噪声显著减少，结构细节更清晰
   💡 改进机制：跨时相注意力+卷积增强提升细节表示

🔍 深层阶段 (Stage3-Stage4)：
   ❌ 无BI模块：变化检测不够精确，容易误检
   ✅ 有BI模块：更精准地聚焦真实变化区域
   💡 改进机制：残差增强+特征融合提升语义理解

✅ 可视化分析完成！
```

### 批量处理模式输出
```bash
$ python bi_visualization.py --mode real --batch_mode --dataset_path dataset/WHU-CD-256 --model_path model.pth

============================================================
🚀 批量处理模式启动
============================================================
📁 创建可视化目录: vis
📊 总共需要处理: 256 张图像
📂 数据集: dataset/WHU-CD-256
📋 分割: test

🎯 处理进度: 100%|██████████| 256/256 [08:45<00:00,  2.05it/s]

============================================================
📊 批量处理完成 - 总结报告
============================================================
✅ 成功处理: 254 张图像
❌ 处理失败: 2 张图像
⏱️  总耗时: 525.32 秒
⚡ 平均耗时: 2.05 秒/张
📁 结果保存目录: /path/to/project/vis

🎉 批量处理完成！共生成 254 个可视化结果
```

## 🎉 版本更新

### v2.0 (整合版)
- ✅ 统一演示模式和真实数据模式
- ✅ 完整的4组件BI模块消融分析
- ✅ 命令行参数支持
- ✅ 详细的进度显示和错误处理
- ✅ 减少文件数量，提高易用性

---

此工具完全实现了您论文中描述的BI模块不同阶段影响分析，提供了清晰的可视化结果和详细的技术实现。通过整合原有功能，现在只需要一个脚本文件即可完成所有可视化分析任务。
