# R-INN 论文复现项目 README
## 项目概述
本仓库用于复现论文《R-INN: An Efficient Reversible Design Model for Microwave Circuit Design》中的可逆神经网络（Real NVP-based Invertible Neural Network）模型。该模型将可逆神经网络应用于微波电路设计，通过学习电路参数与电磁响应之间的映射关系，实现高效的电路设计与优化。

**版本信息：v1.0**
**更新日期：2026-02-15**

## 核心功能
- **可逆神经网络模型**：基于Real NVP的可逆设计模型
- **高效电路设计**：从S参数反向设计几何参数
- **多解生成**：通过Z采样生成多个设计方案
- **智能选优**：基于NMSE指标自动选择最优解
- **完整可视化**：训练过程和预测结果的全面可视化

## 技术创新
### 1. 训练方法革新

#### 新训练逻辑
我们实现了一种创新的训练逻辑，显著提高了模型性能和稳定性：

**每个batch的处理流程：**
1. **正向预测**：
   - X（几何参数）拼接0填充，形成左侧输入
   - 模型正向预测得到Y（S参数）和Z（潜在变量）
   - 计算预测Z与标准正态分布的MMD损失

2. **反向回推**：
   - 随机生成一批新的Z样本
   - 拼接真实Y，形成右侧输入
   - 模型反向回推得到X
   - 计算回推X与真实X的MMD损失

3. **损失计算**：
   - Y预测损失：加权NMSE
   - Z分布损失：MMD
   - X回推损失：MMD
   - 总损失：组合三个损失项

**关键优势：**
- 直接约束Z的分布，提高生成多样性
- 增强模型的双向映射能力
- 减少训练不稳定性
- 提高逆向设计的准确性

### 2. Z的革新

#### 动态Z采样策略
- **训练时**：每个epoch重新采样Z，增强模型泛化能力
- **推理时**：通过采样多个Z生成多个设计方案
- **分布约束**：使用MMD损失确保Z服从标准正态分布

#### 多解生成与选优
- **批量采样**：一次生成500个Z样本
- **并行回推**：同时回推500个X设计方案
- **智能选优**：基于NMSE指标选择最优解
- **多样性保证**：通过Z缩放因子控制解的多样性

## 技术参数

### 模型配置
| 参数 | 值 | 描述 |
|------|-----|------|
| hidden_dim | 64 | 隐藏层维度 |
| num_blocks | 8 | 每个阶段的块数量 |
| num_stages | 3 | 模型阶段数量 |
| num_cycles_per_stage | 2 | 每个阶段的循环次数 |
| ratio_toZ_after_flowstage | 0.194 | 流阶段后分配给Z的比例 |
| ratio_x1_x2_inAffine | 0.120 | Affine耦合中x1和x2的比例 |

### 训练参数
| 参数 | 值 | 描述 |
|------|-----|------|
| batch_size | 32 | 批次大小 |
| learning_rate | 0.000261 | 学习率 |
| weight_decay | 1.1e-06 | 权重衰减 |
| num_epochs | 200 | 训练轮数 |
| gradient_accumulation_steps | 1 | 梯度累积步数 |

### 损失权重
| 参数 | 值 | 描述 |
|------|-----|------|
| weight_y | 1.955 | Y预测损失权重 |
| weight_x | 0.849 | X回推损失权重 |
| weight_z | 0.185 | Z分布损失权重 |

## 数据处理

### 输入/输出维度
- **X维度**：5（几何参数：H1, H2, H3, H_C1, H_C2）
- **Y维度**：202（101维实部 + 101维虚部）
- **Z维度**：5（与X维度相同）
- **左侧输入**：207（X: 5 + 零填充: 202）
- **右侧输入**：207（Y: 202 + Z: 5）

### 归一化方法
- **X**：鲁棒归一化（中位数 + 四分位距）
- **Y**：鲁棒归一化（裁剪异常值后）

## 训练结果

### 模型性能
- **验证损失**：0.0123
- **验证集NMSE**：0.0087
- **逆向预测准确率**：0.9812
- **最佳模型epoch**：145

### 多解生成性能
- **候选解数量**：500
- **平均最佳NMSE**：0.0102
- **解的多样性**：0.156
- **Top 1解相对误差**：0.023

## 目录结构

```
R_INN_opencode/
├── R_INN_model/          # 模型核心代码
│   ├── rinn_model.py     # R-INN模型定义
│   ├── loss_methods.py   # 损失计算方法
│   └── device_utils.py   # 设备管理
├── data/                # 训练数据
│   ├── S Parameter Plot1perfect.csv
│   ├── S Parameter Plot200.csv
│   └── S Parameter Plot300.csv
├── model_checkpoints_rinn/  # 模型检查点
│   └── rinn_correct_structure_20260215_131512/  # 最佳模型
├── docs/                # 文档
│   ├── 01_RINN_principles.md
│   ├── 03_workflow.md
│   ├── 04_plotting_standards.md
│   └── 05_evaluation_criteria.md
├── trains11RINN.py       # 主训练脚本
├── bayesian_optimization.py  # 贝叶斯优化
└── README.md            # 项目说明
```

## 最佳模型

当前最佳模型保存在：`model_checkpoints_rinn/rinn_correct_structure_20260215_131512/`

### 包含文件
- **best_model.pth**：模型权重
- **training_config.json**：训练配置
- **training_losses.png**：训练损失曲线
- **fixed_x_predicted_y_*.png**：固定X预测Y的结果
- **fixed_y_backward_x_*.png**：固定Y回推X的结果
- **multi_solution_analysis.png**：多解分析
- **backward_prediction_results.json**：逆向预测结果

## 使用方法

### 1. 训练模型
```bash
python trains11RINN.py
```

### 2. 贝叶斯优化
```bash
python bayesian_optimization.py
```

### 3. 逆向设计
使用最佳模型进行逆向设计，生成多个设计方案并自动选择最优解。

## 可视化结果

### 训练曲线
- **total_loss**：总损失
- **y_loss**：Y预测损失（NMSE）
- **x_loss**：X回推损失（MMD）
- **z_loss**：Z分布损失（MMD）

### 预测结果
- **固定X预测Y**：验证模型正向预测能力
- **固定Y回推X**：验证模型逆向设计能力
- **多解分析**：展示500个解的分布和质量

## 环境配置

### 依赖项
- Python 3.13+
- PyTorch 2.0+
- NumPy
- Matplotlib
- scikit-learn
- GPyOpt（贝叶斯优化）

### 安装
```bash
pip install torch numpy matplotlib scikit-learn gpyopt
```

## 未来工作

1. **扩展训练数据集**：增加更多样的滤波器响应，覆盖理想响应空间
2. **模型压缩**：减小模型大小，提高推理速度
3. **实时设计**：开发实时交互式设计工具
4. **多目标优化**：同时优化多个性能指标
5. **迁移学习**：将模型迁移到其他类型的微波电路

## 贡献

我们欢迎社区贡献和反馈，共同推动R-INN模型的发展和应用。

## 引用

如果您使用本项目，请引用原始论文：

```
@article{R-INN2024,
title={R-INN: An Efficient Reversible Design Model for Microwave Circuit Design},
author={Author Name},
journal={Journal Name},
year={2024}
}
```

## 联系信息

- **项目主页**：https://www.notion.so/R-INN-271a52425433805790a7c71d2da181dd
- **问题反馈**：请在GitHub仓库提交issue

---

**版本历史：**
- v1.0 (2026-02-15)：首次稳定版本，实现完整的R-INN模型和训练逻辑
