# R-INN 工作流程说明书

## 概述

本文档描述使用R-INN模型进行**正向预测**、**反向设计**和**训练**的完整工作流程，基于v1.0版本的最新实现。

---

## 一、训练流程

### 1.1 新训练逻辑

```
每个batch处理：
┌─────────────────────┐     ┌─────────────────────┐
│ 正向预测流程        │     │ 反向回推流程        │
├─────────────────────┤     ├─────────────────────┤
│ 1. X拼接0填充       │     │ 1. 随机生成Z        │
│ 2. 正向预测Y和Z     │     │ 2. 拼接真实Y        │
│ 3. 计算Z的MMD损失   │     │ 3. 反向回推X        │
│ 4. 计算Y的NMSE损失  │     │ 4. 计算X的MMD损失   │
└─────────────────────┘     └─────────────────────┘
          ↓                           ↓
┌─────────────────────────────────────┐
│ 组合损失：weight_y*Y_loss +        │
│ weight_x*X_loss + weight_z*Z_loss   │
└─────────────────────────────────────┘
```

### 1.2 详细步骤

#### 步骤1：数据准备
- 加载训练数据：`S Parameter Plot200.csv` 和 `S Parameter Plot300.csv`
- 加载验证数据：包含 `S Parameter Plot1perfect.csv` 的5个样本
- 数据归一化：使用鲁棒归一化（中位数 + 四分位距）

#### 步骤2：模型初始化
- 配置模型参数：hidden_dim=64, num_blocks=8, num_stages=3
- 配置训练参数：batch_size=32, learning_rate=0.000261
- 配置损失权重：weight_y=1.955, weight_x=0.849, weight_z=0.185

#### 步骤3：训练循环
- 每个epoch重新采样Z，增强模型泛化能力
- 使用梯度累积和学习率调度
- 早停机制：patience=60

#### 步骤4：模型保存
- 保存最佳模型：基于验证损失
- 保存训练配置和损失曲线

---

## 二、正向预测：X → Y

### 2.1 流程图

```
几何参数 X
    ↓
标准化 (鲁棒归一化)
    ↓
X拼接0填充 → 左侧输入
    ↓
模型正向预测
    ↓
提取Y (前202维)
    ↓
反标准化 (恢复原始尺度)
    ↓
得到预测的S参数
```

### 2.2 详细步骤

#### 步骤1：准备几何参数 X
- 确保参数在有效范围内
- 参考几何参数分析文档

#### 步骤2：数据标准化
```python
# 使用鲁棒归一化
X_normalized = (X - X_median) / X_iqr
```

#### 步骤3：模型推理
```python
# 左侧输入：X + 零填充
left_input = np.concatenate((X_normalized, np.zeros((1, 202), dtype=np.float32)), axis=1)
left_input = torch.FloatTensor(left_input).to(device)

# 正向预测
with torch.no_grad():
    predicted_right, _, _ = model(left_input, return_intermediate=True)
    predicted_y_normalized = predicted_right[:, :202]
```

#### 步骤4：解析输出
```python
# 反标准化
predicted_y = predicted_y_normalized.cpu().numpy() * y_iqr + y_median

# 解析实部和虚部
Re_S11 = predicted_y[:, :101]  # 前101维：实部
Im_S11 = predicted_y[:, 101:]  # 后101维：虚部

# 计算幅度和dB
mag_S11 = np.sqrt(Re_S11**2 + Im_S11**2)
dB_S11 = 20 * np.log10(mag_S11)
```

---

## 三、反向设计：Y → X

### 3.1 流程图

```
目标响应 Y_target
    ↓
标准化
    ↓
批量采样 Z ~ N(0,1) (500个样本)
    ↓
拼接 [Y_target, Z] (500个输入)
    ↓
并行反向回推
    ↓
输出 X_candidates (500个候选)
    ↓
反标准化
    ↓
正向验证：X_candidate → Y_pred
    ↓
计算每个候选的NMSE
    ↓
选择NMSE最小的最佳解
```

### 3.2 详细步骤

#### 步骤1：准备目标响应 Y_target
- 确保Y_target维度为202（101维实部 + 101维虚部）
- 进行鲁棒归一化

#### 步骤2：批量采样Z
```python
# 生成500个Z样本
num_candidates = 500
z_scale = 1.2  # 控制多样性
z_candidates = np.random.randn(num_candidates, z_dim).astype(np.float32) * z_scale
```

#### 步骤3：并行反向回推
```python
# 批量处理500个输入
y_repeated = np.repeat(Y_target_normalized, num_candidates, axis=0)
right_inputs = np.concatenate((y_repeated, z_candidates), axis=1)
right_inputs = torch.FloatTensor(right_inputs).to(device)

# 并行反向回推
with torch.no_grad():
    reconstructed_lefts, _ = model.inverse(right_inputs)
    reconstructed_xs_normalized = reconstructed_lefts[:, :x_dim]
    reconstructed_xs = reconstructed_xs_normalized.cpu().numpy() * x_iqr + x_median
```

#### 步骤4：智能选优
```python
# 对每个候选进行正向验证
nmse_errors = []
for i in range(num_candidates):
    # 正向预测
    x_norm = (reconstructed_xs[i:i+1] - x_median) / (x_iqr + 1e-8)
    left_input = np.concatenate((x_norm, np.zeros((1, 202), dtype=np.float32)), axis=1)
    left_input = torch.FloatTensor(left_input).to(device)
    
    with torch.no_grad():
        predicted_right, _, _ = model(left_input, return_intermediate=True)
        predicted_y_normalized = predicted_right[:, :y_dim]
        predicted_y = predicted_y_normalized.cpu().numpy() * y_iqr + y_median
    
    # 计算NMSE
    mse = np.mean((predicted_y[0] - Y_target[0]) ** 2)
    variance = np.var(Y_target[0])
    nmse = mse / (variance + 1e-8)
    nmse_errors.append(nmse)

# 选择最佳解
best_idx = np.argmin(nmse_errors)
best_reconstructed_x = reconstructed_xs[best_idx]
```

---

## 四、贝叶斯优化流程

### 4.1 优化目标
- 最小化验证损失
- 优化模型超参数和训练参数

### 4.2 优化参数
| 参数类别 | 参数 | 搜索范围 |
|---------|------|----------|
| 模型参数 | hidden_dim | [32, 128] |
| 模型参数 | num_blocks | [4, 12] |
| 模型参数 | num_stages | [2, 4] |
| 训练参数 | batch_size | [16, 64] |
| 训练参数 | learning_rate | [1e-5, 1e-3] |
| 训练参数 | weight_decay | [1e-7, 1e-5] |
| 损失权重 | weight_y | [0.5, 3.0] |
| 损失权重 | weight_x | [0.1, 2.0] |
| 损失权重 | weight_z | [0.01, 1.0] |

### 4.3 执行流程
1. 初始化GPyOpt优化器
2. 运行20-30轮优化迭代
3. 选择最佳参数组合
4. 使用最佳参数进行完整训练

---

## 五、关键要点

### 5.1 必须做的事情

✅ **使用新训练逻辑**：正向预测 + 反向回推的组合损失  
✅ **动态Z采样**：训练时每个epoch重新采样Z  
✅ **批量生成**：推理时生成500个候选解  
✅ **智能选优**：基于NMSE指标选择最佳解  
✅ **鲁棒归一化**：使用中位数和四分位距进行数据归一化  
✅ **全面验证**：对生成的X进行正向验证  
✅ **约束范围**：确保生成的X在有效物理范围内  

### 5.2 不要做的事情

❌ 不要使用旧的训练逻辑  
❌ 不要只采样一个Z就停止  
❌ 不要忽视Z的分布约束  
❌ 不要使用未标准化的数据  
❌ 不要生成超出范围的X而不处理  
❌ 不要忽视训练过程中的损失曲线  

---

## 六、故障排除

### Q1: 训练损失不稳定？
- **解决方案**：检查Z的MMD损失权重，适当调整weight_z
- **建议**：确保Z的分布约束强度合理

### Q2: 反向设计结果质量差？
- **解决方案**：增加Z采样数量，提高z_scale值
- **建议**：检查训练数据是否覆盖目标响应空间

### Q3: 验证损失不下降？
- **解决方案**：调整学习率和批量大小
- **建议**：检查是否存在过拟合，增加早停patience

### Q4: 生成的X超出物理范围？
- **解决方案**：在反向设计后对X进行裁剪
- **建议**：基于训练数据的最小值和最大值设置合理范围

---

## 七、性能评估

### 7.1 评估指标
- **正向预测**：NMSE（归一化均方误差）
- **反向设计**：相对误差、NMSE
- **多解质量**：解的多样性、最佳解质量

### 7.2 预期性能
| 指标 | 目标值 |
|------|--------|
| 验证损失 | < 0.015 |
| 验证集NMSE | < 0.01 |
| 逆向预测准确率 | > 0.97 |
| Top 1解相对误差 | < 0.03 |

---

*文档编号: 03*  
*创建日期: 2026-02-15*  
*主题: 工作流程*  
*版本: v1.0*