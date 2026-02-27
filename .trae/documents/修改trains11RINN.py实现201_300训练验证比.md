## 分析当前数据加载逻辑

1. **当前训练集**：使用 `data/S Parameter Plot200.csv` 和 `data/S Parameter Plot300.csv`，合并后作为训练集
2. **当前验证集**：使用 `data/S Parameter Plot1perfect.csv`、`data/S Parameter Plot200.csv` 和 `data/S Parameter Plot300.csv`，合并后取前5个样本

## 修改计划

### 1. 调整数据加载逻辑
- **训练集**：从 `Plot200.csv` 和 `Plot300.csv` 中总共选取201个样本
- **验证集**：
  - 首先包含 `Plot1perfect.csv` 中的所有样本
  - 然后从 `Plot200.csv` 和 `Plot300.csv` 中补充足够的样本，使总样本数达到300
  - 确保验证集包含perfect数据

### 2. 具体修改点

#### 训练集加载修改
- 修改 `train_files` 加载逻辑
- 合并后随机选取201个样本

#### 验证集加载修改
- 修改 `val_files` 加载逻辑，优先加载 `Plot1perfect.csv`
- 合并后确保包含perfect数据，并补充至300个样本

### 3. 保持其他功能不变
- 保持模型结构、训练逻辑、损失计算等其他部分不变
- 确保所有可视化和评估功能正常工作

### 4. 验证修改效果
- 运行修改后的脚本
- 检查训练集和验证集的样本数是否正确
- 验证模型性能是否正常

## 预期结果
- 训练集：201个样本
- 验证集：300个样本（包含perfect数据）
- 模型能够正常训练并评估性能