"""
训练脚本：使用X.csv和Y.csv数据训练RINN模型
借鉴rinn_main.py的方法，将X和Y映射到200维度
"""
import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
from datetime import datetime
import csv

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ============== 创建训练输出文件夹 ==============
# 每次训练创建独立的输出文件夹
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"rinne_training_{timestamp}"
checkpoint_dir = os.path.join('model_checkpoints_rinn', experiment_name)
os.makedirs(checkpoint_dir, exist_ok=True)
print(f'本次训练输出文件夹: {checkpoint_dir}')

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('使用设备:', device)

# ============== 数据加载与预处理 ==============
print('\n=== 加载数据 ===')

# 加载预处理的数据（从data文件夹）
x_samples = np.load('data/x_samples.npy', allow_pickle=True)
y_data = np.load('data/y_data.npy')

print('X样本数:', len(x_samples))
print('X形状:', x_samples.shape)
print('Y数据形状:', y_data.shape)

# 从X.csv直接读取样本（处理CSV格式）
def load_x_samples_from_csv(csv_path):
    """从X.csv读取样本，处理CSV格式"""
    samples = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                if cell.strip():
                    samples.append(cell.strip())
    return samples

x_samples = load_x_samples_from_csv('data/X.csv')
print('从CSV读取的X样本数:', len(x_samples))

# 解析X样本字符串为数值特征
def parse_x_sample(x_str):
    """将逗号分隔的字符串解析为数值特征"""
    values = x_str.split(',')
    float_values = []
    for v in values:
        v = v.strip()
        if v:
            float_values.append(float(v))
    return np.array(float_values, dtype=np.float32)

x_features = np.array([parse_x_sample(x) for x in x_samples])
print('X特征形状:', x_features.shape)
print('X特征示例 (第一个样本):', x_features[0])

# 数据标准化
x_mean = x_features.mean(axis=0)
x_std = x_features.std(axis=0)
x_features_normalized = (x_features - x_mean) / (x_std + 1e-8)

y_mean = y_data.mean(axis=0)
y_std = y_data.std(axis=0)
y_features_normalized = (y_data - y_mean) / (y_std + 1e-8)

# 将X零填充到200维（与Y相同维度）
x_padded = np.zeros((x_features_normalized.shape[0], 200), dtype=np.float32)
x_padded[:, :x_features_normalized.shape[1]] = x_features_normalized  # 前5维是原始X，其余195维为0

print('\n数据统计:')
print('  X特征均值:', x_features_normalized.mean(axis=0))
print('  X特征标准差:', x_features_normalized.std(axis=0))
print('  Y特征均值范围:', y_features_normalized.min(), '-', y_features_normalized.max())
print('  填充后X形状:', x_padded.shape)
print('  Y形状:', y_features_normalized.shape)

# 划分训练集和验证集 (80% 训练, 20% 验证)
n_samples = len(x_padded)
indices = np.random.permutation(n_samples)
train_size = int(0.8 * n_samples)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# 使用填充后的X（200维）
x_train = torch.FloatTensor(x_padded[train_indices])
y_train = torch.FloatTensor(y_features_normalized[train_indices])  # Y仍然是200维

# 创建200维标准正态分布的z_real（作为z_real的目标分布）
torch.manual_seed(42)
np.random.seed(42)
z_real_train = torch.randn(len(x_train), 200)  # 使用固定种子生成标准正态分布的z_real

x_val = torch.FloatTensor(x_padded[val_indices])
y_val = torch.FloatTensor(y_features_normalized[val_indices])

# 验证集使用不同的随机种子，确保与训练集不同
torch.manual_seed(42)
np.random.seed(42)
z_real_val = torch.randn(len(x_val), 200)

print('\n数据集划分:')
print('  训练集: %d 样本' % len(x_train))
print('  验证集: %d 样本' % len(x_val))
print('  X维度: %d, Y维度: %d, Z_real维度: %d' % (x_train.shape[1], y_train.shape[1], z_real_train.shape[1]))

# 创建DataLoader（包含x, y和z_real，保持与rinn_main.py一致）
batch_size = 8
train_dataset = TensorDataset(x_train, y_train, z_real_train)
val_dataset = TensorDataset(x_val, y_val, z_real_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ============== 模型定义 ==============
print('\n=== 模型定义 ===')

# 导入RINN模型组件
import sys
sys.path.append('d:/1事务/1论文/微波设计/RINN-dev/RINN-dev')
from R_INN_model.rinn_model import RINNModel
from R_INN_model.loss_methods import mmd_loss, nmse_loss

# 配置参数
x_dim = 200  # X现在填充为200维
y_dim = 200  # Y维度
z_dim = 200  # Z维度（标准正态分布的维度）
model_input_dim = x_dim  # 模型输入只有x（200维）

# 创建RINN模型（极简配置适应小样本高维问题）
model = RINNModel(
    input_dim=model_input_dim,
    hidden_dim=64,   # 最小隐藏维度
    num_blocks=8,   # 最少块数
    num_stages=4,   # 单阶段
    num_cycles_per_stage=1
).to(device)

print(f'模型输入维度: {model.input_dim}')
print(f'X维度: {x_dim}, Y维度: {y_dim}')

# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f'模型参数总数: {total_params}')

# ============== 训练配置 ==============
print('\n=== 训练配置 ===')

# 权重系数（调整Ly权重，因为真实数据Ly值很大）
w_x = 1.0
w_y = 1.0 # 大幅降低Ly权重
w_z = 1.0
clip_value = 0.5  # 加强梯度裁剪

# 优化器（降低学习率以稳定训练）
lr = 1e-3
weight_decay = 2e-5
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# 训练参数
num_epochs = 200  # 增加训练轮数
best_val_loss = float('inf')
patience = 50   # 增加早停耐心
patience_counter = 0

# 检查点目录已在上方创建

# 训练历史
train_losses = {'total': [], 'Lx': [], 'Ly': [], 'Lz': []}
val_losses = {'total': [], 'Lx': [], 'Ly': [], 'Lz': []}

# 保存训练配置信息
training_info = {
    'timestamp': timestamp,
    'model_config': {
        'input_dim': model_input_dim,
        'hidden_dim': 8,
        'num_blocks': 1,
        'num_stages': 1,
        'num_cycles_per_stage': 1
    },
    'training_params': {
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'clip_value': clip_value,
        'num_epochs': num_epochs,
        'loss_weights': {'w_x': w_x, 'w_y': w_y, 'w_z': w_z}
    },
    'data_info': {
        'train_samples': len(x_train),
        'val_samples': len(x_val),
        'x_dim': x_train.shape[1],
        'y_dim': y_train.shape[1],
        'z_dim': z_real_train.shape[1]
    }
}

# 保存训练配置
import json
with open(os.path.join(checkpoint_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
    json.dump(training_info, f, ensure_ascii=False, indent=2)

print('开始训练...')
start_time = datetime.now()

# ============== 损失计算 ==============
def calculate_total_loss(x_real, x_recon, y_real, y_pred, z_real, z_recon, log_det_total=None):
    """计算总损失"""
    # Ly: 计算真实y与预测y的正向预测误差
    #     y_pred是模型从x正向映射后得到的y预测
    Ly = nmse_loss(y_real, y_pred)
    
    # Lx: 比较真实x与重建x的分布差异
    #     x_recon是从y通过逆向映射重构得到的x
    Lx = mmd_loss(x_real, x_recon, log_det_total=log_det_total)
    
    # Lz: 比较重建z与标准高斯分布的差异
    #     z_recon是RealNVP层输出的隐空间表示（200维）
    #     z_real是200维标准正态分布，用于正则化z_recon
    Lz = mmd_loss(z_recon, z_real)
    
    # 总损失
    total_loss = w_x * Lx + w_y * Ly + w_z * Lz
    
    return {
        "total_loss": total_loss,
        "Lx": Lx,
        "Ly": Ly,
        "Lz": Lz
    }

# ============== 训练循环 ==============
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    # 训练阶段
    model.train()
    epoch_train_losses = {"total_loss": 0.0, "Lx": 0.0, "Ly": 0.0, "Lz": 0.0}
    
    for batch_x, batch_y, batch_z_real in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_z_real = batch_z_real.to(device)
        
        # 正向映射：模型输入只有x（200维）
        # 返回: y_forward（最终输出，用于y_loss）, log_det_forward, z_from_realnvp（RealNVP输出，用于z_loss）
        y_forward, log_det_forward, z_from_realnvp = model(batch_x, return_intermediate=True)
        
        # y_pred是y的预测（就是y_forward，因为y_forward已经是200维）
        y_pred = y_forward
        
        # 逆向映射：从y重构x
        x_recon, log_det_inverse = model.inverse(batch_y)
        
        # z_recon是RealNVP层输出的z（用于z_loss计算）
        z_recon = z_from_realnvp
        
        # 计算损失
        losses = calculate_total_loss(
            batch_x, x_recon, 
            batch_y, y_pred, 
            batch_z_real, z_recon, 
            log_det_total=log_det_forward
        )
        
        # 梯度更新
        optimizer.zero_grad()
        losses["total_loss"].backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()
        
        # 损失累加
        for key in epoch_train_losses:
            epoch_train_losses[key] += losses[key].item()
    
    # 计算平均损失
    num_batches = len(train_loader)
    for key in epoch_train_losses:
        epoch_train_losses[key] /= num_batches
    
    # 记录训练历史
    train_losses['total'].append(epoch_train_losses['total_loss'])
    train_losses['Lx'].append(epoch_train_losses['Lx'])
    train_losses['Ly'].append(epoch_train_losses['Ly'])
    train_losses['Lz'].append(epoch_train_losses['Lz'])
    
    # 打印训练日志
    epoch_train_time = time.time() - epoch_start_time
    
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {epoch_train_losses['total_loss']:.6f}, "
          f"Lx: {epoch_train_losses['Lx']:.6f}, "
          f"Ly: {epoch_train_losses['Ly']:.6f}, "
          f"Lz: {epoch_train_losses['Lz']:.6f}, "
          f"Train Time: {epoch_train_time:.2f}s")
    
    # 验证阶段
    val_start_time = time.time()
    
    model.eval()
    epoch_val_losses = {"total_loss": 0.0, "Lx": 0.0, "Ly": 0.0, "Lz": 0.0}
    
    with torch.no_grad():
        for batch_x, batch_y, batch_z_real in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_z_real = batch_z_real.to(device)
            
            # 正向映射：模型输入只有x（200维）
            # 返回: y_forward（最终输出，用于y_loss）, log_det_forward, z_from_realnvp（RealNVP输出，用于z_loss）
            y_forward, log_det_forward, z_from_realnvp = model(batch_x, return_intermediate=True)
            
            # y_pred是y的预测（就是y_forward）
            y_pred = y_forward
            
            # 逆向映射：从y重构x
            x_recon, log_det_inverse = model.inverse(batch_y)
            
            # z_recon是RealNVP层输出的z（用于z_loss计算）
            z_recon = z_from_realnvp
            
            # 计算损失
            losses = calculate_total_loss(
                batch_x, x_recon, 
                batch_y, y_pred, 
                batch_z_real, z_recon, 
                log_det_total=log_det_forward
            )
            
            for key in epoch_val_losses:
                epoch_val_losses[key] += losses[key].item()
    
    # 计算验证集平均损失
    for key in epoch_val_losses:
        epoch_val_losses[key] /= len(val_loader)
    
    # 计算验证时间
    val_time = time.time() - val_start_time
    
    # 打印验证日志
    print(f"          Val Loss: {epoch_val_losses['total_loss']:.6f}, "
          f"Lx: {epoch_val_losses['Lx']:.6f}, "
          f"Ly: {epoch_val_losses['Ly']:.6f}, "
          f"Lz: {epoch_val_losses['Lz']:.6f}, "
          f"Val Time: {val_time:.2f}s")
    
    # 记录验证历史
    val_losses['total'].append(epoch_val_losses['total_loss'])
    val_losses['Lx'].append(epoch_val_losses['Lx'])
    val_losses['Ly'].append(epoch_val_losses['Ly'])
    val_losses['Lz'].append(epoch_val_losses['Lz'])
    
    # 保存最佳模型
    if epoch_val_losses['total_loss'] < best_val_loss:
        best_val_loss = epoch_val_losses['total_loss']
        patience_counter = 0
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_losses['total_loss'],
            'val_loss': epoch_val_losses['total_loss'],
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'x_dim': x_dim,
            'z_dim': z_dim,
            'y_dim': y_dim
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
        print(f'  -> 保存最佳模型 (Val Loss: {best_val_loss:.6f})')
    else:
        patience_counter += 1
    
    # 早停
    if patience_counter >= patience:
        print(f'\n早停触发! 验证损失连续{patience}个epoch没有改善')
        break

total_time = datetime.now() - start_time
print(f'\n训练完成! 总时间: {total_time}')
print(f'最佳验证损失: {best_val_loss:.6f}')

# ============== 可视化 ==============
print('\n=== 生成训练曲线 ===')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 总损失曲线
axes[0, 0].plot(train_losses['total'], label='Train Total', color='blue')
axes[0, 0].plot(val_losses['total'], label='Val Total', color='red')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Total Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Lx损失曲线
axes[0, 1].plot(train_losses['Lx'], label='Train Lx', color='blue')
axes[0, 1].plot(val_losses['Lx'], label='Val Lx', color='red')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Lx Loss (X Reconstruction)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Ly损失曲线
axes[1, 0].plot(train_losses['Ly'], label='Train Ly', color='blue')
axes[1, 0].plot(val_losses['Ly'], label='Val Ly', color='red')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Ly Loss (Y Prediction)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Lz损失曲线
axes[1, 1].plot(train_losses['Lz'], label='Train Lz', color='blue')
axes[1, 1].plot(val_losses['Lz'], label='Val Lz', color='red')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Lz Loss (Z Reconstruction)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, 'training_losses.png'), dpi=150, bbox_inches='tight')
plt.close()
print('训练曲线已保存到:', os.path.join(checkpoint_dir, 'training_losses.png'))

# 保存训练历史
np.save(os.path.join(checkpoint_dir, 'train_losses.npy'), np.array(train_losses['total']))
np.save(os.path.join(checkpoint_dir, 'val_losses.npy'), np.array(val_losses['total']))
print('训练历史已保存')

# ============== 模型评估 ==============
print('\n=== 模型评估 ===')

model.eval()
with torch.no_grad():
    # 生成更多测试样本的预测
    test_samples = min(10, len(x_val))  # 最多测试10个样本
    x_test = x_val[:test_samples].to(device)
    y_pred = model(x_test)[0].cpu().numpy()
    
    # 反标准化
    y_pred_original = y_pred * y_std + y_mean
    y_true_original = y_val[:test_samples, :200].numpy() * y_std + y_mean
    
    # 保存预测结果数据
    np.save(os.path.join(checkpoint_dir, 'y_pred_test.npy'), y_pred_original)
    np.save(os.path.join(checkpoint_dir, 'y_true_test.npy'), y_true_original)
    np.save(os.path.join(checkpoint_dir, 'x_test.npy'), x_test.cpu().numpy())
    
    # 计算评估指标
    mse = np.mean((y_true_original - y_pred_original) ** 2)
    mae = np.mean(np.abs(y_true_original - y_pred_original))
    mape = np.mean(np.abs((y_true_original - y_pred_original) / (np.abs(y_true_original) + 1e-8))) * 100
    
    # 保存评估报告
    eval_report = {
        'test_samples': test_samples,
        'mse': float(mse),
        'mae': float(mae),
        'mape': float(mape),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(checkpoint_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)
    
    # 绘制预测结果
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i in range(test_samples):
        axes[i].plot(y_true_original[i], label='True', color='blue', alpha=0.7)
        axes[i].plot(y_pred_original[i], label='Predicted', color='red', linestyle='--', alpha=0.7)
        axes[i].set_xlabel('Frequency Point')
        axes[i].set_ylabel('Value')
        axes[i].set_title(f'Test Sample {i+1}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(test_samples, len(axes)):
        axes[i].set_visible(False)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('RINN Model Predictions vs True Values')
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('预测结果图已保存到:', os.path.join(checkpoint_dir, 'predictions.png'))

print('\n=== 训练完成! ===')
print(f'模型检查点保存在: {checkpoint_dir}')






