"""
基于RINN模型的密度估计实现
正确的数据结构和损失计算：
1. 左侧输入：X(5维) + 零填充(202维) → 总207维
2. 右侧输入：Y(202维) + Z(5维) → 总207维，其中Z是随机生成的标准高斯分布
3. 损失计算：
   - 正向预测的Y'和真实Y的NMSE损失
   - 重建X的损失
   - 正向预测的Z'和标准高斯分布的MMD差异
"""
import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json

# 导入R_INN_model包中的工具函数
from R_INN_model.data_utils import extract_geometry_params, load_data_from_csv, normalize_data
from R_INN_model.training_utils import calculate_loss, train_model
from R_INN_model.prediction_utils import predict_y, backward_predict_x, generate_multiple_solutions
from R_INN_model.visualization_utils import plot_training_curves, plot_predicted_y, plot_backward_prediction, plot_multi_solution_distribution

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ============== 参数解析 ==============
def parse_args():
    parser = argparse.ArgumentParser(description='RINN模型训练脚本')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    return parser.parse_args()

args = parse_args()

# 加载配置 - 使用贝叶斯优化找到的最佳参数
config = {
    "model_config": {
        "hidden_dim": 104,  # 最佳参数
        "num_blocks": 9,   # 最佳参数
        "num_stages": 3,   # 最佳参数
        "num_cycles_per_stage": 2,  # 最佳参数
        "ratio_toZ_after_flowstage": 0.6540162818070427,  # 最佳参数
        "ratio_x1_x2_inAffine": 0.5816437692469725  # 最佳参数
    },
    "training_params": {
        "batch_size": 32,  # 最佳参数
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0005452008335396878,  # 最佳参数
        "weight_decay": 1.7130731509924992e-06,  # 最佳参数
        "clip_value": 0.5,
        "num_epochs": 200,  # 完整训练200 epochs
        "loss_weights": {
            "weight_y": 0.21745801953492716,  # 最佳参数
            "weight_x": 0.17270257896844063,  # 最佳参数
            "weight_z": 0.2534696848821336   # 最佳参数
        }
    },
    "data_params": {
        "normalization_method": "robust"
    }
}

# 如果提供了配置文件，加载配置
if args.config and os.path.exists(args.config):
    print(f'从配置文件加载参数: {args.config}')
    with open(args.config, 'r', encoding='utf-8') as f:
        loaded_config = json.load(f)
    # 更新配置
    if 'model_config' in loaded_config:
        config['model_config'].update(loaded_config['model_config'])
    if 'training_params' in loaded_config:
        config['training_params'].update(loaded_config['training_params'])
    if 'data_params' in loaded_config:
        config['data_params'].update(loaded_config['data_params'])
    print('配置加载完成!')

# ============== 创建训练输出文件夹 ==============
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"rinn_correct_structure_{timestamp}"
checkpoint_dir = os.path.join('model_checkpoints_rinn', experiment_name)
os.makedirs(checkpoint_dir, exist_ok=True)
print(f'本次训练输出文件夹: {checkpoint_dir}')

# 保存使用的配置
with open(os.path.join(checkpoint_dir, 'used_config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设备配置
from R_INN_model.device_utils import get_device
device = get_device()

# ============== 数据加载与预处理 ==============
print('\n=== 加载数据 ===')



# 加载训练集数据
print('\n=== 加载训练集数据 ===')
train_files = ['data/S Parameter Plot200.csv', 'data/S Parameter Plot300.csv']
train_x = []
train_y = []

for file_path in train_files:
    x, y, freq_data = load_data_from_csv(file_path)
    train_x.append(x)
    train_y.append(y)

# 合并训练集数据
train_x = np.vstack(train_x)
train_y = np.vstack(train_y)
print(f'训练集总样本数: {len(train_x)}')

# 加载验证集数据
print('\n=== 加载验证集数据 ===')
# 使用5个样本作为验证集，确保包含perfect.csv
val_files = ['data/S Parameter Plot1perfect.csv', 'data/S Parameter Plot200.csv', 'data/S Parameter Plot300.csv']
val_x = []
val_y = []

for file_path in val_files:
    x, y, freq_data = load_data_from_csv(file_path)
    val_x.append(x)
    val_y.append(y)

# 合并验证集数据
val_x = np.vstack(val_x)
val_y = np.vstack(val_y)

# 确保验证集包含5个样本
if len(val_x) > 5:
    # 确保包含perfect.csv中的样本
    val_x = val_x[:5]
    val_y = val_y[:5]
print(f'验证集样本数: {len(val_x)}')

print(f'\n训练集样本数: {len(train_x)}')
print(f'验证集样本数: {len(val_x)}')

# 数据标准化
normalization_method = config['data_params']['normalization_method']  # 'standard' 或 'robust'
print(f'\n数据归一化方法: {normalization_method}')

# 使用data_utils中的normalize_data函数进行数据标准化
train_x_normalized, val_x_normalized, train_y_normalized, val_y_normalized, x_mean, x_std, y_mean, y_std = normalize_data(
    train_x, val_x, train_y, val_y, method=normalization_method
)

# 数据质量检查
print(f'\n归一化方法: {normalization_method}')
print(f'X训练集归一化后均值: {train_x_normalized.mean(axis=0).mean():.6f}, 标准差: {train_x_normalized.std(axis=0).mean():.6f}')
print(f'X验证集归一化后均值: {val_x_normalized.mean(axis=0).mean():.6f}, 标准差: {val_x_normalized.std(axis=0).mean():.6f}')
print(f'Y训练集归一化后均值: {train_y_normalized.mean(axis=0).mean():.6f}, 标准差: {train_y_normalized.std(axis=0).mean():.6f}')
print(f'Y验证集归一化后均值: {val_y_normalized.mean(axis=0).mean():.6f}, 标准差: {val_y_normalized.std(axis=0).mean():.6f}')

# 检查是否存在NaN或无穷大值
print(f'X训练集是否包含NaN: {np.isnan(train_x_normalized).any()}')
print(f'X训练集是否包含无穷大: {np.isinf(train_x_normalized).any()}')
print(f'X验证集是否包含NaN: {np.isnan(val_x_normalized).any()}')
print(f'X验证集是否包含无穷大: {np.isinf(val_x_normalized).any()}')
print(f'Y训练集是否包含NaN: {np.isnan(train_y_normalized).any()}')
print(f'Y训练集是否包含无穷大: {np.isinf(train_y_normalized).any()}')
print(f'Y验证集是否包含NaN: {np.isnan(val_y_normalized).any()}')
print(f'Y验证集是否包含无穷大: {np.isinf(val_y_normalized).any()}')

# 检查归一化后的数据范围
print(f'X训练集归一化后最小值: {train_x_normalized.min():.6f}, 最大值: {train_x_normalized.max():.6f}')
print(f'X验证集归一化后最小值: {val_x_normalized.min():.6f}, 最大值: {val_x_normalized.max():.6f}')
print(f'Y训练集归一化后最小值: {train_y_normalized.min():.6f}, 最大值: {train_y_normalized.max():.6f}')
print(f'Y验证集归一化后最小值: {val_y_normalized.min():.6f}, 最大值: {val_y_normalized.max():.6f}')

# ============== 维度处理：正确的数据结构 ==============
print('\n=== 维度处理与数据结构 ===')

# 配置参数
x_dim = train_x.shape[1]  # X维度：5
y_dim = train_y.shape[1]  # Y维度：101（dB值）
z_dim = x_dim             # Z维度：5（与X维度相同）

# 左侧输入：X + 零填充 → 总维度 = x_dim + padding_dim = 5 + 101 = 106
padding_dim = y_dim
left_input_dim = x_dim + padding_dim

# 右侧输入：Y + Z → 总维度 = y_dim + z_dim = 101 + 5 = 106
right_input_dim = y_dim + z_dim

print(f'X维度: {x_dim}, Y维度: {y_dim}, Z维度: {z_dim}')
print(f'左侧输入维度: {left_input_dim} (X: {x_dim} + 零填充: {padding_dim})')
print(f'右侧输入维度: {right_input_dim} (Y: {y_dim} + Z: {z_dim})')
print(f'总输入/输出维度: {left_input_dim} (左右侧维度相同)')

# 配置affine coupling比率，根据X和Y的维度调整
if config['model_config']['ratio_x1_x2_inAffine'] is None:
    ratio_x1_x2_inAffine = x_dim / left_input_dim  # X部分作为条件输入的比例
else:
    ratio_x1_x2_inAffine = config['model_config']['ratio_x1_x2_inAffine']
ratio_toZ_after_flowstage = config['model_config']['ratio_toZ_after_flowstage']

print(f'Affine coupling比率: {ratio_x1_x2_inAffine}')

# 创建训练集数据
# 左侧输入：X + 零填充
left_train_input = np.concatenate((train_x_normalized, np.zeros((len(train_x_normalized), padding_dim), dtype=np.float32)), axis=1)
left_val_input = np.concatenate((val_x_normalized, np.zeros((len(val_x_normalized), padding_dim), dtype=np.float32)), axis=1)

# 注意：Z现在在每个epoch重新采样，不再在这里固定生成
# 右侧输入将在训练循环中动态生成

# 转换为torch张量（左侧输入固定，右侧输入将在每个epoch动态生成）
left_train = torch.FloatTensor(left_train_input)
left_val = torch.FloatTensor(left_val_input)

print('\n数据集划分:')
print(f'  训练集: {len(left_train)} 样本')
print(f'  验证集: {len(left_val)} 样本')
print('  注意：Z将在每个epoch重新采样，增强模型泛化能力')

# DataLoader将在训练循环中动态创建
batch_size = config['training_params']['batch_size']

# ============== 模型定义 ==============
print('\n=== 模型定义 ===')

# 导入RINN模型组件
from R_INN_model.rinn_model import RINNModel
from R_INN_model.loss_methods import mmd_loss, nmse_loss, weighted_nmse_loss

# 创建可逆模型（密度估计任务）
model = RINNModel(
    input_dim=left_input_dim,  # 模型输入/输出维度：左右侧维度相同
    hidden_dim=config['model_config']['hidden_dim'],  # 从配置中获取hidden_dim，提高模型拟合能力
    num_blocks=config['model_config']['num_blocks'],   # 从配置中获取num_blocks，增强模型表达能力
    num_stages=config['model_config']['num_stages'],   # 从配置中获取num_stages，控制模型深度
    num_cycles_per_stage=config['model_config']['num_cycles_per_stage'],  # 从配置中获取num_cycles_per_stage
    ratio_toZ_after_flowstage=ratio_toZ_after_flowstage,
    ratio_x1_x2_inAffine=ratio_x1_x2_inAffine  # 使用适合X的affine coupling比率
).to(device)

print(f'模型输入/输出维度: {model.input_dim}')
print(f'左侧输入: X({x_dim}) + 零填充({padding_dim})')
print(f'右侧输入: Y({y_dim}) + Z({z_dim})')

# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f'模型参数总数: {total_params}')

# ============== 训练配置 ==============
print('\n=== 训练配置 ===')

# 调试标志：启用训练模式
skip_training = False  # 设置为True跳过训练，False执行完整训练

# 损失权重
weight_y = config['training_params']['loss_weights']['weight_y']  # 从配置中获取weight_y，提高正向预测精度
weight_x = config['training_params']['loss_weights']['weight_x']  # 从配置中获取weight_x，保持X重建损失权重不变
weight_z = config['training_params']['loss_weights']['weight_z']  # 从配置中获取weight_z，平衡生成多样性和稳定性
clip_value = config['training_params']['clip_value']  # 从配置中获取clip_value，防止梯度爆炸

# 优化器
lr = config['training_params']['learning_rate']  # 从配置中获取学习率，使用更精细的学习率调整
weight_decay = config['training_params']['weight_decay']  # 从配置中获取权重衰减，平衡正则化
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # 使用AdamW优化器

# 使用学习率衰减策略
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6)  # 调整patience，平衡收敛速度和稳定性

# 训练参数
num_epochs = config['training_params']['num_epochs']  # 从配置中获取训练轮数，让模型有更多时间拟合
best_val_loss = float('inf')
best_epoch = 0  # 初始化最佳epoch为0
patience = 60  # 增加早停耐心，避免过早停止
patience_counter = 0

# 梯度累积步数
grad_accum_steps = config['training_params']['gradient_accumulation_steps']  # 从配置中获取梯度累积步数

# 训练历史
train_losses = {'total': [], 'y_loss': [], 'x_loss': [], 'z_loss': []}
val_losses = {'total': [], 'y_loss': [], 'x_loss': [], 'z_loss': []}

# 保存训练配置信息
training_info = {
    'timestamp': timestamp,
    'model_config': {
        'input_dim': model.input_dim,
        'hidden_dim': config['model_config']['hidden_dim'],
        'num_blocks': config['model_config']['num_blocks'],
        'num_stages': config['model_config']['num_stages'],
        'num_cycles_per_stage': config['model_config']['num_cycles_per_stage'],
        'ratio_toZ_after_flowstage': ratio_toZ_after_flowstage,
        'ratio_x1_x2_inAffine': ratio_x1_x2_inAffine
    },
    'training_params': {
        'batch_size': batch_size,
        'gradient_accumulation_steps': grad_accum_steps,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'clip_value': clip_value,
        'num_epochs': num_epochs,
        'loss_weights': {
            'weight_y': weight_y,
            'weight_x': weight_x,
            'weight_z': weight_z
        }
    },
    'data_info': {
        'train_samples': len(left_train),
        'val_samples': len(left_val),
        'x_dim': x_dim,
        'y_dim': y_dim,
        'z_dim': z_dim,
        'left_input_dim': left_input_dim,
        'right_input_dim': right_input_dim,
        'normalization_method': normalization_method,
        'x_mean': x_mean.tolist(),
        'x_std': x_std.tolist(),
        'y_mean': y_mean.tolist(),
        'y_std': y_std.tolist()
    }
}

import json
with open(os.path.join(checkpoint_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
    json.dump(training_info, f, ensure_ascii=False, indent=2)



# ============== 训练循环 ==============
if not skip_training:
    print('开始训练...')
    start_time = datetime.now()

    # 使用training_utils中的train_model函数进行训练
    best_val_loss, train_losses, val_losses = train_model(
        model=model,
        left_train=left_train,
        left_val=left_val,
        train_y_normalized=train_y_normalized,
        val_y_normalized=val_y_normalized,
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
        weight_y=weight_y,
        weight_x=weight_x,
        weight_z=weight_z,
        device=device,
        patience=patience,
        grad_accum_steps=grad_accum_steps,
        clip_value=clip_value
    )

    total_time = datetime.now() - start_time
    print(f'\n训练完成! 总时间: {total_time}')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    
    # 保存最佳验证损失到文件
    best_val_loss_file = os.path.join(checkpoint_dir, 'best_val_loss.txt')
    with open(best_val_loss_file, 'w') as f:
        f.write(f'Best Validation Loss: {best_val_loss:.6f}\n')
        f.write(f'Training Time: {total_time}\n')
    print(f'最佳验证损失已保存到: {best_val_loss_file}')
    
    # 计算验证集上的NMSE
    print('\n=== 计算验证集NMSE ===')
    model.eval()
    total_val_nmse = 0.0
    
    # 创建验证集数据加载器
    val_z = np.random.randn(len(val_y_normalized), z_dim).astype(np.float32)
    right_val_input = np.concatenate((val_y_normalized, val_z), axis=1)
    right_val = torch.FloatTensor(right_val_input)
    val_dataset = TensorDataset(left_val, right_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in val_loader:
            left_batch = batch[0].to(device)
            right_batch = batch[1].to(device)
            
            # 正向映射：left_input → predicted_right
            predicted_right, _, _ = model(left_batch, return_intermediate=True)
            
            # 从predicted_right中提取Y'
            predicted_y = predicted_right[:, :y_dim]
            
            # 从right_input中提取真实Y
            real_y = right_batch[:, :y_dim]
            
            # 计算NMSE
            batch_nmse = nmse_loss(real_y, predicted_y)
            total_val_nmse += batch_nmse.item()

    avg_val_nmse = total_val_nmse / len(val_loader)
    print(f'验证集平均NMSE: {avg_val_nmse:.6f}')

    # 将验证集平均NMSE保存到文件
    best_val_loss_file = os.path.join(checkpoint_dir, 'best_val_loss.txt')
    with open(best_val_loss_file, 'a') as f:
        f.write(f'Validation Set Average NMSE: {avg_val_nmse:.6f}\n')
    print(f'验证集平均NMSE已追加保存到: {best_val_loss_file}')
else:
    print('跳过训练阶段 (skip_training=True)')
    # 生成虚拟损失数据以避免可视化错误
    train_losses['total'] = [1.0, 0.8, 0.6, 0.4, 0.2] * 20
    train_losses['y_loss'] = [0.8, 0.6, 0.4, 0.3, 0.2] * 20
    train_losses['x_loss'] = [0.5, 0.4, 0.3, 0.2, 0.1] * 20
    train_losses['z_loss'] = [0.3, 0.2, 0.2, 0.1, 0.1] * 20
    val_losses['total'] = [1.2, 0.9, 0.7, 0.5, 0.3] * 20
    val_losses['y_loss'] = [0.9, 0.7, 0.5, 0.4, 0.3] * 20
    val_losses['x_loss'] = [0.6, 0.5, 0.4, 0.3, 0.2] * 20
    val_losses['z_loss'] = [0.4, 0.3, 0.3, 0.2, 0.2] * 20
    best_val_loss = 0.2
    avg_val_nmse = 0.2  # 虚拟值

# ============== 可视化训练曲线 ==============
print('\n=== Generating training curves ===')

# 使用visualization_utils中的plot_training_curves函数
plot_training_curves(
    train_losses=train_losses,
    val_losses=val_losses,
    save_path=os.path.join(checkpoint_dir, 'training_losses.png')
)
print('Training curves saved to:', os.path.join(checkpoint_dir, 'training_losses.png'))

# ============== 模型功能实现：固定x预测y ==============
print('\n=== Fixed x predicting y functionality ===')

# 从验证集中选取测试样本，确保索引不超过验证集长度
val_size = len(val_x)
test_indices = [i * val_size // 5 for i in range(5)]  # 均匀选取5个测试样本
print(f'验证集大小: {val_size}, 测试样本索引: {test_indices}')

for i, test_idx in enumerate(test_indices):
    print(f'\nPredicting y for test sample {i+1}:')
    
    x_test = val_x_normalized[test_idx:test_idx+1]  # 形状：(1, x_dim)

    # 获取真实的y值
    real_y = val_y[test_idx:test_idx+1]

    # 使用prediction_utils中的predict_y函数
    predicted_y, nmse_value = predict_y(
        model=model,
        x_test=x_test,
        real_y=real_y,
        padding_dim=padding_dim,
        y_mean=y_mean,
        y_std=y_std,
        device=device
    )
    
    # 确保real_y维度正确
    real_y = real_y[:, :101]  # 只保留101维dB值

    print(f'  Test sample {i+1} prediction result:')
    print(f'    Predicted y shape: {predicted_y.shape}')
    print(f'    NMSE: {nmse_value:.6f}')

    # 使用visualization_utils中的plot_predicted_y函数
    plot_predicted_y(
        freq_data=freq_data,
        real_y=real_y[0],
        predicted_y=predicted_y[0],
        sample_idx=i+1,
        save_path=os.path.join(checkpoint_dir, f'fixed_x_predicted_y_{i+1}.png')
    )
    print(f'  Plot saved for sample {i+1}: fixed_x_predicted_y_{i+1}.png')

# ============== 模型功能实现：固定y回推x（多Z采样+NMSE选优） ==============
print('\n=== Fixed y backward predicting x with multi-Z sampling (500 candidates) ===')

# 从验证集中选取测试样本，使用全部验证集样本
y_test_indices = list(range(val_size))  # 使用全部验证集样本
num_z_candidates = 500  # 每个样本采样500个Z进行选优
print(f'验证集大小: {val_size}, 每个样本采样 {num_z_candidates} 个Z进行选优')

# 存储逆向预测的结果，用于计算正确率
backward_results = []
all_relative_errors = []
all_best_nmse = []  # 存储每个样本的最佳NMSE

print('\n开始多Z采样回推（使用NMSE进行选优）...')

for i, y_test_idx in enumerate(y_test_indices):
    if i % 10 == 0:  # 每10个样本打印一次进度
        print(f'\nProcessing sample {i+1}/{val_size}:')
    
    # 从验证集中选取一个测试样本
    y_test = val_y_normalized[y_test_idx:y_test_idx+1]  # 形状：(1, y_dim)
    real_y = val_y[y_test_idx:y_test_idx+1]  # 真实Y值（未归一化）
    real_y = real_y[:, :y_dim]  # 确保维度正确
    
    # 获取真实的x值
    real_x = val_x[y_test_idx:y_test_idx+1]

    # 使用prediction_utils中的backward_predict_x函数
    best_reconstructed_x, best_nmse, relative_errors = backward_predict_x(
        model=model,
        y_test=y_test,
        real_y=real_y,
        real_x=real_x,
        z_dim=z_dim,
        x_dim=x_dim,
        y_dim=y_dim,
        padding_dim=padding_dim,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        num_z_candidates=num_z_candidates,
        device=device
    )
    
    if i % 10 == 0:  # 每10个样本打印一次结果
        print(f'  Best NMSE: {best_nmse:.6f} (from {num_z_candidates} candidates)')
        print(f'  Real x: {real_x[0]}')
        print(f'  Best backward x: {best_reconstructed_x}')
    
    all_relative_errors.append(relative_errors)
    all_best_nmse.append(best_nmse)
    
    if i % 10 == 0:  # 每10个样本打印一次误差
        print(f'    Relative errors: {relative_errors}')
        print(f'    Avg relative error: {np.mean(relative_errors):.6f}')
    
    # 存储结果
    backward_results.append({
        'real_x': real_x[0].tolist(),
        'predicted_x': best_reconstructed_x.tolist(),
        'relative_errors': relative_errors.tolist(),
        'best_nmse': float(best_nmse),
        'best_z_idx': 0  # 由于使用了工具函数，这里设为0
    })

# 计算所有样本的统计指标
all_relative_errors = np.array(all_relative_errors)
avg_relative_error = np.mean(all_relative_errors)
accuracy = 1.0 - avg_relative_error
avg_best_nmse = np.mean(all_best_nmse)

print(f'\n=== 多Z采样回推结果统计（{num_z_candidates} candidates per sample）===')
print(f'验证集样本数: {val_size}')
print(f'平均最佳NMSE: {avg_best_nmse:.6f}')
print(f'平均相对误差: {avg_relative_error:.6f}')
print(f'回推准确率: {accuracy:.6f}')
print(f'NMSE统计 - 最小: {np.min(all_best_nmse):.6f}, 最大: {np.max(all_best_nmse):.6f}, 标准差: {np.std(all_best_nmse):.6f}')

# 保存结果到文件
best_val_loss_file = os.path.join(checkpoint_dir, 'best_val_loss.txt')
with open(best_val_loss_file, 'a') as f:
    f.write(f'\n=== Multi-Z Backward Prediction Results ({num_z_candidates} candidates) ===\n')
    f.write(f'Average Best NMSE: {avg_best_nmse:.6f}\n')
    f.write(f'Backward Prediction Avg Relative Error: {avg_relative_error:.6f}\n')
    f.write(f'Backward Prediction Accuracy: {accuracy:.6f}\n')
print(f'多Z采样回推结果已保存到: {best_val_loss_file}')

# 保存逆向预测的详细结果
backward_prediction_results = {
    'total_samples': val_size,
    'num_z_candidates': num_z_candidates,
    'average_relative_error': float(avg_relative_error),
    'accuracy': float(accuracy),
    'average_best_nmse': float(avg_best_nmse),
    'nmse_statistics': {
        'min': float(np.min(all_best_nmse)),
        'max': float(np.max(all_best_nmse)),
        'mean': float(np.mean(all_best_nmse)),
        'std': float(np.std(all_best_nmse)),
        'median': float(np.median(all_best_nmse))
    },
    'detailed_results': backward_results
}

# 保存结果到文件
results_file = os.path.join(checkpoint_dir, 'backward_prediction_results.json')
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(backward_prediction_results, f, ensure_ascii=False, indent=2)

print(f'\n多Z采样回推详细结果已保存到: {results_file}')
print(f'每个样本使用了 {num_z_candidates} 个Z候选进行选优')

# 可视化前5个样本的结果作为示例
print('\n=== 可视化前5个样本的逆向预测结果 ===')
for i, y_test_idx in enumerate(y_test_indices[:5]):
    print(f'\nVisualizing backward prediction for sample {i+1}:')
    
    # 从验证集中选取一个测试样本
    y_test = val_y_normalized[y_test_idx:y_test_idx+1]  # 形状：(1, y_dim)

    # 使用visualization_utils中的plot_backward_prediction函数
    plot_backward_prediction(
        model=model,
        y_test=y_test,
        real_x=val_x[y_test_idx:y_test_idx+1],
        real_y=val_y[y_test_idx:y_test_idx+1],
        z_dim=z_dim,
        x_dim=x_dim,
        y_dim=y_dim,
        padding_dim=padding_dim,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        freq_data=freq_data,
        sample_idx=i+1,
        save_path=os.path.join(checkpoint_dir, f'fixed_y_backward_x_{i+1}.png'),
        device=device
    )
    print(f'  Plot saved for sample {i+1}: fixed_y_backward_x_{i+1}.png')

# ============== 多解生成功能测试（500个Z + NMSE选优） ==============
print('\n=== Multiple solutions generation with 500 candidates (NMSE-based selection) ===')

# 选择一个特定的测试样本进行多解生成
multi_solution_idx = 0
y_test = val_y_normalized[multi_solution_idx:multi_solution_idx+1]  # 形状：(1, y_dim)
real_x = val_x[multi_solution_idx:multi_solution_idx+1]  # 真实的x值

# 使用prediction_utils中的generate_multiple_solutions函数
generated_xs, predicted_ys, nmse_errors, top_indices = generate_multiple_solutions(
    model=model,
    y_test=y_test,
    z_dim=z_dim,
    x_dim=x_dim,
    y_dim=y_dim,
    padding_dim=padding_dim,
    x_mean=x_mean,
    x_std=x_std,
    y_mean=y_mean,
    y_std=y_std,
    num_samples=500,
    z_scale=1.2,
    device=device
)

# 对生成的X进行后处理，确保在合理物理范围内
all_x = np.concatenate((train_x, val_x), axis=0)
x_min = all_x.min(axis=0)
x_max = all_x.max(axis=0)
generated_xs_clipped = np.clip(generated_xs, x_min, x_max)

# 验证X的多样性
diversity = np.std(generated_xs_clipped, axis=0)
print(f'\nX diversity (standard deviation for each parameter):')
for j, param_name in enumerate(['H1', 'H2', 'H3', 'H_C1', 'H_C2']):
    print(f'  {param_name}: {diversity[j]:.4f}')
print(f'  Average diversity: {np.mean(diversity):.4f}')

# 排序NMSE，获取统计信息
nmse_errors = np.array(nmse_errors)
print('\nNMSE Statistics:')
print(f'  Min: {np.min(nmse_errors):.6f}')
print(f'  Max: {np.max(nmse_errors):.6f}')
print(f'  Mean: {np.mean(nmse_errors):.6f}')
print(f'  Std: {np.std(nmse_errors):.6f}')
print(f'  Median: {np.median(nmse_errors):.6f}')

# 获取前5个最小NMSE的解
top_indices = np.argsort(nmse_errors)[:5]
print(f'\nTop 5 solutions with smallest NMSE:')
for rank, idx in enumerate(top_indices, 1):
    print(f'  Rank {rank}: Solution {idx + 1}, NMSE = {nmse_errors[idx]:.6f}')
    print(f'    Predicted X: {generated_xs_clipped[idx]}')
    print(f'    Relative errors: {np.abs((generated_xs_clipped[idx] - real_x[0]) / (real_x[0] + 1e-8))}')

# 使用visualization_utils中的plot_multi_solution_distribution函数
plot_multi_solution_distribution(
    generated_xs_clipped=generated_xs_clipped,
    real_x=real_x[0],
    nmse_errors=nmse_errors,
    predicted_ys=predicted_ys,
    y_test_original=val_y[multi_solution_idx:multi_solution_idx+1][0],
    freq_data=freq_data,
    save_path=os.path.join(checkpoint_dir, 'multi_solution_analysis.png')
)
print(f'\nMulti-solution analysis saved: multi_solution_analysis.png')

# 保存多解生成结果
np.save(os.path.join(checkpoint_dir, 'generated_xs.npy'), generated_xs)
np.save(os.path.join(checkpoint_dir, 'predicted_ys.npy'), predicted_ys)

# 保存多解生成结果（包含NMSE信息）
multi_solution_results = {
    'num_candidates': 500,
    'real_x': real_x[0].tolist(),
    'x_diversity': {
        'per_param': diversity.tolist(),
        'average': float(np.mean(diversity))
    },
    'nmse_statistics': {
        'min': float(np.min(nmse_errors)),
        'max': float(np.max(nmse_errors)),
        'mean': float(np.mean(nmse_errors)),
        'std': float(np.std(nmse_errors)),
        'median': float(np.median(nmse_errors))
    },
    'top_5_solutions': [
        {
            'rank': rank + 1,
            'candidate_idx': int(idx),
            'nmse': float(nmse_errors[idx]),
            'predicted_x': generated_xs_clipped[idx].tolist(),
            'relative_errors': np.abs((generated_xs_clipped[idx] - real_x[0]) / (real_x[0] + 1e-8)).tolist()
        }
        for rank, idx in enumerate(top_indices)
    ],
    'all_candidates_nmse': nmse_errors.tolist()
}

with open(os.path.join(checkpoint_dir, 'multi_solution_results.json'), 'w', encoding='utf-8') as f:
    json.dump(multi_solution_results, f, ensure_ascii=False, indent=2)

print('\n多解生成详细结果已保存到: multi_solution_results.json')

# ============== 计算并保存逆向预测x正确率 ==============
# 注意：逆向预测x的正确率已经在上面计算并保存过了，这里不再重复计算

# ============== Saving prediction results ==============
# Note: Prediction results for fixed x predicting y and fixed y backward predicting x have already been saved inside their respective loops

print('\n=== Training completed! ===')
print(f'Model checkpoints saved in: {checkpoint_dir}')
print('Prediction results have been saved.')