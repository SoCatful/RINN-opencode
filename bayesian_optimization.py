"""
贝叶斯优化脚本，用于优化RINN模型的超参数
"""
import os
import torch
import numpy as np
import json
import time
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# 确保optuna模块已安装
try:
    import optuna
except ImportError:
    print("正在安装optuna模块...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

print(f"使用optuna版本: {optuna.__version__}")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 导入R_INN_model包中的工具函数
from R_INN_model.data_utils import extract_geometry_params, load_data_from_csv, normalize_data
from R_INN_model.device_utils import get_device
from R_INN_model.training_utils import calculate_loss, train_model

# 设备配置
device = get_device()
print(f'使用设备: {device}')

# ============== 数据加载与预处理 ==============
print('\n=== 加载数据 ===')

# 加载训练数据
train_data_path = 'data/S Parameter Plot300.csv'
train_x, train_y, freq_data = load_data_from_csv(train_data_path)

# 加载验证数据
val_data_path = 'data/S Parameter Plot200.csv'
val_x, val_y, _ = load_data_from_csv(val_data_path)

print(f'\n训练集样本数: {len(train_x)}')
print(f'验证集样本数: {len(val_x)}')

# 归一化数据
train_x_normalized, val_x_normalized, train_y_normalized, val_y_normalized, x_mean, x_std, y_mean, y_std = normalize_data(
    train_x, val_x, train_y, val_y, method='robust'
)

# ============== 维度处理 ==============
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

# 创建训练集和验证集数据
# 左侧输入：X + 零填充
left_train_input = np.concatenate((train_x_normalized, np.zeros((len(train_x_normalized), padding_dim), dtype=np.float32)), axis=1)
left_val_input = np.concatenate((val_x_normalized, np.zeros((len(val_x_normalized), padding_dim), dtype=np.float32)), axis=1)

# 注意：Z现在在每个epoch重新采样，不再在这里固定生成
# 右侧输入将在训练循环中动态生成

# 转换为torch张量（左侧输入固定）
left_train = torch.FloatTensor(left_train_input)
left_val = torch.FloatTensor(left_val_input)

print('\n数据集划分:')
print(f'  训练集: {len(left_train)} 样本')
print(f'  验证集: {len(left_val)} 样本')
print('  注意：Z将在每个epoch重新采样，增强模型泛化能力')

# ============== 模型定义与损失函数 ==============
from R_INN_model.rinn_model import RINNModel
from R_INN_model.loss_methods import mmd_loss, nmse_loss, weighted_nmse_loss



# ============== 目标函数 ==============
def objective(trial):
    """优化目标函数"""
    # 超参数搜索空间
    params = {
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 128, step=8),
        "num_blocks": trial.suggest_int("num_blocks", 3, 10),
        "num_stages": trial.suggest_int("num_stages", 1, 4),
        "num_cycles_per_stage": trial.suggest_int("num_cycles_per_stage", 1, 4),
        "ratio_toZ_after_flowstage": trial.suggest_float("ratio_toZ_after_flowstage", 0.1, 0.8),
        "ratio_x1_x2_inAffine": trial.suggest_float("ratio_x1_x2_inAffine", 0.05, 0.6),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
        "weight_y": trial.suggest_float("weight_y", 0.1, 3.0),
        "weight_x": trial.suggest_float("weight_x", 0.1, 2.0),
        "weight_z": trial.suggest_float("weight_z", 0.1, 1.0)
    }
    
    print(f"\n尝试参数: {params}")
    
    # 获取batch_size
    batch_size = params["batch_size"]
    
    # 创建模型
    model = RINNModel(
        input_dim=left_input_dim,
        hidden_dim=params["hidden_dim"],
        num_blocks=params["num_blocks"],
        num_stages=params["num_stages"],
        num_cycles_per_stage=params["num_cycles_per_stage"],
        ratio_toZ_after_flowstage=params["ratio_toZ_after_flowstage"],
        ratio_x1_x2_inAffine=params["ratio_x1_x2_inAffine"]
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"]
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6
    )
    
    # 训练模型
    try:
        best_val_loss, _, _ = train_model(
            model, left_train, left_val, train_y_normalized, val_y_normalized,
            batch_size, optimizer, scheduler,
            num_epochs=100,  # 大幅增加训练轮数：100 epochs
            x_dim=x_dim, y_dim=y_dim, z_dim=z_dim,
            weight_y=params["weight_y"],
            weight_x=params["weight_x"],
            weight_z=params["weight_z"],
            device=device,
            patience=30  # 增加早停耐心
        )
        print(f"最佳验证损失: {best_val_loss:.6f}")
        return best_val_loss
    except Exception as e:
        print(f"训练失败: {e}")
        return float('inf')

# ============== 主函数 ==============
def main():
    # 创建优化器
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,  # 增加初始随机搜索次数
            seed=42
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        )
    )
    
    # 运行优化
    n_trials = 400  # 大幅增加尝试次数：400次尝试
    print(f"\n开始贝叶斯优化，计划运行 {n_trials} 次尝试...")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('optimization_results', f'rinn_bayesian_opt_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # 最佳参数
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n=== 最佳参数 ===")
    print(f"最佳验证损失: {best_value:.6f}")
    print(f"最佳参数: {best_params}")
    
    # 保存最佳参数
    with open(os.path.join(results_dir, 'best_params.json'), 'w', encoding='utf-8') as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    
    # 保存所有尝试的参数和结果
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name
        })
    
    with open(os.path.join(results_dir, 'all_trials.json'), 'w', encoding='utf-8') as f:
        json.dump(trials_data, f, ensure_ascii=False, indent=2)
    
    # 保存优化历史
    with open(os.path.join(results_dir, 'optimization_history.txt'), 'w') as f:
        f.write(f"贝叶斯优化结果\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write(f"尝试次数: {n_trials}\n")
        f.write(f"最佳验证损失: {best_value:.6f}\n")
        f.write(f"最佳参数: {json.dumps(best_params, ensure_ascii=False, indent=2)}\n")
    
    print(f"\n优化结果已保存到: {results_dir}")
    
    # 可视化优化过程
    try:
        import matplotlib.pyplot as plt
        # 设置中文字体支持 - 尝试多种字体，提高兼容性
        plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 绘制验证损失随尝试次数的变化
        plt.figure(figsize=(12, 6))
        values = [trial.value for trial in study.trials if trial.value is not None]
        plt.plot(range(len(values)), values, 'b-', alpha=0.5)
        plt.scatter(range(len(values)), values, c='b', s=10)
        plt.xlabel('尝试次数')
        plt.ylabel('验证损失')
        plt.title('贝叶斯优化过程')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'optimization_history.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("优化历史图已保存")
        
        # 绘制参数重要性
        importances = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importances)), list(importances.values()), align='center')
        plt.xticks(range(len(importances)), list(importances.keys()), rotation=45, ha='right')
        plt.xlabel('参数')
        plt.ylabel('重要性')
        plt.title('参数重要性分析')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'param_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("参数重要性图已保存")
    except Exception as e:
        print(f"绘制图表失败: {e}")

if __name__ == "__main__":
    # 检查是否安装了optuna
    try:
        import optuna
        print(f"Optuna版本: {optuna.__version__}")
    except ImportError:
        print("正在安装Optuna...")
        import subprocess
        subprocess.run(["pip", "install", "optuna"], check=True)
        import optuna
        print(f"Optuna版本: {optuna.__version__}")
    
    # 检查是否安装了matplotlib
    try:
        import matplotlib
        print(f"Matplotlib版本: {matplotlib.__version__}")
    except ImportError:
        print("正在安装Matplotlib...")
        import subprocess
        subprocess.run(["pip", "install", "matplotlib"], check=True)
    
    # 运行主函数
    main()
