"""
可视化工具函数
"""
import matplotlib.pyplot as plt
import json
import torch
import numpy as np

def plot_training_curves(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # Total loss curve
    axes[0].plot(train_losses['total'], label='Train Total', color='blue')
    axes[0].plot(val_losses['total'], label='Val Total', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Y loss curve (NMSE)
    axes[1].plot(train_losses['y_loss'], label='Train Y Loss', color='blue')
    axes[1].plot(val_losses['y_loss'], label='Val Y Loss', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Y Loss (NMSE)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # X loss curve (MSE)
    axes[2].plot(train_losses['x_loss'], label='Train X Loss', color='blue')
    axes[2].plot(val_losses['x_loss'], label='Val X Loss', color='red')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('X Loss (MSE)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Z loss curve (MMD)
    axes[3].plot(train_losses['z_loss'], label='Train Z Loss', color='blue')
    axes[3].plot(val_losses['z_loss'], label='Val Z Loss', color='red')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Loss')
    axes[3].set_title('Z Loss (MMD)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to: {save_path}')

def plot_predicted_y(freq_data, real_y, predicted_y, sample_idx, save_path):
    """绘制预测结果（固定x预测y）"""
    # 确保real_y和predicted_y维度正确（101维dB值）
    real_y = real_y[:101]  # 只保留101维dB值
    predicted_y = predicted_y[:101]  # 只保留101维dB值

    # 可视化预测结果 - 显示dB值
    plt.figure(figsize=(12, 6))
    plt.plot(freq_data[:101], real_y, label='Real S11 (dB)', color='blue', linewidth=2)
    plt.plot(freq_data[:101], predicted_y, label='Predicted S11 (dB)', color='red', linestyle='--', linewidth=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('S11 (dB)')
    plt.title(f'Comparison of S11 (dB) prediction with fixed x - Sample {sample_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved for sample {sample_idx}: {save_path}')

def plot_backward_prediction(model, y_test, real_x, real_y, z_dim, x_dim, y_dim, padding_dim, x_mean, x_std, y_mean, y_std, freq_data, sample_idx, save_path, device):
    """绘制逆向预测结果（固定y回推x）"""
    # 创建右侧输入：Y + Z（Z是随机生成的标准高斯分布）
    z_test = np.random.randn(1, z_dim).astype(np.float32)
    right_test_input = np.concatenate((y_test, z_test), axis=1)
    right_test_input = torch.FloatTensor(right_test_input).to(device)

    # 使用模型进行反向预测
    with torch.no_grad():
        reconstructed_left, _ = model.inverse(right_test_input)
        
        # 从reconstructed_left中提取X'
        reconstructed_x_normalized = reconstructed_left[:, :x_dim]
        
        # 反标准化得到回推的x
        reconstructed_x = reconstructed_x_normalized.cpu().numpy() * x_std + x_mean

    # 使用回推的x进行正向预测，验证一致性
    with torch.no_grad():
        # 左侧输入：回推的X + 零填充
        left_test_input = np.concatenate((reconstructed_x_normalized.cpu().numpy(), np.zeros((1, padding_dim), dtype=np.float32)), axis=1)
        left_test_input = torch.FloatTensor(left_test_input).to(device)
        
        # 正向预测
        predicted_right, _, _ = model(left_test_input, return_intermediate=True)
        
        # 从predicted_right中提取Y'，并确保维度是202维
        predicted_y_normalized = predicted_right[:, :y_dim]
        
        # 反标准化得到预测的y
        predicted_y = predicted_y_normalized.cpu().numpy() * y_std + y_mean

    # 确保real_y和predicted_y维度正确（101维dB值）
    real_y = real_y[0, :101]  # 只保留101维dB值
    predicted_y = predicted_y[0, :101]  # 只保留101维dB值

    # 加载几何参数范围
    param_ranges = {}
    try:
        with open('geometry_params_ranges.json', 'r') as f:
            param_ranges = json.load(f)
    except Exception:
        pass
    
    # 参数名称
    params = ['H1', 'H2', 'H3', 'H_C1', 'H_C2']
    
    # 创建图：5个几何参数子图（横向排列） + 2个曲线子图（纵向排列）
    fig = plt.figure(figsize=(20, 15))
    
    # 创建整个图表的网格布局
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1])
    
    # 创建5个几何参数子图（第一行横向排列）
    for j, (param, real_value, recon_value) in enumerate(zip(params, real_x[0], reconstructed_x[0])):
        ax = fig.add_subplot(gs[0, j])
        
        # 绘制竖线代表数轴
        ax.axvline(x=0.5, color='black', linewidth=1)
        
        # 获取参数范围
        if param in param_ranges:
            y_min = param_ranges[param]['min']
            y_max = param_ranges[param]['max']
            
            # 绘制上下界参考线
            ax.axhline(y=y_min, xmin=0.2, xmax=0.8, color='green', linestyle='--', linewidth=1, label=f'{param} min')
            ax.axhline(y=y_max, xmin=0.2, xmax=0.8, color='red', linestyle='--', linewidth=1, label=f'{param} max')
            
            # 绘制真实值（蓝色空心圆圈）
            ax.scatter(0.5, real_value, s=100, facecolors='none', edgecolors='blue', linewidths=2, label=f'Real {param}')
            
            # 绘制回推值（红色空心圆圈）
            if recon_value < y_min or recon_value > y_max:
                # 超出范围，用橙色圆圈
                ax.scatter(0.5, recon_value, s=100, facecolors='none', edgecolors='orange', linewidths=2, label=f'Backward {param}')
                ax.text(0.5, recon_value, f'Out of range', 
                         ha='center', va='bottom' if recon_value > y_max else 'top',
                         fontsize=8, color='red')
            else:
                ax.scatter(0.5, recon_value, s=100, facecolors='none', edgecolors='red', linewidths=2, label=f'Backward {param}')
            
            # 设置Y轴范围
            ax.set_ylim((y_min * 0.95, y_max * 1.05))
        else:
            # 绘制真实值（蓝色空心圆圈）
            ax.scatter(0.5, real_value, s=100, facecolors='none', edgecolors='blue', linewidths=2, label=f'Real {param}')
            # 绘制回推值（红色空心圆圈）
            ax.scatter(0.5, recon_value, s=100, facecolors='none', edgecolors='red', linewidths=2, label=f'Backward {param}')
        
        # 设置子图属性
        ax.set_title(f'{param}', fontsize=12)
        ax.set_ylabel('Parameter Value (mm)', fontsize=10)
        ax.set_xticks([])
        ax.set_xlim((0, 1))
        
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 创建dB值曲线子图（第二行，跨越所有5列）
    ax2 = fig.add_subplot(gs[1:, :])
    ax2.plot(freq_data[:101], real_y, 'blue', linewidth=2, label='Original S11 (dB)')
    ax2.plot(freq_data[:101], predicted_y, 'red', linestyle='--', linewidth=2, label='Predicted S11 (dB)')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('S11 (dB)')
    ax2.set_title(f'Prediction Consistency - Sample {sample_idx}')
    ax2.set_xlim((10.5, 11.5))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 设置整个图表的标题
    fig.suptitle(f'Fixed Y Backward X - Sample {sample_idx}', fontsize=16, y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Plot saved for sample {sample_idx}: {save_path}')

def plot_multi_solution_distribution(generated_xs_clipped, real_x, nmse_errors, predicted_ys, y_test_original, freq_data, save_path):
    """绘制多解生成的X分布和预测结果"""
    # 可视化：X分布 + NMSE分布 + Top 5 Y预测
    fig = plt.figure(figsize=(16, 12))

    # 1. X参数分布（2行3列布局）
    x_dim = real_x.shape[0]
    for param_idx in range(x_dim):
        ax = plt.subplot(3, 3, param_idx + 1)
        ax.hist(generated_xs_clipped[:, param_idx], bins=20, alpha=0.7, color='green', label='Generated X')
        ax.axvline(real_x[param_idx], color='red', linestyle='--', linewidth=2, label='Real X')
        ax.set_title(f'Parameter {param_idx + 1} Distribution')
        ax.set_xlabel('Value (mm)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. NMSE分布直方图
    ax_nmse = plt.subplot(3, 3, 6)
    try:
        # 检查所有值是否相同
        if len(np.unique(nmse_errors)) == 1:
            # 如果所有值相同，绘制一个简单的条形图
            ax_nmse.bar([0], [len(nmse_errors)], color='blue', alpha=0.7, edgecolor='black')
            ax_nmse.set_xticks([0])
            ax_nmse.set_xticklabels([f'{nmse_errors[0]:.2f}'])
        else:
            # 正常绘制直方图
            ax_nmse.hist(nmse_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax_nmse.axvline(np.min(nmse_errors), color='red', linestyle='--', linewidth=2, label=f'Best NMSE: {np.min(nmse_errors):.4f}')
        ax_nmse.axvline(np.mean(nmse_errors), color='orange', linestyle='--', linewidth=2, label=f'Mean NMSE: {np.mean(nmse_errors):.4f}')
    except Exception as e:
        # 如果出现错误，绘制一个空图
        ax_nmse.text(0.5, 0.5, f'Error plotting NMSE: {str(e)}', ha='center', va='center')
    ax_nmse.set_title(f'NMSE Distribution ({len(nmse_errors)} samples)')
    ax_nmse.set_xlabel('NMSE')
    ax_nmse.set_ylabel('Count')
    ax_nmse.legend()
    ax_nmse.grid(True, alpha=0.3)

    # 3. Top 5 解的Y预测 - dB值
    top_indices = np.argsort(nmse_errors)[:5]
    ax_db = plt.subplot(3, 1, 2)
    ax_db.plot(freq_data[:101], y_test_original[:101], 'blue', linewidth=2.5, label='Original S11 (dB)')
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for rank, idx in enumerate(top_indices):
        color = colors[rank % len(colors)]
        ax_db.plot(freq_data[:101], predicted_ys[idx, :101], color=color, linestyle='--', 
                   linewidth=1.5, alpha=0.8, label=f'Rank {rank+1} (NMSE: {nmse_errors[idx]:.4f})')
    ax_db.set_xlabel('Frequency (GHz)')
    ax_db.set_ylabel('S11 (dB)')
    ax_db.set_title(f'Top 5 Predicted S11 (dB) - Best NMSE: {nmse_errors[top_indices[0]]:.6f}')
    ax_db.set_xlim((10.5, 11.5))
    ax_db.legend(loc='upper right', fontsize='small')
    ax_db.grid(True, alpha=0.3)

    # 4. 额外的空白子图（保持布局一致）
    ax_blank = plt.subplot(3, 1, 3)
    ax_blank.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Multi-solution analysis saved: {save_path}')