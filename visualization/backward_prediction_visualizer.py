import matplotlib.pyplot as plt
import os
import json
import numpy as np


def visualize_backward_prediction(real_x, reconstructed_x, real_y, predicted_y, freq_data, sample_index, save_dir, param_ranges_path=None):
    """
    可视化反向预测结果（固定y回推x）
    
    Args:
        real_x: 真实的x值，形状为(1, 5)
        reconstructed_x: 预测的x值，形状为(1, 5)
        real_y: 真实的y值，形状为(1, 202)，前101维为实部，后101维为虚部
        predicted_y: 使用回推的x预测的y值，形状为(1, 202)
        freq_data: 频率数据，形状为(101,)
        sample_index: 样本索引，用于图表标题和文件名
        save_dir: 保存图表的目录
        param_ranges_path: 几何参数范围文件路径，默认为None
    """
    # 确保save_dir存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载几何参数范围
    param_ranges = {}
    if param_ranges_path and os.path.exists(param_ranges_path):
        with open(param_ranges_path, 'r') as f:
            param_ranges = json.load(f)
    
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
            # 没有参数范围时，直接绘制
            ax.scatter(0.5, real_value, s=100, facecolors='none', edgecolors='blue', linewidths=2, label=f'Real {param}')
            ax.scatter(0.5, recon_value, s=100, facecolors='none', edgecolors='red', linewidths=2, label=f'Backward {param}')
        
        # 设置子图属性
        ax.set_title(f'{param}', fontsize=12)
        ax.set_ylabel('Parameter Value (mm)', fontsize=10)
        ax.set_xticks([])
        ax.set_xlim((0, 1))
        
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 创建实部曲线子图（第二行，跨越所有5列）
    ax_re = fig.add_subplot(gs[1, :])
    ax_re.plot(freq_data[:101], real_y[0, :101], 'blue', linewidth=2, label='Original Re(S11)')
    ax_re.plot(freq_data[:101], predicted_y[0, :101], 'red', linestyle='--', linewidth=2, label='Predicted Re(S11)')
    ax_re.set_xlabel('Frequency (GHz)')
    ax_re.set_ylabel('Re(S11)')
    ax_re.set_title(f'Real Part Prediction Consistency - Sample {sample_index}')
    ax_re.set_xlim((10.5, 11.5))
    ax_re.legend()
    ax_re.grid(True, alpha=0.3)
    
    # 创建虚部曲线子图（第三行，跨越所有5列）
    ax_im = fig.add_subplot(gs[2, :])
    ax_im.plot(freq_data[:101], real_y[0, 101:], 'green', linewidth=2, label='Original Im(S11)')
    ax_im.plot(freq_data[:101], predicted_y[0, 101:], 'orange', linestyle='--', linewidth=2, label='Predicted Im(S11)')
    ax_im.set_xlabel('Frequency (GHz)')
    ax_im.set_ylabel('Im(S11)')
    ax_im.set_title(f'Imaginary Part Prediction Consistency - Sample {sample_index}')
    ax_im.set_xlim((10.5, 11.5))
    ax_im.legend()
    ax_im.grid(True, alpha=0.3)
    
    # 设置整个图表的标题
    fig.suptitle(f'Fixed Y Backward X - Sample {sample_index}', fontsize=16, y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    # 保存图表
    save_path = os.path.join(save_dir, f'fixed_y_backward_x_{sample_index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'  Plot saved for sample {sample_index}: fixed_y_backward_x_{sample_index}.png')
    return save_path


def visualize_multi_solution_analysis(real_x, reconstructed_xs, predicted_ys, y_test_original, freq_data, nmse_errors, save_dir):
    """
    可视化多解生成结果分析
    
    Args:
        real_x: 真实的x值，形状为(1, 5)
        reconstructed_xs: 生成的多个x值，形状为(N, 5)
        predicted_ys: 使用生成的x预测的y值，形状为(N, 202)
        y_test_original: 原始的y值，形状为(1, 202)
        freq_data: 频率数据，形状为(101,)
        nmse_errors: 每个生成解的NMSE误差，形状为(N,)
        save_dir: 保存图表的目录
    """
    # 确保save_dir存在
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = len(reconstructed_xs)
    x_dim = reconstructed_xs.shape[1]
    
    # 获取Top 5解
    top_indices = np.argsort(nmse_errors)[:5]
    
    # 可视化：X分布 + NMSE分布 + Top 5 Y预测
    fig = plt.figure(figsize=(16, 12))
    
    # 1. X参数分布（2行3列布局）
    for param_idx in range(x_dim):
        ax = plt.subplot(3, 3, param_idx + 1)
        ax.hist(reconstructed_xs[:, param_idx], bins=20, alpha=0.7, color='green', label='Generated X')
        ax.axvline(real_x[0, param_idx], color='red', linestyle='--', linewidth=2, label='Real X')
        ax.set_title(f'Parameter {param_idx + 1} Distribution')
        ax.set_xlabel('Value (mm)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. NMSE分布直方图
    ax_nmse = plt.subplot(3, 3, 6)
    ax_nmse.hist(nmse_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax_nmse.axvline(np.min(nmse_errors), color='red', linestyle='--', linewidth=2, label=f'Best NMSE: {np.min(nmse_errors):.4f}')
    ax_nmse.axvline(np.mean(nmse_errors), color='orange', linestyle='--', linewidth=2, label=f'Mean NMSE: {np.mean(nmse_errors):.4f}')
    ax_nmse.set_title(f'NMSE Distribution ({num_samples} samples)')
    ax_nmse.set_xlabel('NMSE')
    ax_nmse.set_ylabel('Count')
    ax_nmse.legend()
    ax_nmse.grid(True, alpha=0.3)
    
    # 3. Top 5 解的Y预测 - 实部
    ax_re = plt.subplot(3, 1, 2)
    ax_re.plot(freq_data[:101], y_test_original[0, :101], 'blue', linewidth=2.5, label='Original Re(S11)')
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for rank, idx in enumerate(top_indices):
        color = colors[rank % len(colors)]
        ax_re.plot(freq_data[:101], predicted_ys[idx, :101], color=color, linestyle='--', 
                   linewidth=1.5, alpha=0.8, label=f'Rank {rank+1} (NMSE: {nmse_errors[idx]:.4f})')
    ax_re.set_xlabel('Frequency (GHz)')
    ax_re.set_ylabel('Re(S11)')
    ax_re.set_title(f'Top 5 Predicted Re(S11) - Best NMSE: {nmse_errors[top_indices[0]]:.6f}')
    ax_re.set_xlim((10.5, 11.5))
    ax_re.legend(loc='upper right', fontsize='small')
    ax_re.grid(True, alpha=0.3)
    
    # 4. Top 5 解的Y预测 - 虚部
    ax_im = plt.subplot(3, 1, 3)
    ax_im.plot(freq_data[:101], y_test_original[0, 101:], 'green', linewidth=2.5, label='Original Im(S11)')
    for rank, idx in enumerate(top_indices):
        color = colors[rank % len(colors)]
        ax_im.plot(freq_data[:101], predicted_ys[idx, 101:], color=color, linestyle='--', 
                   linewidth=1.5, alpha=0.8, label=f'Rank {rank+1} (NMSE: {nmse_errors[idx]:.4f})')
    ax_im.set_xlabel('Frequency (GHz)')
    ax_im.set_ylabel('Im(S11)')
    ax_im.set_title(f'Top 5 Predicted Im(S11) - Best NMSE: {nmse_errors[top_indices[0]]:.6f}')
    ax_im.set_xlim((10.5, 11.5))
    ax_im.legend(loc='upper right', fontsize='small')
    ax_im.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, 'multi_solution_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\nMulti-solution analysis saved: multi_solution_analysis.png')
    return save_path


def visualize_x_distribution(reconstructed_xs, real_x, save_dir):
    """
    可视化生成的X的分布
    
    Args:
        reconstructed_xs: 生成的多个x值，形状为(N, 5)
        real_x: 真实的x值，形状为(1, 5)
        save_dir: 保存图表的目录
    """
    # 确保save_dir存在
    os.makedirs(save_dir, exist_ok=True)
    
    x_dim = reconstructed_xs.shape[1]
    
    plt.figure(figsize=(12, 8))
    
    for param_idx in range(x_dim):
        plt.subplot(2, 3, param_idx + 1)
        plt.hist(reconstructed_xs[:, param_idx], bins=10, alpha=0.7, color='green', label='Generated X')
        plt.axvline(real_x[0, param_idx], color='red', linestyle='--', label='Real X')
        plt.title(f'Parameter {param_idx + 1} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, 'multi_solution_x_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path
