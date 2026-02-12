import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 添加项目根目录到Python路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型和工具
from R_INN_model.rinn_model import RINNModel
from filter_ideal.core import generate_ideal_response

# 配置参数
result_dir = 'result_RINN_transformed'
os.makedirs(result_dir, exist_ok=True)

# 加载最佳模型
model_path = 'model_checkpoints_rinn/rinn_correct_structure_20260212_223456/best_model.pth'
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# 恢复模型参数
x_dim = checkpoint['x_dim']
y_dim = checkpoint['y_dim']
z_dim = checkpoint['z_dim']
left_input_dim = checkpoint['left_input_dim']

# 创建模型
model = RINNModel(
    input_dim=left_input_dim,
    hidden_dim=56,  # 从最佳配置
    num_blocks=4,    # 从最佳配置
    num_stages=2,    # 从最佳配置
    num_cycles_per_stage=2,  # 从最佳配置
    ratio_toZ_after_flowstage=0.273,  # 从最佳配置
    ratio_x1_x2_inAffine=0.421  # 从最佳配置
)

# 加载模型权重
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 恢复归一化参数
x_mean = checkpoint['x_mean']
x_std = checkpoint['x_std']
y_mean = checkpoint['y_mean']
y_std = checkpoint['y_std']

# 生成理想数据
def generate_ideal_data():
    """生成理想数据和变换后的数据"""
    from filter_ideal.core import filter_configs
    
    # 生成理想响应
    cfg = filter_configs[0]
    result = generate_ideal_response(cfg, result_dir='result_ideal_transformed')
    
    # 获取原始数据和变换后的数据
    freq = result['freq']
    S11_real = result['S11_real']
    S11_imag = result['S11_imag']
    transformed_data = result['transformed_data']
    
    # 提取变换后的数据
    S11_real_neg = transformed_data['negative']['real']
    S11_imag_neg = transformed_data['negative']['imag']
    S11_real_sym = transformed_data['symmetric']['real']
    S11_imag_sym = transformed_data['symmetric']['imag']
    
    return {
        'freq': freq,
        'original': {
            'real': S11_real,
            'imag': S11_imag
        },
        'negative': {
            'real': S11_real_neg,
            'imag': S11_imag_neg
        },
        'symmetric': {
            'real': S11_real_sym,
            'imag': S11_imag_sym
        }
    }

# 对数据应用R-INN模型
def apply_rinn_model(data):
    """对数据应用R-INN模型，执行逆向和正向预测"""
    results = {}
    
    for data_type, data_values in data.items():
        if data_type == 'freq':
            results['freq'] = data_values
            continue
        
        print(f"Processing {data_type} data...")
        
        # 准备输入数据
        real_part = data_values['real']
        imag_part = data_values['imag']
        
        # 合并实部和虚部
        y_data = np.concatenate((real_part, imag_part), axis=0)
        y_data = y_data.reshape(1, -1)  # 形状：(1, 202)
        
        # 归一化
        y_normalized = (y_data - y_mean) / (y_std + 1e-8)
        
        # 计算原始S11
        def calculate_s11(real, imag):
            s11_complex = real + 1j * imag
            s11_abs = np.abs(s11_complex)
            s11_dB = 20 * np.log10(s11_abs)
            return s11_dB

        # 计算NMSE
        def calculate_nmse(original, predicted):
            """计算归一化均方误差"""
            mse = np.mean((original - predicted) ** 2)
            variance = np.var(original)
            if variance == 0:
                return 0
            nmse = mse / variance
            return nmse
        
        original_s11 = calculate_s11(real_part, imag_part)
        
        # 采样100个Z值并选择最佳结果
        print(f"Sampling 100 Z values for {data_type} data...")
        best_nmse = float('inf')
        best_result = None
        
        for i in range(100):
            # 生成随机Z
            z_data = np.random.randn(1, z_dim).astype(np.float32)
            
            # 右侧输入：Y + Z
            right_input = np.concatenate((y_normalized, z_data), axis=1)
            right_input = torch.FloatTensor(right_input)
            
            # 逆向预测：重建几何参数
            with torch.no_grad():
                reconstructed_left, _ = model.inverse(right_input)
                
                # 提取重建的X
                reconstructed_x_normalized = reconstructed_left[:, :x_dim]
                reconstructed_x = reconstructed_x_normalized.numpy() * x_std + x_mean
            
            # 正向预测：使用重建的X预测Y
            with torch.no_grad():
                # 左侧输入：重建的X + 零填充
                padding_dim = y_dim
                left_input = np.concatenate((reconstructed_x_normalized.numpy(), np.zeros((1, padding_dim), dtype=np.float32)), axis=1)
                left_input = torch.FloatTensor(left_input)
                
                # 正向预测
                predicted_right, _ = model(left_input)
                
                # 提取预测的Y
                predicted_y_normalized = predicted_right[:, :y_dim]
                predicted_y = predicted_y_normalized.numpy() * y_std + y_mean
            
            # 分离实部和虚部
            predicted_real = predicted_y[0, :101]
            predicted_imag = predicted_y[0, 101:]
            
            # 计算预测的S11
            predicted_s11 = calculate_s11(predicted_real, predicted_imag)
            
            # 计算NMSE
            nmse = calculate_nmse(original_s11, predicted_s11)
            
            # 更新最佳结果
            if nmse < best_nmse:
                best_nmse = nmse
                best_result = {
                    'reconstructed_x': reconstructed_x[0],
                    'predicted_real': predicted_real,
                    'predicted_imag': predicted_imag,
                    'predicted_s11': predicted_s11,
                    'nmse': nmse
                }
            
            # 每10个Z值打印一次进度
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/100 Z values, current best NMSE: {best_nmse:.6f}")
        
        print(f"Best NMSE for {data_type} data: {best_nmse:.6f}")
        
        # 存储最佳结果
        results[data_type] = {
            'real': real_part,
            'imag': imag_part,
            'reconstructed_x': best_result['reconstructed_x'],
            'predicted_real': best_result['predicted_real'],
            'predicted_imag': best_result['predicted_imag'],
            'original_s11': original_s11,
            'predicted_s11': best_result['predicted_s11'],
            'nmse': best_result['nmse']
        }
    
    return results

# 生成第一张图：重建几何参数和Re/Im曲线
def plot_reconstructed_parameters_and_curves(results):
    """生成包含重建几何参数和Re/Im曲线的图"""
    freq = results['freq']
    
    # 加载几何参数范围
    param_ranges = {}
    if os.path.exists('geometry_params_ranges.json'):
        with open('geometry_params_ranges.json', 'r') as f:
            param_ranges = json.load(f)
    
    # 参数名称
    params = ['H1', 'H2', 'H3', 'H_C1', 'H_C2']
    
    # 为每个数据类型生成图
    for data_type in ['original', 'negative', 'symmetric']:
        if data_type not in results:
            continue
        
        data = results[data_type]
        reconstructed_x = data['reconstructed_x']
        real_part = data['real']
        imag_part = data['imag']
        predicted_real = data['predicted_real']
        predicted_imag = data['predicted_imag']
        nmse = data.get('nmse', 0)
        
        # 创建图：5个几何参数子图（横向排列） + 2个曲线子图（纵向排列）
        fig = plt.figure(figsize=(20, 15))
        
        # 创建整个图表的网格布局
        gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1])
        
        # 创建5个几何参数子图（第一行横向排列）
        param_axes = []
        for i, (param, value) in enumerate(zip(params, reconstructed_x)):
            ax = fig.add_subplot(gs[0, i])
            param_axes.append(ax)
            
            # 绘制竖线代表数轴
            ax.axvline(x=0.5, color='black', linewidth=1)
            
            # 获取参数范围
            if param in param_ranges:
                y_min = param_ranges[param]['min']
                y_max = param_ranges[param]['max']
                
                # 绘制上下界参考线
                ax.axhline(y=y_min, xmin=0.2, xmax=0.8, color='green', linestyle='--', linewidth=1, label=f'{param} min')
                ax.axhline(y=y_max, xmin=0.2, xmax=0.8, color='red', linestyle='--', linewidth=1, label=f'{param} max')
                
                # 绘制取值（空心圆圈）
                if value < y_min or value > y_max:
                    # 超出范围，用橙色圆圈
                    ax.scatter(0.5, value, s=100, facecolors='none', edgecolors='orange', linewidths=2, label=f'{param} Value')
                    # 添加文本标注
                    ax.text(0.5, value, f'Out of range', 
                             ha='center', va='bottom' if value > y_max else 'top',
                             fontsize=8, color='red')
                else:
                    # 在范围内，用蓝色圆圈
                    ax.scatter(0.5, value, s=100, facecolors='none', edgecolors='blue', linewidths=2, label=f'{param} Value')
                
                # 设置Y轴范围
                ax.set_ylim([y_min * 0.95, y_max * 1.05])
            
            # 设置子图属性
            ax.set_title(f'{param}', fontsize=12)
            ax.set_ylabel('Parameter Value (mm)', fontsize=10)
            ax.set_xticks([])
            ax.set_xlim([0, 1])
            
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 创建实部曲线子图（第二行，跨越所有5列）
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(freq/1e9, real_part, 'blue', linewidth=2, label='Original Re(S11)')
        ax2.plot(freq/1e9, predicted_real, 'red', linestyle='--', linewidth=2, label='Predicted Re(S11)')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Re(S11)')
        ax2.set_title(f'Real Part Comparison ({data_type} data)')
        ax2.set_xlim([10.5, 11.5])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 创建虚部曲线子图（第三行，跨越所有5列）
        ax3 = fig.add_subplot(gs[2, :])
        ax3.plot(freq/1e9, imag_part, 'blue', linewidth=2, label='Original Im(S11)')
        ax3.plot(freq/1e9, predicted_imag, 'red', linestyle='--', linewidth=2, label='Predicted Im(S11)')
        ax3.set_xlabel('Frequency (GHz)')
        ax3.set_ylabel('Im(S11)')
        ax3.set_title(f'Imaginary Part Comparison ({data_type} data)')
        ax3.set_xlim([10.5, 11.5])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 设置整个图表的标题
        fig.suptitle(f'Reconstructed Geometry Parameters and S11 Curves ({data_type} data)\nBest NMSE: {nmse:.6f}', fontsize=16, y=0.98)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(result_dir, f'{data_type}_parameters_and_curves.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f'Saved {data_type} parameters and curves plot to {save_path}')

# 生成第二张图：S11值
def plot_s11_values(results):
    """生成包含S11值的图"""
    freq = results['freq']
    
    # 为每个数据类型生成图
    for data_type in ['original', 'negative', 'symmetric']:
        if data_type not in results:
            continue
        
        data = results[data_type]
        original_s11 = data['original_s11']
        predicted_s11 = data['predicted_s11']
        
        # 创建图
        plt.figure(figsize=(12, 7))
        
        plt.plot(freq/1e9, original_s11, 'blue', linewidth=2, label='Original S11 (dB)')
        plt.plot(freq/1e9, predicted_s11, 'red', linestyle='--', linewidth=2, label='Predicted S11 (dB)')
        
        # 通带阴影
        plt.axvspan(10.85, 11.15, color='green', alpha=0.15, label='Passband(10.85-11.15GHz)')
        
        # 坐标与样式
        plt.xlim(10.5, 11.5)
        plt.ylim(-60, 0)
        plt.xlabel('Frequency (GHz)', fontsize=14)
        plt.ylabel('S11 (dB)', fontsize=14)
        plt.title(f'S11 Comparison ({data_type} data)', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # 保存图
        save_path = os.path.join(result_dir, f'{data_type}_s11.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f'Saved {data_type} S11 plot to {save_path}')

# 主函数
def main():
    print('Generating ideal data...')
    ideal_data = generate_ideal_data()
    
    print('Applying R-INN model to transformed data...')
    results = apply_rinn_model(ideal_data)
    
    print('Generating reconstructed parameters and curves plots...')
    plot_reconstructed_parameters_and_curves(results)
    
    print('Generating S11 values plots...')
    plot_s11_values(results)
    
    print('All tasks completed successfully!')

if __name__ == '__main__':
    main()
