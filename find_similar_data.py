import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从filter_ideal/core.py导入必要的函数
from filter_ideal.core import generate_ideal_response

# 加载训练数据
def load_training_data():
    """加载所有训练数据"""
    data_files = ['data/S Parameter Plot300.csv', 'data/S Parameter Plot200.csv', 'data/S Parameter Plot1perfect.csv']
    
    all_data = []
    all_params = []
    
    for file_path in data_files:
        # 读取频率数据
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        freq_data = data[:, 0]  # 频率数据
        
        # 读取S11数据（实部和虚部）
        # 提取所有列，跳过第一列频率
        s11_data = data[:, 1:]
        
        # 每两个列组成一个样本（实部+虚部）
        num_samples = s11_data.shape[1] // 2
        
        for i in range(num_samples):
            real_part = s11_data[:, 2*i]
            imag_part = s11_data[:, 2*i+1]
            
            # 计算S11 dB值
            s11_complex = real_part + 1j * imag_part
            s11_abs = np.abs(s11_complex)
            s11_dB = 20 * np.log10(s11_abs)
            
            all_data.append({
                'real': real_part,
                'imag': imag_part,
                's11_dB': s11_dB
            })
    
    return all_data, freq_data

# 计算两个S11曲线之间的距离
def calculate_distance(s11_1, s11_2):
    """计算两个S11曲线之间的均方误差"""
    return np.mean((s11_1 - s11_2) ** 2)

# 找到与理想S11最相近的10个数据
def find_similar_data(ideal_data, training_data):
    """找到与理想S11最相近的10个训练数据"""
    # 计算每个训练数据与理想数据的距离
    distances = []
    for i, data in enumerate(training_data):
        # 计算S11 dB的距离
        s11_distance = calculate_distance(ideal_data['s11_dB'], data['s11_dB'])
        distances.append((i, s11_distance))
    
    # 按距离排序，取前10个
    distances.sort(key=lambda x: x[1])
    top_10_indices = [x[0] for x in distances[:10]]
    
    # 提取最相近的10个数据
    top_10_data = [training_data[i] for i in top_10_indices]
    
    return top_10_data

# 生成S11对比图
def plot_s11_comparison(ideal_data, top_10_data, freq_data, result_dir):
    """生成S11对比图"""
    plt.figure(figsize=(12, 7))
    
    # 绘制理想S11
    plt.plot(freq_data, ideal_data['s11_dB'], 'red', linewidth=3, label='Ideal S11 (Negative Transformation)')
    
    # 计算通带频率范围的索引
    passband_start = 10.85
    passband_end = 11.15
    passband_indices = np.where((freq_data >= passband_start) & (freq_data <= passband_end))[0]
    
    # 绘制最相近的10个训练数据的S11
    for i, data in enumerate(top_10_data):
        alpha = 0.3 + (0.7 * (9 - i) / 9)  # 控制透明度，最相近的透明度最高
        # 计算该样本在通带内满足要求的百分比（S11 ≤ -26dB）
        passband_s11 = data['s11_dB'][passband_indices]
        meets_requirement = passband_s11 <= -26
        percentage = np.mean(meets_requirement) * 100
        # 绘制曲线
        plt.plot(freq_data, data['s11_dB'], linewidth=1.5, alpha=alpha, label=f'Top {i+1} ({percentage:.1f}% meets req)')
    
    # 通带阴影
    plt.axvspan(10.85, 11.15, color='green', alpha=0.15, label='Passband(10.85-11.15GHz)')
    
    # 需求红线
    plt.axhline(-26, color='red', linestyle='--', linewidth=2, label='-26dB Requirement Line')
    
    # 坐标与样式
    plt.xlim(10.5, 11.5)
    plt.ylim(-60, 0)
    plt.xlabel('Frequency (GHz)', fontsize=14)
    plt.ylabel('S11 (dB)', fontsize=14)
    plt.title('Comparison of Ideal S11 with Top 10 Most Similar Training Data', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')
    
    # 添加修复说明
    plt.figtext(0.02, 0.02, 'Fix: Used negative transformation on Re(S11) and Im(S11) to better match training data distribution', 
                fontsize=10, ha='left', va='bottom', color='blue')
    
    plt.tight_layout()
    
    # 保存图
    save_path = os.path.join(result_dir, 's11_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f'Saved S11 comparison plot to {save_path}')

# 生成Re/Im对比图
def plot_re_im_comparison(ideal_data, top_10_data, freq_data, result_dir):
    """生成Re/Im对比图"""
    # 实部对比
    plt.figure(figsize=(12, 7))
    
    # 绘制理想实部（取相反数）
    plt.plot(freq_data, ideal_data['real'], 'red', linewidth=3, label='Ideal Re(S11) (Negative Transformation)')
    
    # 绘制最相近的10个训练数据的实部
    for i, data in enumerate(top_10_data):
        alpha = 0.3 + (0.7 * (9 - i) / 9)  # 控制透明度，最相近的透明度最高
        plt.plot(freq_data, data['real'], linewidth=1.5, alpha=alpha, label=f'Top {i+1} Similar')
    
    # 通带阴影
    plt.axvspan(10.85, 11.15, color='green', alpha=0.15, label='Passband(10.85-11.15GHz)')
    
    # 坐标与样式
    plt.xlim(10.5, 11.5)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Frequency (GHz)', fontsize=14)
    plt.ylabel('Re(S11)', fontsize=14)
    plt.title('Comparison of Ideal Re(S11) with Top 10 Most Similar Training Data', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    
    # 保存图
    save_path = os.path.join(result_dir, 're_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f'Saved Re(S11) comparison plot to {save_path}')
    
    # 虚部对比
    plt.figure(figsize=(12, 7))
    
    # 绘制理想虚部（取相反数）
    plt.plot(freq_data, ideal_data['imag'], 'red', linewidth=3, label='Ideal Im(S11) (Negative Transformation)')
    
    # 绘制最相近的10个训练数据的虚部
    for i, data in enumerate(top_10_data):
        alpha = 0.3 + (0.7 * (9 - i) / 9)  # 控制透明度，最相近的透明度最高
        plt.plot(freq_data, data['imag'], linewidth=1.5, alpha=alpha, label=f'Top {i+1} Similar')
    
    # 通带阴影
    plt.axvspan(10.85, 11.15, color='green', alpha=0.15, label='Passband(10.85-11.15GHz)')
    
    # 坐标与样式
    plt.xlim(10.5, 11.5)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Frequency (GHz)', fontsize=14)
    plt.ylabel('Im(S11)', fontsize=14)
    plt.title('Comparison of Ideal Im(S11) with Top 10 Most Similar Training Data', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    
    # 保存图
    save_path = os.path.join(result_dir, 'im_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f'Saved Im(S11) comparison plot to {save_path}')

# 主函数
def main():
    result_dir = 'result_similarity_analysis'
    os.makedirs(result_dir, exist_ok=True)
    
    print('Generating ideal data...')
    from filter_ideal.core import filter_configs
    cfg = filter_configs[0]
    ideal_result = generate_ideal_response(cfg, result_dir='result_ideal_temp')
    
    # 计算理想数据的S11 dB值（使用取相反数后的Re和Im）
    ideal_real = -ideal_result['S11_real']  # 取相反数
    ideal_imag = -ideal_result['S11_imag']  # 取相反数
    ideal_complex = ideal_real + 1j * ideal_imag
    ideal_abs = np.abs(ideal_complex)
    ideal_s11_dB = 20 * np.log10(ideal_abs)
    
    ideal_data = {
        'real': ideal_real,
        'imag': ideal_imag,
        's11_dB': ideal_s11_dB
    }
    
    print('Loading training data...')
    training_data, freq_data = load_training_data()
    print(f'Total training samples: {len(training_data)}')
    
    print('Finding top 10 most similar training data...')
    top_10_data = find_similar_data(ideal_data, training_data)
    
    print('Generating comparison plots...')
    plot_s11_comparison(ideal_data, top_10_data, freq_data, result_dir)
    plot_re_im_comparison(ideal_data, top_10_data, freq_data, result_dir)
    
    print('All tasks completed successfully!')

if __name__ == '__main__':
    main()
