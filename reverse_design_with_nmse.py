import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import re

# 设置设备
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f'Using device: {device}')

# 提取几何参数信息
def extract_geometry_params(col_name):
    """从列名中提取几何参数H1, H2, H3, H_C1, H_C2"""
    col_name = col_name.replace('\n', '').replace('"', '')
    
    h1_match = re.search(r"H1='([\d.]+)mm'", col_name)
    h2_match = re.search(r"H2='([\d.]+)mm'", col_name)
    h3_match = re.search(r"H3='([\d.]+)mm'", col_name)
    hc1_match = re.search(r"H_C1='([\d.]+)mm'", col_name)
    hc2_match = re.search(r"H_C2='([\d.]+)mm'", col_name)
    
    # 检查是否是实部还是虚部
    is_real = 're(S(1,1))' in col_name
    is_imag = 'im(S(1,1))' in col_name
    
    if all([h1_match, h2_match, h3_match, hc1_match, hc2_match]):
        return {
            'params': [
                float(h1_match.group(1)),
                float(h2_match.group(1)),
                float(h3_match.group(1)),
                float(hc1_match.group(1)),
                float(hc2_match.group(1))
            ],
            'type': 'real' if is_real else 'imaginary'
        }
    return None

def load_data_from_csv(data_path):
    """从CSV文件加载数据"""
    print(f'正在加载数据文件: {data_path}')
    
    # 读取表头获取几何参数
    with open(data_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    # 提取所有几何参数样本和数据列
    geometry_dict = {}
    
    for i, col in enumerate(header[1:]):  # 跳过第一列频率
        params = extract_geometry_params(col)
        if params:
            geo_key = tuple(params['params'])
            if geo_key not in geometry_dict:
                geometry_dict[geo_key] = {'real': None, 'imag': None}
            if params['type'] == 'real':
                geometry_dict[geo_key]['real'] = i+1
            else:
                geometry_dict[geo_key]['imag'] = i+1
    
    # 只保留同时有实部和虚部的样本
    valid_samples = []
    real_columns = []
    imag_columns = []
    
    for geo_key, cols in geometry_dict.items():
        if cols['real'] is not None and cols['imag'] is not None:
            valid_samples.append(list(geo_key))
            real_columns.append(cols['real'])
            imag_columns.append(cols['imag'])
    
    x_features = np.array(valid_samples, dtype=np.float32)
    print(f'  X特征形状: {x_features.shape}')
    print(f'  X特征示例 (第一个样本): {x_features[0]}')
    
    # 读取S11数据
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    freq_data = data[:, 0]  # 频率数据
    print(f'  频率点数: {len(freq_data)}')
    print(f'  频率范围: {freq_data[0]} GHz - {freq_data[-1]} GHz')
    
    # 提取实部和虚部数据列
    real_data = data[:, real_columns]  # 实部数据
    imag_data = data[:, imag_columns]  # 虚部数据
    
    # 转置为(样本数, 频率点数)
    real_data = real_data.T
    imag_data = imag_data.T
    
    # 合并实部和虚部为一个202维的输出（101维实部 + 101维虚部）
    y_data = np.concatenate((real_data, imag_data), axis=1)
    print(f'  Y数据形状: {y_data.shape} (101维实部 + 101维虚部)')
    
    return x_features, y_data, freq_data

# 主函数
def main():
    # 模型路径
    model_path = '/Users/tianzhuohang/Desktop/科研/R_INN_opencode/model_checkpoints_rinn/rinn_correct_structure_20260215_004551/best_model.pth'
    # 数据路径
    data_path = '/Users/tianzhuohang/Desktop/科研/R_INN_opencode/data/S Parameter Plot1perfect.csv'
    # 结果保存路径
    result_dir = 'result_reverse_design'
    os.makedirs(result_dir, exist_ok=True)
    
    print('\n=== 加载模型 ===')
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 恢复模型参数
    x_dim = checkpoint['x_dim']
    y_dim = checkpoint['y_dim']
    z_dim = checkpoint['z_dim']
    left_input_dim = checkpoint['left_input_dim']
    
    # 恢复归一化参数
    x_mean = checkpoint['x_mean']
    x_std = checkpoint['x_std']
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']
    
    # 创建模型
    from R_INN_model.rinn_model import RINNModel
    # 从训练配置中恢复模型参数
    model = RINNModel(
        input_dim=left_input_dim,
        hidden_dim=64,  # 从训练配置
        num_blocks=8,    # 从训练配置
        num_stages=3,    # 从训练配置
        num_cycles_per_stage=2,  # 从训练配置
        ratio_toZ_after_flowstage=0.194,  # 从训练配置
        ratio_x1_x2_inAffine=0.12  # 从训练配置
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('模型加载完成!')
    
    print('\n=== 加载数据 ===')
    # 加载数据
    x_features, y_data, freq_data = load_data_from_csv(data_path)
    
    # 使用第一个样本
    y_test = y_data[0:1]
    real_x = x_features[0:1]
    
    # 归一化Y数据
    y_test_normalized = (y_test - y_mean) / (y_std + 1e-8)
    
    print('\n=== 执行反向X设计 ===')
    print(f'采样500个Z值，计算每个Z对应的X，然后找到正推NMSE最好的X')
    
    # 采样500个Z
    num_z_candidates = 500
    z_candidates = np.random.randn(num_z_candidates, z_dim).astype(np.float32)
    
    # 批量回推X
    y_test_repeated = np.repeat(y_test_normalized, num_z_candidates, axis=0)
    right_test_inputs = np.concatenate((y_test_repeated, z_candidates), axis=1)
    right_test_inputs = torch.FloatTensor(right_test_inputs).to(device)
    
    with torch.no_grad():
        reconstructed_lefts, _ = model.inverse(right_test_inputs)
        reconstructed_xs_normalized = reconstructed_lefts[:, :x_dim]
        reconstructed_xs = reconstructed_xs_normalized.cpu().numpy() * x_std + x_mean
    
    print(f'完成{num_z_candidates}个X的反向设计')
    
    # 对每个回推的X进行正向预测，计算NMSE
    print('\n=== 计算每个X的正推NMSE ===')
    nmse_errors = []
    predicted_ys = []
    
    # 计算左侧输入维度
    padding_dim = y_dim
    
    for j in range(num_z_candidates):
        # 准备左侧输入：回推的X + 零填充
        x_norm = (reconstructed_xs[j:j+1] - x_mean) / (x_std + 1e-8)
        left_input = np.concatenate((x_norm, np.zeros((1, padding_dim), dtype=np.float32)), axis=1)
        left_input = torch.FloatTensor(left_input).to(device)
        
        # 正向预测Y
        with torch.no_grad():
            predicted_right, _, _ = model(left_input, return_intermediate=True)
            predicted_y_normalized = predicted_right[:, :y_dim]
            predicted_y = predicted_y_normalized.cpu().numpy() * y_std + y_mean
        
        # 计算NMSE
        mse = np.mean((predicted_y[0] - y_test[0]) ** 2)
        variance = np.var(y_test[0])
        nmse = mse / (variance + 1e-8)
        nmse_errors.append(nmse)
        predicted_ys.append(predicted_y[0])
        
        # 每100个打印一次进度
        if (j + 1) % 100 == 0:
            print(f'  处理 {j + 1}/{num_z_candidates} 个X，当前最小NMSE: {min(nmse_errors):.6f}')
    
    # 找到NMSE最小的X
    best_idx = np.argmin(nmse_errors)
    best_x = reconstructed_xs[best_idx]
    best_nmse = nmse_errors[best_idx]
    best_predicted_y = predicted_ys[best_idx]
    
    print('\n=== 最佳结果 ===')
    print(f'最佳NMSE: {best_nmse:.6f}')
    print(f'真实X: {real_x[0]}')
    print(f'最佳X: {best_x}')
    
    # 计算相对误差
    relative_errors = np.abs((best_x - real_x[0]) / (real_x[0] + 1e-8))
    print(f'相对误差: {relative_errors}')
    print(f'平均相对误差: {np.mean(relative_errors):.6f}')
    
    # 可视化结果
    print('\n=== 可视化结果 ===')
    
    # 绘制X参数对比
    plt.figure(figsize=(15, 6))
    params = ['H1', 'H2', 'H3', 'H_C1', 'H_C2']
    
    for i, (param, real_val, pred_val) in enumerate(zip(params, real_x[0], best_x)):
        plt.subplot(1, 5, i+1)
        plt.bar(['Real', 'Predicted'], [real_val, pred_val], color=['blue', 'red'])
        plt.title(param)
        plt.ylabel('Value (mm)')
        plt.ylim(min(real_val, pred_val) * 0.95, max(real_val, pred_val) * 1.05)
    
    plt.suptitle(f'Geometry Parameters Comparison (Best NMSE: {best_nmse:.6f})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(result_dir, 'best_x_comparison.png'), dpi=300)
    plt.close()
    
    # 绘制S11对比
    plt.figure(figsize=(12, 8))
    
    # 实部
    plt.subplot(2, 1, 1)
    plt.plot(freq_data, y_test[0, :101], 'blue', linewidth=2, label='Original Re(S11)')
    plt.plot(freq_data, best_predicted_y[:101], 'red', linestyle='--', linewidth=2, label='Predicted Re(S11)')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Re(S11)')
    plt.title('Real Part Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 虚部
    plt.subplot(2, 1, 2)
    plt.plot(freq_data, y_test[0, 101:], 'blue', linewidth=2, label='Original Im(S11)')
    plt.plot(freq_data, best_predicted_y[101:], 'red', linestyle='--', linewidth=2, label='Predicted Im(S11)')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Im(S11)')
    plt.title('Imaginary Part Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'best_s11_comparison.png'), dpi=300)
    plt.close()
    
    # 绘制NMSE分布
    plt.figure(figsize=(10, 6))
    plt.hist(nmse_errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(best_nmse, color='red', linestyle='--', linewidth=2, label=f'Best NMSE: {best_nmse:.6f}')
    plt.xlabel('NMSE')
    plt.ylabel('Count')
    plt.title('NMSE Distribution of 500 Z Candidates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'nmse_distribution.png'), dpi=300)
    plt.close()
    
    print('\n=== 结果保存 ===')
    print(f'结果保存在: {result_dir}')
    print('文件列表:')
    print(f'  - best_x_comparison.png: 最佳X与真实X的对比')
    print(f'  - best_s11_comparison.png: 最佳X对应的S11与真实S11的对比')
    print(f'  - nmse_distribution.png: 500个Z候选的NMSE分布')
    
    print('\n=== 完成 ===')

if __name__ == '__main__':
    main()
