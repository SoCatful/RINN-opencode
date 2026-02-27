"""
数据加载和预处理工具函数
"""
import numpy as np
import csv
import re

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
    
    # 计算S11 dB值: 20 * log10(sqrt(real^2 + imag^2))
    s11_magnitude = np.sqrt(real_data**2 + imag_data**2 + 1e-12)  # 加小值避免除零
    y_data = 20 * np.log10(s11_magnitude)
    print(f'  Y数据形状: {y_data.shape} (101维dB值)')
    
    # 打印dB值范围
    print(f'  dB值范围: {y_data.min():.2f} dB - {y_data.max():.2f} dB')
    
    return x_features, y_data, freq_data

def normalize_data(train_x, val_x, train_y, val_y, method='robust'):
    """数据标准化"""
    if method == 'standard':
        # 使用均值-标准差归一化
        x_mean = train_x.mean(axis=0)
        x_std = train_x.std(axis=0)
        train_x_normalized = (train_x - x_mean) / (x_std + 1e-8)
        val_x_normalized = (val_x - x_mean) / (x_std + 1e-8)
        
        y_mean = train_y.mean(axis=0)
        y_std = train_y.std(axis=0)
        
        train_y_normalized = (train_y - y_mean) / (y_std + 1e-8)
        val_y_normalized = (val_y - y_mean) / (y_std + 1e-8)
    else:  # 'robust'
        # 使用四分位数归一化（中位数和四分位距）
        x_median = np.median(train_x, axis=0)
        x_q1 = np.percentile(train_x, 25, axis=0)
        x_q3 = np.percentile(train_x, 75, axis=0)
        x_iqr = x_q3 - x_q1
        train_x_normalized = (train_x - x_median) / (x_iqr + 1e-8)
        val_x_normalized = (val_x - x_median) / (x_iqr + 1e-8)
        
        # 对Y数据进行更鲁棒的处理，先裁剪异常值
        y_median = np.median(train_y, axis=0)
        y_q1 = np.percentile(train_y, 25, axis=0)
        y_q3 = np.percentile(train_y, 75, axis=0)
        y_iqr = y_q3 - y_q1
        
        # 裁剪异常值到[Q1-3*IQR, Q3+3*IQR]范围内
        y_lower_bound = y_q1 - 3 * y_iqr
        y_upper_bound = y_q3 + 3 * y_iqr
        
        # 对训练数据进行裁剪
        train_y_clipped = np.clip(train_y, y_lower_bound, y_upper_bound)
        
        # 使用裁剪后的数据重新计算归一化参数
        y_median_clipped = np.median(train_y_clipped, axis=0)
        y_q1_clipped = np.percentile(train_y_clipped, 25, axis=0)
        y_q3_clipped = np.percentile(train_y_clipped, 75, axis=0)
        y_iqr_clipped = y_q3_clipped - y_q1_clipped
        
        # 归一化
        train_y_normalized = (train_y_clipped - y_median_clipped) / (y_iqr_clipped + 1e-8)
        # 对验证数据也进行同样的裁剪和归一化
        val_y_clipped = np.clip(val_y, y_lower_bound, y_upper_bound)
        val_y_normalized = (val_y_clipped - y_median_clipped) / (y_iqr_clipped + 1e-8)
        
        # 保存鲁棒归一化参数
        x_mean, x_std = x_median, x_iqr
        y_mean, y_std = y_median_clipped, y_iqr_clipped
    
    return train_x_normalized, val_x_normalized, train_y_normalized, val_y_normalized, x_mean, x_std, y_mean, y_std