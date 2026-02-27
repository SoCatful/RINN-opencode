import os
import json
import numpy as np
import torch


def ensure_directory(directory):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    os.makedirs(directory, exist_ok=True)


def load_param_ranges(param_ranges_path):
    """
    加载几何参数范围
    
    Args:
        param_ranges_path: 参数范围文件路径
    
    Returns:
        dict: 参数范围字典
    """
    param_ranges = {}
    if param_ranges_path and os.path.exists(param_ranges_path):
        with open(param_ranges_path, 'r') as f:
            param_ranges = json.load(f)
    return param_ranges


def calculate_nmse(real_values, predicted_values):
    """
    计算归一化均方误差(NMSE)
    
    Args:
        real_values: 真实值，形状为(N, D)或(1, D)
        predicted_values: 预测值，形状为(N, D)或(1, D)
    
    Returns:
        float: NMSE值
    """
    # 确保输入是numpy数组
    if isinstance(real_values, torch.Tensor):
        real_values = real_values.cpu().numpy()
    if isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.cpu().numpy()
    
    # 计算MSE
    mse = np.mean((predicted_values - real_values) ** 2)
    
    # 计算方差
    variance = np.var(real_values)
    
    # 计算NMSE
    nmse = mse / (variance + 1e-8)
    
    return nmse


def get_top_solutions(nmse_errors, top_k=5):
    """
    获取NMSE最小的前k个解的索引
    
    Args:
        nmse_errors: NMSE误差数组，形状为(N,)
        top_k: 返回前k个解
    
    Returns:
        np.ndarray: 前k个解的索引
    """
    return np.argsort(nmse_errors)[:top_k]


def clip_values(values, min_values, max_values):
    """
    将值裁剪到指定范围内
    
    Args:
        values: 要裁剪的值，形状为(N, D)
        min_values: 最小值，形状为(D,)
        max_values: 最大值，形状为(D,)
    
    Returns:
        np.ndarray: 裁剪后的值
    """
    return np.clip(values, min_values, max_values)


def get_parameter_ranges(data):
    """
    从数据中获取参数范围
    
    Args:
        data: 数据数组，形状为(N, D)
    
    Returns:
        dict: 每个参数的最小值和最大值
    """
    ranges = {}
    for i in range(data.shape[1]):
        ranges[f'param_{i+1}'] = {
            'min': float(data[:, i].min()),
            'max': float(data[:, i].max())
        }
    return ranges


def save_parameter_ranges(ranges, save_path):
    """
    保存参数范围到文件
    
    Args:
        ranges: 参数范围字典
        save_path: 保存路径
    """
    ensure_directory(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(ranges, f, indent=2)


def load_data_from_npy(file_path):
    """
    从npy文件加载数据
    
    Args:
        file_path: 文件路径
    
    Returns:
        np.ndarray: 加载的数据
    """
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"文件不存在: {file_path}")


def save_data_to_npy(data, file_path):
    """
    将数据保存到npy文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    ensure_directory(os.path.dirname(file_path))
    np.save(file_path, data)
