"""
预测和多解生成工具函数
"""
import torch
import numpy as np
from R_INN_model.loss_methods import nmse_loss

def predict_y(model, x_test, real_y, padding_dim, y_mean, y_std, device):
    """固定x预测y"""
    # 左侧输入：X + 零填充
    left_test_input = np.concatenate((x_test, np.zeros((1, padding_dim), dtype=np.float32)), axis=1)
    left_test_input = torch.FloatTensor(left_test_input).to(device)

    # 使用模型进行正向预测
    model.eval()
    with torch.no_grad():
        predicted_right, _, _ = model(left_test_input, return_intermediate=True)
        
        # 从predicted_right中提取Y'
        predicted_y_normalized = predicted_right[:, :y_mean.shape[0]]  # 根据y_mean的形状确定维度
        
        # 反标准化得到预测的y
        predicted_y = predicted_y_normalized.cpu().numpy() * y_std + y_mean

    # 计算NMSE
    mse = np.mean((predicted_y[0] - real_y[0]) ** 2)
    variance = np.var(real_y[0])
    nmse_value = mse / (variance + 1e-8)

    return predicted_y, nmse_value

def backward_predict_x(model, y_test, real_y, real_x, z_dim, x_dim, y_dim, padding_dim, x_mean, x_std, y_mean, y_std, num_z_candidates, device):
    """固定y回推x（多Z采样+NMSE选优）"""
    # 采样多个Z
    z_candidates = np.random.randn(num_z_candidates, z_dim).astype(np.float32)
    
    # 批量回推X
    y_test_repeated = np.repeat(y_test, num_z_candidates, axis=0)
    right_test_inputs = np.concatenate((y_test_repeated, z_candidates), axis=1)
    right_test_inputs = torch.FloatTensor(right_test_inputs).to(device)
    
    with torch.no_grad():
        reconstructed_lefts, _ = model.inverse(right_test_inputs)
        reconstructed_xs_normalized = reconstructed_lefts[:, :x_dim]
        reconstructed_xs = reconstructed_xs_normalized.cpu().numpy() * x_std + x_mean
    
    # 对每个回推的X进行正向预测，计算NMSE
    nmse_errors = []
    
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
        mse = np.mean((predicted_y[0] - real_y[0]) ** 2)
        variance = np.var(real_y[0])
        nmse = mse / (variance + 1e-8)
        nmse_errors.append(nmse)
    
    # 选择NMSE最小的回推结果
    best_idx = np.argmin(nmse_errors)
    best_reconstructed_x = reconstructed_xs[best_idx]
    best_nmse = nmse_errors[best_idx]
    
    # 计算相对误差
    relative_errors = np.abs((best_reconstructed_x - real_x[0]) / (real_x[0] + 1e-8))
    
    return best_reconstructed_x, best_nmse, relative_errors

def generate_multiple_solutions(model, y_test, z_dim, x_dim, y_dim, padding_dim, x_mean, x_std, y_mean, y_std, num_samples, z_scale, device):
    """多解生成功能"""
    # 对于同一个y，生成多个z样本
    z_samples = np.random.randn(num_samples, z_dim).astype(np.float32) * z_scale
    
    # 为每个z样本创建右侧输入
    y_test_repeated = np.repeat(y_test, num_samples, axis=0)
    right_test_inputs = np.concatenate((y_test_repeated, z_samples), axis=1)
    right_test_inputs = torch.FloatTensor(right_test_inputs).to(device)
    
    # 使用模型进行批量反向预测
    with torch.no_grad():
        reconstructed_lefts, _ = model.inverse(right_test_inputs)
        
        # 从reconstructed_lefts中提取X'
        reconstructed_xs_normalized = reconstructed_lefts[:, :x_dim]
        
        # 反标准化得到回推的x样本
        reconstructed_xs = reconstructed_xs_normalized.cpu().numpy() * x_std + x_mean
    
    # 验证X的多样性
    diversity = np.std(reconstructed_xs, axis=0)
    
    # 对生成的X进行后处理，确保在合理物理范围内
    # 这里可以根据实际情况添加约束
    
    # 对每个生成的X进行正向预测，计算NMSE
    generated_xs_normalized = (reconstructed_xs - x_mean) / (x_std + 1e-8)
    left_predict_inputs = np.concatenate((generated_xs_normalized, np.zeros((num_samples, padding_dim), dtype=np.float32)), axis=1)
    left_predict_inputs = torch.FloatTensor(left_predict_inputs).to(device)
    
    with torch.no_grad():
        predicted_rights, _ = model(left_predict_inputs)
        predicted_y_normalized = predicted_rights[:, :y_dim]
        predicted_y = predicted_y_normalized.cpu().numpy() * y_std + y_mean
    
    # 计算每个预测的Y与原始Y的NMSE
    y_test_original = y_test * y_std + y_mean
    nmse_errors = []
    y_variance = np.var(y_test_original[0])  # 计算原始Y的方差
    
    for i in range(num_samples):
        mse = np.mean((predicted_y[i] - y_test_original[0]) ** 2)
        nmse = mse / (y_variance + 1e-8)
        nmse_errors.append(nmse)
    
    # 获取前5个最小NMSE的索引
    top_indices = np.argsort(nmse_errors)[:5]
    
    return reconstructed_xs, predicted_y, nmse_errors, top_indices