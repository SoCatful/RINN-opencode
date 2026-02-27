"""
训练和评估工具函数
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from R_INN_model.loss_methods import mmd_loss, nmse_loss, weighted_nmse_loss

def calculate_loss(model, left_input, right_input, x_dim, y_dim, weight_y, weight_x, weight_z, device):
    """计算RINN模型损失
    核心：
    1. 正向预测的Y'和真实Y的加权NMSE损失
    2. 正向预测的Z'和标准正态分布的MMD差异
    3. 随机生成Z，拼接Y回推X，对X使用MMD损失
    """
    # 正向映射：left_input → predicted_right
    predicted_right, log_det_forward, _ = model(left_input, return_intermediate=True)
    
    # 从predicted_right中提取Y'和Z'
    predicted_y = predicted_right[:, :y_dim]
    predicted_z = predicted_right[:, y_dim:]
    
    # 从right_input中提取真实Y
    real_y = right_input[:, :y_dim]
    
    # 1. Y预测损失：加权NMSE
    y_loss = weighted_nmse_loss(real_y, predicted_y)
    
    # 2. Z分布约束：predicted_z与标准高斯分布的MMD差异
    z_target = torch.randn_like(predicted_z).to(device)  # 标准高斯分布
    z_loss = mmd_loss(predicted_z, z_target)
    
    # 3. 随机生成Z，拼接Y回推X，计算X的MMD损失
    # 随机生成一批Z
    batch_size = left_input.size(0)
    random_z = torch.randn(batch_size, x_dim).to(device)
    
    # 拼接Y和随机生成的Z
    right_input_random_z = torch.cat((real_y, random_z), dim=1)
    
    # 反向映射：Y+Z → X+0填充
    reconstructed_left, _ = model.inverse(right_input_random_z)
    
    # 从reconstructed_left中提取X'
    reconstructed_x = reconstructed_left[:, :x_dim]
    
    # 从left_input中提取真实X
    real_x = left_input[:, :x_dim]
    
    # 使用MMD损失计算x损失
    x_loss = mmd_loss(real_x, reconstructed_x)
    
    # 总损失：组合三个损失项
    total_loss = weight_y * y_loss + weight_x * x_loss + weight_z * z_loss
    
    return {
        "total_loss": total_loss,
        "y_loss": y_loss,
        "x_loss": x_loss,
        "z_loss": z_loss
    }

def train_model(model, left_train, left_val, train_y_normalized, val_y_normalized, 
                batch_size, optimizer, scheduler, num_epochs, 
                x_dim, y_dim, z_dim, weight_y, weight_x, weight_z, device, 
                patience=30, grad_accum_steps=1, clip_value=0.5):
    """训练模型 - 每个epoch重新采样Z"""
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 训练历史
    train_losses = {'total': [], 'y_loss': [], 'x_loss': [], 'z_loss': []}
    val_losses = {'total': [], 'y_loss': [], 'x_loss': [], 'z_loss': []}
    
    for epoch in range(num_epochs):
        # ========== 每个epoch重新采样Z ==========
        # 重新采样训练集和验证集的Z
        train_z = np.random.randn(len(train_y_normalized), z_dim).astype(np.float32)
        val_z = np.random.randn(len(val_y_normalized), z_dim).astype(np.float32)
        
        # 创建右侧输入：Y + Z
        right_train_input = np.concatenate((train_y_normalized, train_z), axis=1)
        right_val_input = np.concatenate((val_y_normalized, val_z), axis=1)
        
        # 转换为torch张量
        right_train = torch.FloatTensor(right_train_input)
        right_val = torch.FloatTensor(right_val_input)
        
        # 创建DataLoader
        train_dataset = TensorDataset(left_train, right_train)
        val_dataset = TensorDataset(left_val, right_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练阶段
        model.train()
        epoch_train_losses = {"total_loss": 0.0, "y_loss": 0.0, "x_loss": 0.0, "z_loss": 0.0}
        
        optimizer.zero_grad()  # 初始清零梯度
        
        for i, batch in enumerate(train_loader):
            left_batch = batch[0].to(device)
            right_batch = batch[1].to(device)
            
            # 计算损失
            losses = calculate_loss(model, left_batch, right_batch, x_dim, y_dim, 
                                   weight_y, weight_x, weight_z, device)
            
            # 梯度更新（使用梯度累积）
            scaled_loss = losses["total_loss"] / grad_accum_steps
            scaled_loss.backward()
            
            # 损失累加
            for key in epoch_train_losses:
                epoch_train_losses[key] += losses[key].item()
            
            # 每grad_accum_steps个batch进行一次梯度裁剪和参数更新
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
                optimizer.step()
                optimizer.zero_grad()
        
        # 计算平均损失
        num_batches = len(train_loader)
        for key in epoch_train_losses:
            epoch_train_losses[key] /= num_batches
        
        # 记录训练历史
        train_losses['total'].append(epoch_train_losses['total_loss'])
        train_losses['y_loss'].append(epoch_train_losses['y_loss'])
        train_losses['x_loss'].append(epoch_train_losses['x_loss'])
        train_losses['z_loss'].append(epoch_train_losses['z_loss'])
        
        # 验证阶段
        model.eval()
        epoch_val_losses = {"total_loss": 0.0, "y_loss": 0.0, "x_loss": 0.0, "z_loss": 0.0}
        
        with torch.no_grad():
            for batch in val_loader:
                left_batch = batch[0].to(device)
                right_batch = batch[1].to(device)
                
                # 计算损失
                losses = calculate_loss(model, left_batch, right_batch, x_dim, y_dim, 
                                       weight_y, weight_x, weight_z, device)
                
                for key in epoch_val_losses:
                    epoch_val_losses[key] += losses[key].item()
        
        # 计算验证集平均损失
        for key in epoch_val_losses:
            epoch_val_losses[key] /= len(val_loader)
        
        # 记录验证历史
        val_losses['total'].append(epoch_val_losses['total_loss'])
        val_losses['y_loss'].append(epoch_val_losses['y_loss'])
        val_losses['x_loss'].append(epoch_val_losses['x_loss'])
        val_losses['z_loss'].append(epoch_val_losses['z_loss'])
        
        # 更新学习率调度器（基于验证损失）
        scheduler.step(epoch_val_losses['total_loss'])
        
        # 早停检查
        if epoch_val_losses['total_loss'] < best_val_loss:
            best_val_loss = epoch_val_losses['total_loss']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停触发! 验证损失连续{patience}个epoch没有改善')
                break
        
        # 每10个epoch打印一次
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {epoch_train_losses['total_loss']:.6f}, "
                  f"Y Loss: {epoch_train_losses['y_loss']:.6f}, "
                  f"X Loss: {epoch_train_losses['x_loss']:.6f}, "
                  f"Z Loss: {epoch_train_losses['z_loss']:.6f}")
            print(f"          Val Loss: {epoch_val_losses['total_loss']:.6f}, "
                  f"Y Loss: {epoch_val_losses['y_loss']:.6f}, "
                  f"X Loss: {epoch_val_losses['x_loss']:.6f}, "
                  f"Z Loss: {epoch_val_losses['z_loss']:.6f}")
    
    return best_val_loss, train_losses, val_losses