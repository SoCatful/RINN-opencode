#!/usr/bin/env python3
"""
测试可视化模块
"""

import os
import sys
import numpy as np

# 添加当前目录到Python模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.forward_prediction_visualizer import visualize_forward_prediction
from visualization.backward_prediction_visualizer import visualize_backward_prediction, visualize_multi_solution_analysis
from visualization.utils import save_data_to_npy, ensure_directory


def generate_test_data():
    """
    生成测试数据
    """
    # 生成频率数据 (101个点，10.5-11.5 GHz)
    freq_data = np.linspace(10.5, 11.5, 101)
    
    # 生成真实的y值 (1, 202)，前101维为实部，后101维为虚部
    real_y_re = np.random.randn(1, 101) * 0.1 - 0.5  # 实部
    real_y_im = np.random.randn(1, 101) * 0.1       # 虚部
    real_y = np.concatenate((real_y_re, real_y_im), axis=1)
    
    # 生成预测的y值，在真实值基础上添加一些噪声
    predicted_y_re = real_y_re + np.random.randn(1, 101) * 0.05
    predicted_y_im = real_y_im + np.random.randn(1, 101) * 0.05
    predicted_y = np.concatenate((predicted_y_re, predicted_y_im), axis=1)
    
    # 生成真实的x值 (1, 5)
    real_x = np.array([[1.0, 2.0, 3.0, 0.5, 0.5]])
    
    # 生成重建的x值，在真实值基础上添加一些噪声
    reconstructed_x = real_x + np.random.randn(1, 5) * 0.1
    
    # 生成多解数据
    num_samples = 100
    reconstructed_xs = real_x + np.random.randn(num_samples, 5) * 0.2
    predicted_ys = np.tile(real_y, (num_samples, 1)) + np.random.randn(num_samples, 202) * 0.1
    nmse_errors = np.random.rand(num_samples) * 0.5
    
    return {
        'freq_data': freq_data,
        'real_y': real_y,
        'predicted_y': predicted_y,
        'real_x': real_x,
        'reconstructed_x': reconstructed_x,
        'reconstructed_xs': reconstructed_xs,
        'predicted_ys': predicted_ys,
        'nmse_errors': nmse_errors
    }


def test_forward_prediction_visualization():
    """
    测试正向预测可视化
    """
    print("测试正向预测可视化...")
    
    # 生成测试数据
    test_data = generate_test_data()
    
    # 创建测试数据目录
    test_data_dir = './test_data'
    ensure_directory(test_data_dir)
    
    # 保存测试数据
    save_data_to_npy(test_data['freq_data'], os.path.join(test_data_dir, 'freq_data.npy'))
    save_data_to_npy(test_data['real_y'], os.path.join(test_data_dir, 'real_y.npy'))
    save_data_to_npy(test_data['predicted_y'], os.path.join(test_data_dir, 'predicted_y.npy'))
    
    # 创建保存目录
    save_dir = './test_visualization_results/forward'
    ensure_directory(save_dir)
    
    # 调用可视化函数
    save_path = visualize_forward_prediction(
        test_data['real_y'],
        test_data['predicted_y'],
        test_data['freq_data'],
        1,
        save_dir
    )
    
    print(f"正向预测可视化结果已保存到: {save_path}")
    print("正向预测可视化测试完成！")


def test_backward_prediction_visualization():
    """
    测试反向逆设计可视化
    """
    print("\n测试反向逆设计可视化...")
    
    # 生成测试数据
    test_data = generate_test_data()
    
    # 创建保存目录
    save_dir = './test_visualization_results/backward'
    ensure_directory(save_dir)
    
    # 调用可视化函数
    save_path = visualize_backward_prediction(
        test_data['real_x'],
        test_data['reconstructed_x'],
        test_data['real_y'],
        test_data['predicted_y'],
        test_data['freq_data'],
        1,
        save_dir
    )
    
    print(f"反向逆设计可视化结果已保存到: {save_path}")
    print("反向逆设计可视化测试完成！")


def test_multi_solution_visualization():
    """
    测试多解生成可视化
    """
    print("\n测试多解生成可视化...")
    
    # 生成测试数据
    test_data = generate_test_data()
    
    # 创建保存目录
    save_dir = './test_visualization_results/multi_solution'
    ensure_directory(save_dir)
    
    # 调用可视化函数
    save_path = visualize_multi_solution_analysis(
        test_data['real_x'],
        test_data['reconstructed_xs'],
        test_data['predicted_ys'],
        test_data['real_y'],
        test_data['freq_data'],
        test_data['nmse_errors'],
        save_dir
    )
    
    print(f"多解生成可视化结果已保存到: {save_path}")
    print("多解生成可视化测试完成！")


def main():
    """
    主测试函数
    """
    print("开始测试可视化模块...")
    
    # 测试正向预测可视化
    test_forward_prediction_visualization()
    
    # 测试反向逆设计可视化
    test_backward_prediction_visualization()
    
    # 测试多解生成可视化
    test_multi_solution_visualization()
    
    print("\n所有测试完成！")


if __name__ == '__main__':
    main()
