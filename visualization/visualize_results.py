#!/usr/bin/env python3
"""
可视化结果入口脚本

支持两种类型的可视化：
1. 正向预测可视化（固定x预测y）
2. 反向逆设计可视化（固定y回推x）
"""

import argparse
import os
import numpy as np
from visualization.forward_prediction_visualizer import visualize_forward_prediction, visualize_multiple_forward_predictions
from visualization.backward_prediction_visualizer import visualize_backward_prediction, visualize_multi_solution_analysis, visualize_x_distribution
from visualization.utils import ensure_directory, load_data_from_npy


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='RINN模型结果可视化脚本')
    
    # 子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='可视化命令')
    
    # 正向预测可视化命令
    forward_parser = subparsers.add_parser('forward', help='正向预测结果可视化')
    forward_parser.add_argument('--real-y', type=str, required=True, help='真实y值的npy文件路径')
    forward_parser.add_argument('--predicted-y', type=str, required=True, help='预测y值的npy文件路径')
    forward_parser.add_argument('--freq-data', type=str, required=True, help='频率数据的npy文件路径')
    forward_parser.add_argument('--sample-index', type=int, default=1, help='样本索引')
    forward_parser.add_argument('--save-dir', type=str, default='./visualization_results', help='保存图表的目录')
    forward_parser.add_argument('--multiple', action='store_true', help='可视化多个样本')
    
    # 反向逆设计可视化命令
    backward_parser = subparsers.add_parser('backward', help='反向逆设计结果可视化')
    backward_parser.add_argument('--real-x', type=str, required=True, help='真实x值的npy文件路径')
    backward_parser.add_argument('--reconstructed-x', type=str, required=True, help='重建x值的npy文件路径')
    backward_parser.add_argument('--real-y', type=str, required=True, help='真实y值的npy文件路径')
    backward_parser.add_argument('--predicted-y', type=str, required=True, help='预测y值的npy文件路径')
    backward_parser.add_argument('--freq-data', type=str, required=True, help='频率数据的npy文件路径')
    backward_parser.add_argument('--sample-index', type=int, default=1, help='样本索引')
    backward_parser.add_argument('--save-dir', type=str, default='./visualization_results', help='保存图表的目录')
    backward_parser.add_argument('--param-ranges', type=str, help='几何参数范围文件路径')
    
    # 多解生成可视化命令
    multi_solution_parser = subparsers.add_parser('multi-solution', help='多解生成结果可视化')
    multi_solution_parser.add_argument('--real-x', type=str, required=True, help='真实x值的npy文件路径')
    multi_solution_parser.add_argument('--reconstructed-xs', type=str, required=True, help='生成的多个x值的npy文件路径')
    multi_solution_parser.add_argument('--predicted-ys', type=str, required=True, help='预测的多个y值的npy文件路径')
    multi_solution_parser.add_argument('--y-original', type=str, required=True, help='原始y值的npy文件路径')
    multi_solution_parser.add_argument('--freq-data', type=str, required=True, help='频率数据的npy文件路径')
    multi_solution_parser.add_argument('--nmse-errors', type=str, required=True, help='NMSE误差的npy文件路径')
    multi_solution_parser.add_argument('--save-dir', type=str, default='./visualization_results', help='保存图表的目录')
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 确保保存目录存在
    ensure_directory(args.save_dir)
    
    if args.command == 'forward':
        # 加载数据
        real_y = load_data_from_npy(args.real_y)
        predicted_y = load_data_from_npy(args.predicted_y)
        freq_data = load_data_from_npy(args.freq_data)
        
        if args.multiple:
            # 可视化多个正向预测结果
            saved_paths = visualize_multiple_forward_predictions(
                real_y, predicted_y, freq_data, args.save_dir
            )
            print(f"\n已保存{len(saved_paths)}个正向预测可视化结果")
        else:
            # 可视化单个正向预测结果
            save_path = visualize_forward_prediction(
                real_y, predicted_y, freq_data, args.sample_index, args.save_dir
            )
            print(f"\n正向预测可视化结果已保存到: {save_path}")
    
    elif args.command == 'backward':
        # 加载数据
        real_x = load_data_from_npy(args.real_x)
        reconstructed_x = load_data_from_npy(args.reconstructed_x)
        real_y = load_data_from_npy(args.real_y)
        predicted_y = load_data_from_npy(args.predicted_y)
        freq_data = load_data_from_npy(args.freq_data)
        
        # 可视化反向预测结果
        save_path = visualize_backward_prediction(
            real_x, reconstructed_x, real_y, predicted_y, freq_data, 
            args.sample_index, args.save_dir, args.param_ranges
        )
        print(f"\n反向逆设计可视化结果已保存到: {save_path}")
    
    elif args.command == 'multi-solution':
        # 加载数据
        real_x = load_data_from_npy(args.real_x)
        reconstructed_xs = load_data_from_npy(args.reconstructed_xs)
        predicted_ys = load_data_from_npy(args.predicted_ys)
        y_original = load_data_from_npy(args.y_original)
        freq_data = load_data_from_npy(args.freq_data)
        nmse_errors = load_data_from_npy(args.nmse_errors)
        
        # 可视化X分布
        x_dist_path = visualize_x_distribution(
            reconstructed_xs, real_x, args.save_dir
        )
        print(f"X分布可视化结果已保存到: {x_dist_path}")
        
        # 可视化多解分析
        multi_sol_path = visualize_multi_solution_analysis(
            real_x, reconstructed_xs, predicted_ys, y_original, 
            freq_data, nmse_errors, args.save_dir
        )
        print(f"多解生成分析结果已保存到: {multi_sol_path}")
    
    else:
        print("请指定可视化命令: forward, backward, 或 multi-solution")
        return


if __name__ == '__main__':
    main()
