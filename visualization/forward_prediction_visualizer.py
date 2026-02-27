import matplotlib.pyplot as plt
import os


def visualize_forward_prediction(real_y, predicted_y, freq_data, sample_index, save_dir):
    """
    可视化正向预测结果（固定x预测y）
    
    Args:
        real_y: 真实的y值，形状为(1, 202)，前101维为实部，后101维为虚部
        predicted_y: 预测的y值，形状为(1, 202)，前101维为实部，后101维为虚部
        freq_data: 频率数据，形状为(101,)
        sample_index: 样本索引，用于图表标题和文件名
        save_dir: 保存图表的目录
    """
    # 确保save_dir存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 可视化预测结果 - 分别显示实部和虚部
    # 实部（前101维）
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(freq_data[:101], real_y[0, :101], label='Real Re(S11)', color='blue', linewidth=2)
    plt.plot(freq_data[:101], predicted_y[0, :101], label='Predicted Re(S11)', color='red', linestyle='--', linewidth=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Re(S11)')
    plt.title(f'Comparison of Re(S11) prediction with fixed x - Sample {sample_index}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 虚部（后101维）
    plt.subplot(2, 1, 2)
    plt.plot(freq_data[:101], real_y[0, 101:], label='Real Im(S11)', color='green', linewidth=2)
    plt.plot(freq_data[:101], predicted_y[0, 101:], label='Predicted Im(S11)', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Im(S11)')
    plt.title(f'Comparison of Im(S11) prediction with fixed x - Sample {sample_index}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, f'fixed_x_predicted_y_{sample_index}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  Plot saved for sample {sample_index}: fixed_x_predicted_y_{sample_index}.png')
    return save_path


def visualize_multiple_forward_predictions(real_ys, predicted_ys, freq_data, save_dir):
    """
    可视化多个正向预测结果
    
    Args:
        real_ys: 真实的y值列表，每个元素形状为(1, 202)
        predicted_ys: 预测的y值列表，每个元素形状为(1, 202)
        freq_data: 频率数据，形状为(101,)
        save_dir: 保存图表的目录
    """
    # 确保save_dir存在
    os.makedirs(save_dir, exist_ok=True)
    
    saved_paths = []
    for i, (real_y, predicted_y) in enumerate(zip(real_ys, predicted_ys)):
        sample_index = i + 1
        save_path = visualize_forward_prediction(real_y, predicted_y, freq_data, sample_index, save_dir)
        saved_paths.append(save_path)
    
    return saved_paths
