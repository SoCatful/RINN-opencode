# 绘图规范

## 1. 基本要求

### 1.1 字体设置

```python
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

### 1.2 图表大小

```python
# 标准单图
plt.figure(figsize=(12, 7))

# 多子图（用于Y->X参数和曲线）
plt.figure(figsize=(20, 15))

# 其他布局
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
```

### 1.3 分辨率

```python
plt.savefig('figure.png', dpi=300)
```

---

## 2. Y->X 标准绘图范式

### 2.1 重建几何参数和曲线图

```python
def plot_reconstructed_parameters_and_curves(results, result_dir):
    """
    生成包含重建几何参数和Re/Im曲线的图
    
    参数:
        results: 包含频率、原始数据和重建数据的字典
        result_dir: 保存结果的目录
    """
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
```

### 2.2 S11值图

```python
def plot_s11_values(results, result_dir):
    """
    生成包含S11值的图
    
    参数:
        results: 包含频率、原始S11和预测S11的字典
        result_dir: 保存结果的目录
    """
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
```

### 2.3 数据类型处理

```python
def generate_ideal_data():
    """
    生成理想数据和变换后的数据
    
    返回:
        包含频率、原始数据和变换后数据的字典
    """
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
```
