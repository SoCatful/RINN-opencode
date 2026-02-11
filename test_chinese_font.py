import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持 - 尝试多种字体，提高兼容性
plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建测试数据
x = np.linspace(0, 4, 5)
y = np.array([0.275, 0.298, 0.485, 0.255, 0.35])

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'b-', alpha=0.5)
plt.scatter(x, y, c='b', s=10)
plt.xlabel('尝试次数')
plt.ylabel('验证损失')
plt.title('贝叶斯优化过程')
plt.grid(True, alpha=0.3)

# 保存图表
plt.savefig('test_chinese_font.png', dpi=150, bbox_inches='tight')
plt.close()

print("测试图表已保存为 test_chinese_font.png")
