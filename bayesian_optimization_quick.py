"""
贝叶斯优化快速测试脚本（5 trials）
"""
exec(open('bayesian_optimization.py').read().replace('n_trials = 50', 'n_trials = 5').replace('num_epochs=50', 'num_epochs=20'))
