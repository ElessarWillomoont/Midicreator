import numpy as np

# 已知的令牌数量
TOKEN_SIZE = 358818

# 线性回归模型的斜率和截距
m = 0.9606135203483412
c = -0.8980869297587443

# 预测模型参数数量
# 使用线性回归方程并将令牌数量转换为对数尺度
query_params = 10**(m * np.log10(TOKEN_SIZE) + c)

# 打印预测结果
print(f"Predicted parameters for {TOKEN_SIZE:,} tokens: {query_params:,.0f}")
