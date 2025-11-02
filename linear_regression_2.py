import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

w = 0
b = 0
learning_rate = 0.2
n_iterations = 1000

for i in range(n_iterations):
    y_pred = w * x + b
    dw = (-1 / len(x)) * np.sum((y - y_pred) * x)
    db = (-1 / len(x)) * np.sum(y - y_pred)
    w = w - learning_rate * dw
    b = b - learning_rate * db

# 输出最终参数
print(f"手动实现的斜率 (w): {w}")
print(f"手动实现的截距 (b): {b}")

y_pred = w * x + b
plt.scatter(x, y)
plt.plot(x, y_pred, color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Manual Gradient Descent Fit")
plt.show()
