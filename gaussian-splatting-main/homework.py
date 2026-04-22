import numpy as np
import matplotlib.pyplot as plt

# 1) 定义函数：按需修改
def f(x, y):
    return x**2 + y**2 + 0.5 * x * y
    # 示例：如果想画 x + 2y + x*y
    # return x + 2*y + x*y

# 2) 网格（[-1, 1]^2 区间）
n = 200
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 3) 3D 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 曲面
ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.9)

# 等高线（可选；注释掉也可以）
ax.contour(X, Y, Z, offset=Z.min(), levels=15)

# 轴标签与标题
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('f(x, y) = x^2 + y^2 + 0.5xy')

plt.tight_layout()
plt.show()
