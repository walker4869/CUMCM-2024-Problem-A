import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
plt.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 参数设置
v_head = 1  # 龙头速度（m/s）
p = 1.7 # 螺距（米）
R_ratio = 2  # 半径之比 R1:R2 = 2:1
R = 4.5  # 螺旋线在 r=4.5 处与圆相切
b = p / (2 * np.pi)  # 增长系数
length_head = 3.41  # 龙头长度（米）
length_body = 2.2  # 龙身和龙尾长度（米）
num_segments = 223  # 总节数（1 个龙头 + 221 个龙身 + 1 个龙尾）
theta_0 = 32 * np.pi  # 初始角度（弧度）

def equations(vars):
    x1, y1, x2, y2 = vars
    k = (b * np.sin(R / b) + R * np.cos(R /  b)) / (b * np.cos(R / b) - R * np.sin(R / b))
    eq1 = y1 - R * np.sin(R / b) + (x1 - R * np.cos(R / b)) / k
    eq2 = y2 + R * np.sin(R / b) + (x2 + R * np.cos(R / b)) / k
    eq3 = (x1 - R * np.cos(R / b)) ** 2 + (y1 - R * np.sin(R / b)) ** 2 - 4 * ((x2 + R * np.cos(R / b)) ** 2 + (y2 + R * np.sin(R / b)) ** 2)
    eq4 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) - np.sqrt((x1 - R * np.cos(R / b)) ** 2 + (y1 - R * np.sin(R / b)) ** 2) \
          - np.sqrt((x2 + R * np.cos(R / b)) ** 2 + (y2 + R * np. sin(R / b)) ** 2)
    return [eq1, eq2, eq3, eq4]


def plot_circle(ax, x_center, y_center, radius, color='b', label=None):
    circle = plt.Circle((x_center, y_center), radius, color=color, fill=False, label=label)
    ax.add_patch(circle)


def spiral(b, theta):
    r = b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


initial_guess = [-1, 3, 1, -2]
solution = fsolve(equations, initial_guess)
x1, y1, x2, y2 = solution
R1 = np.sqrt((x1 - R * np.cos(R / b)) ** 2 + (y1 - R * np.sin(R / b))** 2)
R2 = np.sqrt((x2 + R * np.cos(R / b)) ** 2 + (y2 + R * np.sin(R / b)) ** 2)
print(f"圆半径: ({R1}), 圆心坐标: ({x1}, {y1})")
print(f"圆半径: ({R2}), 圆心坐标: ({x2}, {y2})")
o1_to_o2 = np.array([x1 - x2, y1 - y2])
vec1 = np.array([x1 - R * np.cos(R / b), y1 - R * np.sin(R / b)])
vec2 = np.array([x2 + R * np.cos(R / b), y2 + R * np.sin(R / b)])
sigma1 = np.arccos(np.dot(o1_to_o2, vec1) / (np.linalg.norm(o1_to_o2) * np.linalg.norm(vec1)))
sigma2 = np.arccos(-np.dot(o1_to_o2, vec2) / (np.linalg.norm(o1_to_o2) * np.linalg.norm(vec2)))
print(f"弧长1：{R1 * sigma1}, 弧长2：{R2 * sigma2}, 弧长和：{R1 * sigma1 + (R2 * sigma2)}")

fig, ax = plt.subplots(figsize=(8, 8))
plot_circle(ax, x1, y1, R1, color='r', label='圆1')
plot_circle(ax, x2, y2, R2, color='g', label='圆2')
theta_vals = np.linspace(0, 20 * np.pi, 1000)
x_vals, y_vals = spiral(b, theta_vals)
ax.plot(x_vals, y_vals, label='盘入螺旋线', color='cyan')
x_vals, y_vals = spiral(-b, theta_vals)
ax.plot(x_vals, y_vals, label='盘出螺旋线', color='#E6E6FA')
plot_circle(ax, 0, 0, R, color='black', label='r = 4.5的圆')

x_values1 = [x1, x2]
y_values1 = [y1, y2]
plt.plot(x_values1, y_values1, marker='o')
x_values2 = [x1, R * np.cos(R / b)]
y_values2 = [y1, R * np.sin(R / b)]
plt.plot(x_values2, y_values2, marker='o')
x_values3 = [x2, -R * np.cos(R / b)]
y_values3 = [y2, -R * np.sin(R / b)]
plt.plot(x_values3, y_values3, marker='o')

ax.set_aspect('equal')
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.grid(True)
plt.legend()
plt.show()