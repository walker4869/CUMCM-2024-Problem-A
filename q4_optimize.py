import math
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
plt.rcParams['font.sans-serif'] = ['STZhongsong']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
v_head = 1  # 龙头速度（m/s）
p = 1.7 # 螺距（米）
R_ratio = 2  # 半径之比 R1:R2 = 2:1
R = 4.5  # 螺旋线在 r=4.5 处与圆相切
b = p / (2 * np.pi)  # 增长系数
# 板凳长度
length_head = 3.41  # 龙头长度（米）
length_body = 2.2  # 龙身和龙尾长度（米）
num_segments = 223  # 总节数（1 个龙头 + 221 个龙身 + 1 个龙尾）
# 初始角度
theta_0 = 32 * np.pi  # 初始角度（弧度）


def plot_circle(ax, x_center, y_center, radius, color='b', label=None):
    circle = plt.Circle((x_center, y_center), radius, color=color, fill=False, label=label)
    ax.add_patch(circle)


def spiral(b, theta):
    r = b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def curve_distribution(R):
    def equations(vars):
        x1, y1, x2, y2 = vars
        k = (b * np.sin(R / b) + R * np.cos(R / b)) / (b * np.cos(R / b) - R * np.sin(R / b))
        eq1 = y1 - R * np.sin(R / b) + (x1 - R * np.cos(R / b)) / k
        eq2 = y2 + R * np.sin(R / b) + (x2 + R * np.cos(R / b)) / k
        eq3 = (x1 - R * np.cos(R / b)) ** 2 + (y1 - R * np.sin(R / b)) ** 2 - 4 * (
                (x2 + R * np.cos(R / b)) ** 2 + (y2 + R * np.sin(R / b)) ** 2)
        eq4 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) - np.sqrt(
            (x1 - R * np.cos(R / b)) ** 2 + (y1 - R * np.sin(R / b)) ** 2) \
              - np.sqrt((x2 + R * np.cos(R / b)) ** 2 + (y2 + R * np.sin(R / b)) ** 2)
        return [eq1, eq2, eq3, eq4]

    initial_guess = [-1, 2, 1, -2]
    solution = fsolve(equations, initial_guess)
    x1, y1, x2, y2 = solution
    R1 = np.sqrt((x1 - R * np.cos(R / b)) ** 2 + (y1 - R * np.sin(R / b)) ** 2)
    R2 = np.sqrt((x2 + R * np.cos(R / b)) ** 2 + (y2 + R * np.sin(R / b)) ** 2)
    o1_to_o2 = np.array([x1 - x2, y1 - y2])
    vec1 = np.array([x1 - R * np.cos(R / b), y1 - R * np.sin(R / b)])
    vec2 = np.array([x2 + R * np.cos(R / b), y2 + R * np.sin(R / b)])
    sigma1 = np.arccos(np.dot(o1_to_o2, vec1) / (np.linalg.norm(o1_to_o2) * np.linalg.norm(vec1)))
    sigma2 = np.arccos(-np.dot(o1_to_o2, vec2) / (np.linalg.norm(o1_to_o2) * np.linalg.norm(vec2)))

    return R1, R2, sigma1, sigma2, x1, y1, x2, y2


# 模拟退火算法
def simulated_annealing(initial_R, max_iter=1000, init_temp=100, cooling_rate=0.99):
    current_R = initial_R
    R1, R2, sigma1, sigma2, x1, y1, x2, y2 = curve_distribution(current_R)
    current_length = R1 * sigma1 + R2 * sigma2
    best_R = current_R
    best_length = current_length
    temp = init_temp

    for i in range(max_iter):
        new_R = current_R - random.uniform(0, 0.5)  # 产生小范围的扰动, 需要保证R < 4.5
        if new_R < 0:
            continue
        R1, R2, sigma1, sigma2, x1, y1, x2, y2 = curve_distribution(new_R)
        if 2 * R2 < length_head - 0.55:
            continue
        new_length = R1 * sigma1 + R2 * sigma2
        delta_length = new_length - current_length
        if delta_length < 0 or random.uniform(0, 1) < math.exp(-delta_length / temp):
            current_R = new_R
            current_length = new_length
            if new_length < best_length:
                R1_best = R1
                R2_best = R2
                sigma1_best = sigma1
                sigma2_best = sigma2
                loc1_best = np.array([x1, y1])
                loc2_best = np.array([x2, y2])
                best_R = new_R
                best_length = new_length
        temp *= cooling_rate
    return best_R, best_length, R1_best, R2_best, sigma1_best, sigma2_best, loc1_best, loc2_best


best_R, best_length, R1_best, R2_best, sigma1_best, sigma2_best, loc1_best, loc2_best = simulated_annealing(R, max_iter=10000, init_temp=100, cooling_rate=0.99)
print('最优弧长1(m)：', R1_best * sigma1_best, '最优弧长2(m)：', R2_best * sigma2_best)
print('最优调头半径(m)：', best_R, '总弧长(m)：', best_length)
print('最优 R1(m)：', R1_best, '最优 R2(m)：', R2_best)
print('最优 sigma1(rad)：', sigma1_best, '最优 sigma2(rad)：', sigma2_best)
print('最优点1：', loc1_best, '最优点2：', loc2_best)

fig, ax = plt.subplots(figsize=(8, 8))
plot_circle(ax, loc1_best[0], loc1_best[1], R1_best, color='r', label='圆1')
plot_circle(ax, loc2_best[0], loc2_best[1], R2_best, color='g', label='圆2')
theta_vals = np.linspace(0, 20 * np.pi, 1000)
x_vals, y_vals = spiral(b, theta_vals)
ax.plot(x_vals, y_vals, label='盘入螺旋线', color='cyan')
x_vals, y_vals = spiral(-b, theta_vals)
ax.plot(x_vals, y_vals, label='盘出螺旋线', color='#E6E6FA')
plot_circle(ax, 0, 0, best_R, color='black', label='r = 4.5的圆')

x_values1 = [loc1_best[0], loc2_best[0]]
y_values1 = [loc1_best[1], loc2_best[1]]
plt.plot(x_values1, y_values1, marker='o')
x_values2 = [loc1_best[0], best_R * np.cos(best_R / b)]
y_values2 = [loc1_best[1], best_R * np.sin(best_R / b)]
plt.plot(x_values2, y_values2, marker='o')
x_values3 = [loc2_best[0], -best_R * np.cos(best_R / b)]
y_values3 = [loc2_best[1], -best_R * np.sin(best_R / b)]
plt.plot(x_values3, y_values3, marker='o')

ax.set_aspect('equal')
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.grid(True)
plt.legend()
plt.show()