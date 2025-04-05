import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# 参数设置
p = 0.55  # 螺距（米）
b = p / (2 * np.pi)  # 增长系数
v_head = 1  # 龙头速度（m/s）
time_duration = 300  # 模拟时间 300 秒
time_steps = np.arange(0, time_duration + 1)  # 时间序列
length_head = 3.41  # 龙头长度（米）
length_body = 2.2  # 龙身和龙尾长度（米）
num_segments = 223  # 总节数（1 个龙头 + 221 个龙身 + 1 个龙尾）
theta_0 = 32 * np.pi  # 初始角度（弧度）


def line_vector(theta_n, theta_n1, b):
    x_n = b * theta_n * np.cos(theta_n)
    y_n = b * theta_n * np.sin(theta_n)
    x_n1 = b * theta_n1 * np.cos(theta_n1)
    y_n1 = b * theta_n1 * np.sin(theta_n1)

    return np.array([x_n1 - x_n, y_n1 - y_n])


def tangent_vector(theta):
    tan_slope = (np.sin(theta) + theta * np.cos(theta)) / (np.cos(theta) - theta * np.sin(theta))
    return np.array([1, tan_slope])


def cos_phi(theta_n, theta_n1, b):
    line_vec = line_vector(theta_n, theta_n1, b)
    tangent_vec_n = tangent_vector(theta_n)
    tangent_vec_n1 = tangent_vector(theta_n1)
    cos_phi_n = np.dot(tangent_vec_n, line_vec) / (np.linalg.norm(tangent_vec_n) * np.linalg.norm(line_vec))
    cos_phi_n1 = np.dot(tangent_vec_n1, line_vec) / (np.linalg.norm(tangent_vec_n1) * np.linalg.norm(line_vec))
    return np.abs(cos_phi_n), np.abs(cos_phi_n1)


def calculate_v_n1(v_n, theta_n, theta_n1, b):
    cos_phi_n, cos_phi_n1= cos_phi(theta_n, theta_n1, b)
    v_n1 = (v_n * cos_phi_n) / cos_phi_n1
    return v_n1


def arc_length(theta):
    return 0.5 * b * (theta * np.sqrt(theta ** 2 + 1) + np.arcsinh(theta))


def func_head(y, x):
    return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (0.55 ** 2) * (length_head - 0.55) ** 2


def func_body(y, x):
    return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (0.55 ** 2) * (length_body - 0.55) ** 2



s_0 = arc_length(theta_0)
s_head = s_0 - v_head * time_steps

theta_head = []
for s in s_head:
    func = lambda theta: arc_length(theta) - s
    theta_initial_guess = theta_0 + (s - s_0) / (b * np.sqrt(theta_0 ** 2 + 1))
    theta_solution = fsolve(func, theta_initial_guess)
    theta_head.append(theta_solution[0])
theta_head = np.array(theta_head)

r_head = b * theta_head
x_head = r_head * np.cos(theta_head)
y_head = r_head * np.sin(theta_head)

positions_x = [x_head]
positions_y = [y_head]
velocities = [v_head]

theta_initial = []
for theta in theta_head:
    theta_initial_guess = theta + 0.1
    theta_initial_solution = fsolve(func_head, theta_initial_guess, args=theta)
    theta_initial.append(theta_initial_solution[0])
theta_initial = np.array(theta_initial)

r_initial = b * theta_initial
x_initial = r_initial * np.cos(theta_initial)
y_initial = r_initial * np.sin(theta_initial)
v_initial = []
for i in range(0, time_duration + 1):
    v_n1 = calculate_v_n1(v_head, theta_head[i], theta_initial[i], b)
    v_initial.append(v_n1)
v_initial = np.array(v_initial)

positions_x.append(x_initial)
positions_y.append(y_initial)
velocities.append(v_initial)

for i in range(2, num_segments + 1):
    theta_l = []
    for theta in theta_initial:
        theta_guess = theta + 0.1
        theta_solution = fsolve(func_body, theta_guess, args=theta)
        theta_l.append(theta_solution[0])
    theta_l = np.array(theta_l)
    r = b * theta_l
    x = r * np.cos(theta_l)
    y = r * np.sin(theta_l)
    v = []
    for j in range(0, time_duration + 1):
        v_n1 = calculate_v_n1(v_initial[j], theta_initial[j], theta_l[j], b)
        v.append(v_n1)
    v = np.array(v)
    positions_x.append(x)
    positions_y.append(y)
    velocities.append(v)
    theta_initial = theta_l
    v_initial = v

data_dict1 = {
    'Time (s)': time_steps
}
data_dict2 = {
    'Time (s)': time_steps
}
for i in range(num_segments + 1):
    data_dict1[f'第{i+1}节x (m)'] = positions_x[i]
    data_dict1[f'第{i+1}节y (m)'] = positions_y[i]
for i in range(num_segments + 1):
    data_dict2[f'第{i+1}节速度 (m/s)'] = velocities[i]
df1 = pd.DataFrame(data_dict1)
df2 = pd.DataFrame(data_dict2)
df1_transposed = df1.transpose()
df2_transposed = df2.transpose()
with pd.ExcelWriter('result.xlsx') as writer:
    df1_transposed.to_excel(writer, sheet_name='Sheet1', header=True)
    df2_transposed.to_excel(writer, sheet_name='Sheet2', header=True)

