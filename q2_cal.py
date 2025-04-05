import numpy as np
from scipy.optimize import fsolve

# 参数设置
p = 0.55  # 螺距（米）
b = p / (2 * np.pi)  # 增长系数
v_head = 1  # 龙头速度（m/s）
time_duration = 430  # 模拟时间 400 秒
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
    # 计算v_(n+1)
    v_n1 = (v_n * cos_phi_n) / cos_phi_n1
    return v_n1


def arc_length(theta):
    return 0.5 * b * (theta * np.sqrt(theta ** 2 + 1) + np.arcsinh(theta))


def func_head(y, x):
    return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (0.55 ** 2) * (length_head - 0.55) ** 2


def func_body(y, x):
    return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (0.55 ** 2) * (length_body - 0.55) ** 2


def calculate_rectangle(x1, y1, x2, y2, extend_length=0.275, width=0.15):
    vec = np.array([x2 - x1, y2 - y1])
    distance = np.linalg.norm(vec)
    unit_vec = vec / distance
    p1 = np.array([x1, y1]) - extend_length * unit_vec
    p2 = np.array([x2, y2]) + extend_length * unit_vec
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])
    top_left = p1 + width * perp_vec
    bottom_left = p1 - width * perp_vec
    top_right = p2 + width * perp_vec
    bottom_right = p2 - width * perp_vec
    return top_left, bottom_left, top_right, bottom_right


def is_point_in_rectangle(point, rect_points):
    Px, Py = point
    A, B, C, D = rect_points


    def cross_product(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])


    AB_P = cross_product(A, B, (Px, Py))
    BC_P = cross_product(B, C, (Px, Py))
    CD_P = cross_product(C, D, (Px, Py))
    DA_P = cross_product(D, A, (Px, Py))
    if (AB_P >= 0 and BC_P >= 0 and CD_P >= 0 and DA_P >= 0) or \
            (AB_P <= 0 and BC_P <= 0 and CD_P <= 0 and DA_P <= 0):
        return 1
    else:
        return 0


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

theta_initial = []
for theta in theta_head:
    theta_initial_guess = theta + 0.5
    theta_initial_solution = fsolve(func_head, theta_initial_guess, args=theta)
    theta_initial.append(theta_initial_solution[0])
theta_initial = np.array(theta_initial)

r_initial = b * theta_initial
x_initial = r_initial * np.cos(theta_initial)
y_initial = r_initial * np.sin(theta_initial)

positions_x.append(x_initial)
positions_y.append(y_initial)

theta_every = [theta_head, theta_initial]
theta_first = theta_initial
for i in range(2, num_segments + 1):
    theta_l = []
    for theta in theta_initial:
        theta_guess = theta + 0.5
        theta_solution = fsolve(func_body, theta_guess, args=theta)
        theta_l.append(theta_solution[0])
    theta_l = np.array(theta_l)
    r = b * theta_l
    x = r * np.cos(theta_l)
    y = r * np.sin(theta_l)
    positions_x.append(x)
    positions_y.append(y)
    theta_initial = theta_l
    theta_every.append(theta_l)

theta_get_final = []
for i in range(360, time_duration + 1):
    theta_cal = []
    theta_guess1 = theta_head[i] + 2 * np.pi
    theta_guess2 = theta_first[i] + 2 * np.pi
    for j in range(1, num_segments):
        if theta_every[j][i] < theta_guess1 < theta_every[j + 1][i]:
            theta_cal.append((theta_every[j][i], theta_every[j + 1][i]))
        if theta_every[j][i] < theta_guess2 < theta_every[j + 1][i]:
            theta_cal.append((theta_every[j][i], theta_every[j + 1][i]))
            break
    theta_get_final.append(theta_cal)
theta_get_final = np.array(theta_get_final)

crash_point_large = 0
for i in range(0, len(theta_get_final)):
    top_left_head, bottom_left_head, top_right_head, bottom_right_head = calculate_rectangle(x_head[i + 360], y_head[i + 360], x_initial[i + 360], y_initial[i + 360])
    r_impact1 = b * theta_get_final[i][0]
    x_impact1 = r_impact1 * np.cos(theta_get_final[i][0])
    y_impact1 = r_impact1 * np.sin(theta_get_final[i][0])
    top_left1, bottom_left1, top_right1, bottom_right1 = calculate_rectangle(x_impact1[0], y_impact1[0], x_impact1[1], y_impact1[1])
    r_impact2 = b * theta_get_final[i][1]
    x_impact2 = r_impact2 * np.cos(theta_get_final[i][1])
    y_impact2 = r_impact2 * np.sin(theta_get_final[i][1])
    top_left2, bottom_left2, top_right2, bottom_right2 = calculate_rectangle(x_impact2[0], y_impact2[0], x_impact2[1], y_impact2[1])
    vertices1 = np.array([top_left1, bottom_left1, bottom_right1, top_right1])
    vertices2 = np.array([top_left2, bottom_left2, bottom_right2, top_right2])
    if is_point_in_rectangle(top_left_head, vertices1) or is_point_in_rectangle(bottom_left_head, vertices1) or \
            is_point_in_rectangle(top_right_head, vertices1) or is_point_in_rectangle(bottom_right_head, vertices1) or \
            is_point_in_rectangle(top_left_head, vertices2) or is_point_in_rectangle(bottom_left_head, vertices2) or \
            is_point_in_rectangle(top_right_head, vertices2) or is_point_in_rectangle(bottom_right_head, vertices2):
        crash_point_large = i + 360
        break

# 细分时间范围
time_steps_fine = np.arange(crash_point_large - 1, crash_point_large, 0.01)

# 重新计算头部位置和初始位置
s_head_fine = s_0 - v_head * time_steps_fine
theta_head_fine = []
for s in s_head_fine:
    func = lambda theta: arc_length(theta) - s
    theta_initial_guess = theta_0 + (s - s_0) / (b * np.sqrt(theta_0 ** 2 + 1))
    theta_solution = fsolve(func, theta_initial_guess)
    theta_head_fine.append(theta_solution[0])
theta_head_fine = np.array(theta_head_fine)

r_head_fine = b * theta_head_fine
x_head_fine = r_head_fine * np.cos(theta_head_fine)
y_head_fine = r_head_fine * np.sin(theta_head_fine)

theta_initial_fine = []
for theta in theta_head_fine:
    theta_initial_guess = theta + 0.5
    theta_initial_solution = fsolve(func_head, theta_initial_guess, args=theta)
    theta_initial_fine.append(theta_initial_solution[0])
theta_initial_fine = np.array(theta_initial_fine)

r_initial_fine = b * theta_initial_fine
x_initial_fine = r_initial_fine * np.cos(theta_initial_fine)
y_initial_fine = r_initial_fine * np.sin(theta_initial_fine)

# 重新细分计算每一节的位置
theta_every_fine = [theta_head_fine, theta_initial_fine]
theta_first_fine = theta_initial_fine
for i in range(2, num_segments + 1):
    theta_l_fine = []
    for theta in theta_initial_fine:
        theta_guess_fine = theta + 0.5
        theta_solution_fine = fsolve(func_body, theta_guess_fine, args=theta)
        theta_l_fine.append(theta_solution_fine[0])
    theta_l_fine = np.array(theta_l_fine)
    theta_every_fine.append(theta_l_fine)
    theta_initial_fine = theta_l_fine

# 计算细分时间段的碰撞检测
theta_get_final_fine = []
for i in range(len(time_steps_fine)):
    theta_cal_fine = []
    theta_guess1_fine = theta_head_fine[i] + 2 * np.pi
    theta_guess2_fine = theta_first_fine[i] + 2 * np.pi
    for j in range(1, num_segments - 1):
        if theta_every_fine[j][i] < theta_guess1_fine < theta_every_fine[j + 1][i]:
            theta_cal_fine.append((theta_every_fine[j][i], theta_every_fine[j + 1][i]))
        if theta_every_fine[j][i] < theta_guess2_fine < theta_every_fine[j + 1][i]:
            theta_cal_fine.append((theta_every_fine[j][i], theta_every_fine[j + 1][i]))
            break
    theta_get_final_fine.append(theta_cal_fine)
theta_get_final_fine = np.array(theta_get_final_fine)

# 进行碰撞检测
for i in range(len(theta_get_final_fine)):
    top_left_head_fine, bottom_left_head_fine, top_right_head_fine, bottom_right_head_fine = calculate_rectangle(
        x_head_fine[i], y_head_fine[i], x_initial_fine[i], y_initial_fine[i])
    r_impact1_fine = b * theta_get_final_fine[i][0]
    x_impact1_fine = r_impact1_fine * np.cos(theta_get_final_fine[i][0])
    y_impact1_fine = r_impact1_fine * np.sin(theta_get_final_fine[i][0])
    top_left1_fine, bottom_left1_fine, top_right1_fine, bottom_right1_fine = calculate_rectangle(
        x_impact1_fine[0], y_impact1_fine[0], x_impact1_fine[1], y_impact1_fine[1])

    r_impact2_fine = b * theta_get_final_fine[i][1]
    x_impact2_fine = r_impact2_fine * np.cos(theta_get_final_fine[i][1])
    y_impact2_fine = r_impact2_fine * np.sin(theta_get_final_fine[i][1])
    top_left2_fine, bottom_left2_fine, top_right2_fine, bottom_right2_fine = calculate_rectangle(
        x_impact2_fine[0], y_impact2_fine[0], x_impact2_fine[1], y_impact2_fine[1])

    vertices1_fine = np.array([top_left1_fine, bottom_left1_fine, bottom_right1_fine, top_right1_fine])
    vertices2_fine = np.array([top_left2_fine, bottom_left2_fine, bottom_right2_fine, top_right2_fine])

    if is_point_in_rectangle(top_left_head_fine, vertices1_fine) or \
            is_point_in_rectangle(bottom_left_head_fine, vertices1_fine) or \
            is_point_in_rectangle(top_right_head_fine, vertices1_fine) or \
            is_point_in_rectangle(bottom_right_head_fine, vertices1_fine) or \
            is_point_in_rectangle(top_left_head_fine, vertices2_fine) or \
            is_point_in_rectangle(bottom_left_head_fine, vertices2_fine) or \
            is_point_in_rectangle(top_right_head_fine, vertices2_fine) or \
            is_point_in_rectangle(bottom_right_head_fine, vertices2_fine):
        crash_point_fine = time_steps_fine[i]
        print(f"碰撞时间为: {round(crash_point_fine, 2)} 秒")
        break


