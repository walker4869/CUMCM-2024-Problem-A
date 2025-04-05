import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
plt.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 参数设置
v_head = 1  # 龙头速度（m/s）
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


def arc_length(theta, b):
    return 0.5 * b * (theta * np.sqrt(theta ** 2 + 1) + np.arcsinh(theta))


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


def calculate_crash_point(p):
    b = p / (2 * np.pi)
    s_0 = arc_length(theta_0, b)
    time_duration = int(1000 * p - 120)
    time_steps = np.arange(0, time_duration + 1)
    s_head = s_0 - v_head * time_steps
    theta_head = []
    for s in s_head:
        func = lambda theta: arc_length(theta, b) - s
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
    def func_head(y, x):
        return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (length_head - 0.55) ** 2

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
    def func_body(y, x):
        return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (length_body - 0.55) ** 2

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
    for i in range(0, time_duration + 1):
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

    crash_point = 0
    for i in range(0, len(theta_get_final)):
        top_left_head, bottom_left_head, top_right_head, bottom_right_head = calculate_rectangle(x_head[i], y_head[i], x_initial[i], y_initial[i])
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
            crash_point = i
            break
    r_head_final = b * theta_head[crash_point]
    return r_head_final


def compute_gradient(p, epsilon=1e-3):
    f_p = calculate_crash_point(p)
    f_p_plus_epsilon = calculate_crash_point(p + epsilon)
    gradient = (f_p_plus_epsilon - f_p) / epsilon
    return gradient


def binary(left, right):
    left = left * 1000
    right = right * 1000
    while right > left:
        mid = (left + right) // 2
        r_head_final = calculate_crash_point(mid / 1000)
        if r_head_final <= 4.5:
            right = mid - 1
        else:
            left = mid + 1
    return left / 1000


g = []
p = np.arange(0.3, 0.56, 0.01)
for p_x in p:
    g.append(calculate_crash_point(p_x))
g = np.array(g)

x_point1 = p[0]
y_point1 = g[0]
plt.scatter(x_point1, y_point1, color='red')
plt.annotate(f'({x_point1}, {y_point1})',  xy=(x_point1, y_point1),  xytext=(x_point1, y_point1))
max_index = np.argmax(g)
x_point2 = round(p[max_index], 2)
y_point2 = round(g[max_index], 2)
plt.scatter(x_point2, y_point2, color='red')
plt.annotate(f'({x_point2}, {y_point2})',  xy=(x_point2, y_point2),  xytext=(x_point2, y_point2))
plt.plot(p, g)
plt.title('f(p)关于p的图像')
plt.xlabel('p')
plt.ylabel('f(p)')
plt.show()

p_max = 0.55
p_min = p[max_index]
p_final = binary(p_min, p_max)
print("最小螺距为(m)", p_final)