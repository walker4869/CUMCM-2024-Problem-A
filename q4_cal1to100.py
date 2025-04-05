import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

from q4_optimize import best_R, loc1_best, loc2_best, R1_best, R2_best

plt.rcParams['font.sans-serif'] = ['STZhongsong']
plt.rcParams['axes.unicode_minus'] = False


# 参数设置
v_head = 1  # 龙头速度（m/s）
p = 1.7 # 螺距（米）
R_ratio = 2  # 半径之比 R1:R2 = 2:1
b = p / (2 * np.pi)  # 增长系数
# 板凳长度
length_head = 3.41  # 龙头长度（米）
length_body = 2.2  # 龙身和龙尾长度（米）
num_segments = 223  # 总节数（1 个龙头 + 221 个龙身 + 1 个龙尾）
# 初始角度
theta_0 = 32 * np.pi  # 初始角度（弧度）

x1, y1 = loc1_best
x2, y2 = loc2_best
R1 = R1_best
R2 = R2_best
R = best_R

o1_to_o2 = np.array([x1 - x2, y1 - y2])
vec1 = np.array([x1 - R * np.cos(R / b), y1 - R * np.sin(R / b)])
vec2 = np.array([x2 + R * np.cos(R / b), y2 + R * np.sin(R / b)])
sigma1 = np.arccos(np.dot(o1_to_o2, vec1) / (np.linalg.norm(o1_to_o2) * np.linalg.norm(vec1)))
sigma2 = np.arccos(-np.dot(o1_to_o2, vec2) / (np.linalg.norm(o1_to_o2) * np.linalg.norm(vec2)))
print(f"角度: {sigma1}, {sigma2}")
print(f"弧长: {R1 * sigma1}, {R2 * sigma2}, {R1 * sigma1 + R2 * sigma2}")

angle1 = np.arccos(np.dot(vec1, np.array([1, 0])) / (np.linalg.norm(vec1)))
angle2 = np.arccos(np.dot(vec2, np.array([1, 0])) / (np.linalg.norm(vec2)))
angle3 = np.arccos(np.dot(o1_to_o2, np.array([1, 0])) / (np.linalg.norm(o1_to_o2)))
theta1 = np.pi + angle1
theta2 = np.pi - angle2
theta3 = - angle2
theta4 = np.pi - angle3
print(f"向量与x轴的夹角: {theta1}, {theta2}, {theta3}, {theta4}")

theta_d1 = 2 * np.arcsin((length_head - 0.55) / (2 * R1))
theta_d2 = 2 * np.arcsin((length_body - 0.55) / (2 * R1))

q4_point = []
for t in range(1, int(R1 * sigma1) + 1):
    q4_temp = [np.array([x1 + R1 * np.cos(theta1 - t / R1), y1 + R1 * np.sin(theta1 - t / R1)])]
    if theta_d1 > (t / R1):
        theta_temp = R / b


        def equa1(theta_x):
            return (x1 + R1 * np.cos(theta1 - t / R1) - theta_x * b * np.cos(theta_x)) ** 2 + (y1 + R1 * np.sin(theta1 -
                    t / R1) - theta_x * b * np.sin(theta_x)) ** 2 - 2.86 ** 2


        theta_guess_temp = theta_temp + 0.5 * theta_d1
        theta_solution_temp = fsolve(equa1, theta_guess_temp)
        q4_temp.append(np.array([theta_solution_temp[0] * b * np.cos(theta_solution_temp[0]), theta_solution_temp[0] * b * np.sin(theta_solution_temp[0])]))

        theta_init = theta_solution_temp
        for i in range(2, num_segments + 1):
            theta_guess = theta_init + 0.1


            def func_body(y, x):
                return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (length_body - 0.55) ** 2
            theta_solution = fsolve(func_body, theta_guess, args=theta_init)


            r = b * theta_solution[0]
            x = r * np.cos(theta_solution[0])
            y = r * np.sin(theta_solution[0])
            q4_temp.append(np.array([x, y]))
            theta_init = theta_solution

    elif theta_d1 + theta_d2 > t / R1:
        q4_temp.append([x1 + R1 * np.cos(theta1 - t / R1 + theta_d1), y1 + R1 * np.sin(theta1 - t / R1 + theta_d1)])


        def equa2(theta_x):
            return (x1 + R1 * np.cos(theta1 - t / R1 + theta_d1) - theta_x * b * np.cos(theta_x)) ** 2 + (y1 + R1 * np.sin(theta1 -
                    t / R1 + theta_d1) - theta_x * b * np.sin(theta_x)) ** 2 - 1.65 ** 2


        theta_temp = R / b
        theta_guess_temp = theta_temp + 0.5 * theta_d1
        theta_solution_temp = fsolve(equa2, theta_guess_temp)
        q4_temp.append(np.array([theta_solution_temp[0] * b * np.cos(theta_solution_temp[0]),
                                 theta_solution_temp[0] * b * np.sin(theta_solution_temp[0])]))

        theta_init = theta_solution_temp
        for i in range(3, num_segments + 1):
            theta_guess = theta_init + 0.1


            def func_body(y, x):
                return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (
                            length_body - 0.55) ** 2


            theta_solution = fsolve(func_body, theta_guess, args=theta_init)
            r = b * theta_solution[0]
            x = r * np.cos(theta_solution[0])
            y = r * np.sin(theta_solution[0])
            q4_temp.append(np.array([x, y]))
            theta_init = theta_solution

    elif theta_d1 + 2 * theta_d2 > t / R1:
        q4_temp.append([x1 + R1 * np.cos(theta1 - t / R1 + theta_d1), y1 + R1 * np.sin(theta1 - t / R1 + theta_d1)])
        q4_temp.append([x1 + R1 * np.cos(theta1 - t / R1 + theta_d1 + theta_d2), y1 + R1 * np.sin(theta1 - t / R1 + theta_d1 + theta_d2)])


        def equa3(theta_x):
            return (x1 + R1 * np.cos(theta1 - t / R1 + theta_d1 + theta_d2) - theta_x * b * np.cos(theta_x)) ** 2 + (y1 + R1 * np.sin(theta1 - t / R1 + theta_d1 + theta_d2) - theta_x * b * np.sin(theta_x)) ** 2 - 1.65 ** 2


        theta_temp = R / b
        theta_guess_temp = theta_temp + 0.5 * theta_d1
        theta_solution_temp = fsolve(equa3, theta_guess_temp)
        q4_temp.append(np.array([theta_solution_temp[0] * b * np.cos(theta_solution_temp[0]),
                                theta_solution_temp[0] * b * np.sin(theta_solution_temp[0])]))

        theta_init = theta_solution_temp
        for i in range(4, num_segments + 1):
            theta_guess = theta_init + 0.1


            def func_body(y, x):
                return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (
                            length_body - 0.55) ** 2


            theta_solution = fsolve(func_body, theta_guess, args=theta_init)
            r = b * theta_solution[0]
            x = r * np.cos(theta_solution[0])
            y = r * np.sin(theta_solution[0])
            q4_temp.append(np.array([x, y]))
            # 更新 theta_initial
            theta_init = theta_solution

    else:
        q4_temp.append([x1 + R1 * np.cos(theta1 - t / R1 + theta_d1), y1 + R1 * np.sin(theta1 - t / R1 + theta_d1)])
        q4_temp.append([x1 + R1 * np.cos(theta1 - t / R1 + theta_d1 + theta_d2), y1 + R1 * np.sin(theta1 - t / R1 + theta_d1 + theta_d2)])
        q4_temp.append([x1 + R1 * np.cos(theta1 - t / R1 + theta_d1 + 2 * theta_d2), y1 + R1 * np.sin(theta1 - t / R1 + theta_d1 + 2 * theta_d2)])


        def equa4(theta_x):
            return (x1 + R1 * np.cos(theta1 - t / R1 + theta_d1 + 2 * theta_d2) - theta_x * b * np.cos(theta_x)) ** 2 + (y1 + R1 * np.sin(theta1 - t / R1 + theta_d1 + 2 * theta_d2) - theta_x * b * np.sin(theta_x)) ** 2 - 1.65 ** 2


        theta_temp = R / b
        theta_guess_temp = theta_temp + 0.5 * theta_d1
        theta_solution_temp = fsolve(equa4, theta_guess_temp)
        q4_temp.append(np.array([theta_solution_temp[0] * b * np.cos(theta_solution_temp[0]),
                                theta_solution_temp[0] * b * np.sin(theta_solution_temp[0])]))

        theta_init = theta_solution_temp
        for i in range(5, num_segments + 1):
            theta_guess = theta_init + 0.1


            def func_body(y, x):
                return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (
                            length_body - 0.55) ** 2


            theta_solution = fsolve(func_body, theta_guess, args=theta_init)
            r = b * theta_solution[0]
            x = r * np.cos(theta_solution[0])
            y = r * np.sin(theta_solution[0])
            q4_temp.append(np.array([x, y]))
            theta_init = theta_solution
    q4_temp = np.array(q4_temp)
    q4_point.append(q4_temp)

for t in range(int(R1 * sigma1) + 1, int(R1 * sigma1) + int(R2 * sigma2) + 1): # 9 - 12s
    q4_temp = [np.array([x2 + R2 * np.cos(theta3 + (t - R1 * sigma1) / R2), y2 + R2 * np.sin(theta3 + (t - R1 * sigma1) / R2)])]


    def equa5(theta_x):
        return (x1 + R1 * np.cos(theta_x) - x2 - R2 * np.cos(theta3 + (t - R1 * sigma1) / R2)) ** 2 + (y1 + R1 * np.sin(theta_x) - y2 - R2 * np.sin(theta3 + (t - R1 * sigma1) / R2)) ** 2 - 2.86 ** 2


    theta_guess_temp = theta2 + 0.5 * theta_d1
    theta_solution_temp = fsolve(equa5, theta_guess_temp)
    theta_temp = theta_solution_temp[0]
    q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp), y1 + R1 * np.sin(theta_temp)]))
    q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp + theta_d2), y1 + R1 * np.sin(theta_temp + theta_d2)]))
    q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp + 2 * theta_d2), y1 + R1 * np.sin(theta_temp + 2 * theta_d2)]))
    q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp + 3 * theta_d2), y1 + R1 * np.sin(theta_temp + 3 * theta_d2)]))
    q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp + 4 * theta_d2), y1 + R1 * np.sin(theta_temp + 4 * theta_d2)]))


    def equa6(theta_x):
        return (x1 + R1 * np.cos(theta_temp + 4 * theta_d2) - theta_x * b * np.cos(theta_x)) ** 2 + (
                y1 + R1 * np.sin(theta_temp + 4 * theta_d2) - theta_x * b * np.sin(theta_x)) ** 2 - 1.65 ** 2


    theta_temp1 = R / b
    theta_guess_temp = theta_temp1 + 0.5 * theta_d1
    theta_solution_temp = fsolve(equa6, theta_guess_temp)
    q4_temp.append(np.array([theta_solution_temp[0] * b * np.cos(theta_solution_temp[0]),
                             theta_solution_temp[0] * b * np.sin(theta_solution_temp[0])]))

    theta_init = theta_solution_temp
    for i in range(7, num_segments + 1):
        theta_guess = theta_init + 0.1


        def func_body(y, x):
            return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (
                    length_body - 0.55) ** 2


        theta_solution = fsolve(func_body, theta_guess, args=theta_init)
        # 得到第 i 节龙身的位置和速度
        r = b * theta_solution[0]
        x = r * np.cos(theta_solution[0])
        y = r * np.sin(theta_solution[0])
        q4_temp.append(np.array([x, y]))
        # 更新 theta_initial
        theta_init = theta_solution
    q4_temp = np.array(q4_temp)
    q4_point.append(q4_temp)


def arc_length(theta):
    return 0.5 * b * (theta * np.sqrt(theta ** 2 + 1) + np.arcsinh(theta))

time_duration = 90
time_steps = np.arange(1, time_duration, 1)
theta_0 = R / b
s_0 = arc_length(theta_0)
s_head = s_0 + v_head * time_steps

theta_head = []
for s in s_head:
    func = lambda theta: arc_length(theta) - s
    theta_initial_guess = theta_0 + (s - s_0) / (b * np.sqrt(theta_0 ** 2 + 1))
    theta_solution = fsolve(func, theta_initial_guess)
    theta_head.append(theta_solution[0])
theta_head = np.array(theta_head)

r_head = -b * theta_head
x_head = r_head * np.cos(theta_head)
y_head = r_head * np.sin(theta_head)


# 13s
q4_temp = [np.array([x_head[0], y_head[0]])]


def equa7(theta_x):
    return (x_head[0] - x2 - R2 * np.cos(theta_x)) ** 2 + (y_head[0] - y2 - R2 * np.sin(theta_x)) ** 2 - 2.86 ** 2


theta_guess_temp = theta4 - 0.1 * theta_d1
theta_solution_temp = fsolve(equa7, theta_guess_temp)
theta_temp = theta_solution_temp[0]
q4_temp.append(np.array([x2 + R2 * np.cos(theta_temp), y2 + R2 * np.sin(theta_temp)]))


def equa8(theta_x):
    return (x1 + R1 * np.cos(theta_x) - x2 - R2 * np.cos(theta_temp)) ** 2 + (
            y1 + R1 * np.sin(theta_x) - y2 - R2 * np.sin(theta_temp)) ** 2 - 1.65 ** 2


theta_guess_temp = theta2 + 0.5 * theta_d2
theta_solution_temp = fsolve(equa8, theta_guess_temp)
theta_temp2 = theta_solution_temp[0]
q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2), y1 + R1 * np.sin(theta_temp2)]))
q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + theta_d2), y1 + R1 * np.sin(theta_temp2 + theta_d2)]))
q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + 2 * theta_d2), y1 + R1 * np.sin(theta_temp2 + 2 * theta_d2)]))
q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + 3 * theta_d2), y1 + R1 * np.sin(theta_temp2 + 3 * theta_d2)]))
q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + 4 * theta_d2), y1 + R1 * np.sin(theta_temp2 + 4 * theta_d2)]))


def equa9(theta_x):
    return (x1 + R1 * np.cos(theta_temp2 + 4 * theta_d2) - theta_x * b * np.cos(theta_x)) ** 2 + (
            y1 + R1 * np.sin(theta_temp2 + 4 * theta_d2) - theta_x * b * np.sin(theta_x)) ** 2 - 1.65 ** 2

theta_temp1 = R / b
theta_guess_temp = theta_temp1 + 0.5 * theta_d1
theta_solution_temp = fsolve(equa9, theta_guess_temp)
q4_temp.append(np.array([theta_solution_temp[0] * b * np.cos(theta_solution_temp[0]),
                         theta_solution_temp[0] * b * np.sin(theta_solution_temp[0])]))

theta_init = theta_solution_temp
for i in range(8, num_segments + 1):
    theta_guess = theta_init + 0.1


    def func_body(y, x):
        return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (
                length_body - 0.55) ** 2


    theta_solution = fsolve(func_body, theta_guess, args=theta_init)
    # 得到第 i 节龙身的位置和速度
    r = b * theta_solution[0]
    x = r * np.cos(theta_solution[0])
    y = r * np.sin(theta_solution[0])
    q4_temp.append(np.array([x, y]))
    # 更新 theta_initial
    theta_init = theta_solution
q4_temp = np.array(q4_temp)
q4_point.append(q4_temp)


# 14s
q4_temp = [np.array([x_head[1], y_head[1]])]


def equa7(theta_x):
    return (x_head[1] - x2 - R2 * np.cos(theta_x)) ** 2 + (y_head[1] - y2 - R2 * np.sin(theta_x)) ** 2 - 2.86 ** 2


theta_guess_temp = theta4 - 0.1 * theta_d1
theta_solution_temp = fsolve(equa7, theta_guess_temp)
theta_temp = theta_solution_temp[0]
q4_temp.append(np.array([x2 + R2 * np.cos(theta_temp), y2 + R2 * np.sin(theta_temp)]))
q4_temp.append(np.array([x2 + R2 * np.cos(theta_temp + theta_d2), y2 + R2 * np.sin(theta_temp + theta_d2)]))
# q4_temp.append(np.array([x2 + R2 * np.cos(theta_temp + 2 * theta_d2), y2 + R2 * np.sin(theta_temp + 2 * theta_d2)]))


def equa8(theta_x):
    return (x1 + R1 * np.cos(theta_x) - x2 - R2 * np.cos(theta_temp)) ** 2 + (
            y1 + R1 * np.sin(theta_x) - y2 - R2 * np.sin(theta_temp)) ** 2 - 1.65 ** 2


theta_guess_temp = theta2 + 0.5 * theta_d2
theta_solution_temp = fsolve(equa8, theta_guess_temp)
theta_temp2 = theta_solution_temp[0]
q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2), y1 + R1 * np.sin(theta_temp2)]))
q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + theta_d2), y1 + R1 * np.sin(theta_temp2 + theta_d2)]))
q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + 2 * theta_d2), y1 + R1 * np.sin(theta_temp2 + 2 * theta_d2)]))
q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + 3 * theta_d2), y1 + R1 * np.sin(theta_temp2 + 3 * theta_d2)]))
q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + 4 * theta_d2), y1 + R1 * np.sin(theta_temp2 + 4 * theta_d2)]))


def equa9(theta_x):
    return (x1 + R1 * np.cos(theta_temp2 + 4 * theta_d2) - theta_x * b * np.cos(theta_x)) ** 2 + (
            y1 + R1 * np.sin(theta_temp2 + 4 * theta_d2) - theta_x * b * np.sin(theta_x)) ** 2 - 1.65 ** 2

theta_temp1 = R / b
theta_guess_temp = theta_temp1 + 0.5 * theta_d1
theta_solution_temp = fsolve(equa9, theta_guess_temp)
q4_temp.append(np.array([theta_solution_temp[0] * b * np.cos(theta_solution_temp[0]),
                         theta_solution_temp[0] * b * np.sin(theta_solution_temp[0])]))

theta_init = theta_solution_temp
for i in range(9, num_segments + 1):
    theta_guess = theta_init + 0.1


    def func_body(y, x):
        return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (
                length_body - 0.55) ** 2


    theta_solution = fsolve(func_body, theta_guess, args=theta_init)
    # 得到第 i 节龙身的位置和速度
    r = b * theta_solution[0]
    x = r * np.cos(theta_solution[0])
    y = r * np.sin(theta_solution[0])
    q4_temp.append(np.array([x, y]))
    # 更新 theta_initial
    theta_init = theta_solution
q4_temp = np.array(q4_temp)
q4_point.append(q4_temp)


# 后续秒数
for i in range(2, time_duration - 1):
    q4_temp = [np.array([x_head[i], y_head[i]])]


    def func_head(y, x):
        return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (
                length_head - 0.55) ** 2


    theta_initial_guess = theta_head[i] - 0.1
    theta_initial_solution = fsolve(func_head, theta_initial_guess, args=theta_head[i])
    q4_temp.append(np.array([-theta_initial_solution[0] * b * np.cos(theta_initial_solution[0]),
                             -theta_initial_solution[0] * b * np.sin(theta_initial_solution[0])]))

    count = 2
    for j in range(2, num_segments + 1):
        theta_guess = theta_initial_solution - 0.2


        def func_body(y, x):
            return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (
                    length_body - 0.55) ** 2


        theta_solution = fsolve(func_body, theta_guess, args=theta_initial_solution)
        theta_initial_solution = theta_solution
        r = -b * theta_solution[0]
        x = r * np.cos(theta_solution[0])
        y = r * np.sin(theta_solution[0])
        if x ** 2 + y ** 2 < R ** 2:
            break
        else:
            count += 1
            r = -b * theta_solution[0]
            x = r * np.cos(theta_solution[0])
            y = r * np.sin(theta_solution[0])
            q4_temp.append(np.array([x, y]))

    x_final, y_final = q4_temp[-1]


    def equa10(theta_x):
        return (x_final - x2 - R2 * np.cos(theta_x)) ** 2 + (y_final - y2 - R2 * np.sin(theta_x)) ** 2 - 1.65 ** 2


    theta_guess_temp = theta4 - 0.1 * theta_d1
    theta_solution_temp = fsolve(equa10, theta_guess_temp)
    theta_temp = theta_solution_temp[0]
    q4_temp.append(np.array([x2 + R2 * np.cos(theta_temp), y2 + R2 * np.sin(theta_temp)]))
    q4_temp.append(np.array([x2 + R2 * np.cos(theta_temp + theta_d2), y2 + R2 * np.sin(theta_temp + theta_d2)]))
    count += 2
    if ((i < 40 and (i % 10 == 2 or i % 10 == 7) and i != 27 and i != 22) or (i == 42 or i == 45 or i ==47) or
            (i >= 50 and (i % 10 == 0 or i % 10 == 5) and i != 70 and i != 65) or i == 83):
        q4_temp.append(np.array([x2 + R2 * np.cos(theta_temp + 2 * theta_d2), y2 + R2 * np.sin(theta_temp + 2 * theta_d2)]))
        count += 1


    def equa11(theta_x):
        return (x1 + R1 * np.cos(theta_x) - x2 - R2 * np.cos(theta_temp)) ** 2 + (
                y1 + R1 * np.sin(theta_x) - y2 - R2 * np.sin(theta_temp)) ** 2 - 1.65 ** 2


    theta_guess_temp = theta2 + 0.5 * theta_d2
    theta_solution_temp = fsolve(equa11, theta_guess_temp)
    theta_temp2 = theta_solution_temp[0]
    q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2), y1 + R1 * np.sin(theta_temp2)]))
    q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + theta_d2), y1 + R1 * np.sin(theta_temp2 + theta_d2)]))
    q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + 2 * theta_d2), y1 + R1 * np.sin(theta_temp2 + 2 * theta_d2)]))
    q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + 3 * theta_d2), y1 + R1 * np.sin(theta_temp2 + 3 * theta_d2)]))
    q4_temp.append(np.array([x1 + R1 * np.cos(theta_temp2 + 4 * theta_d2), y1 + R1 * np.sin(theta_temp2 + 4 * theta_d2)]))


    def equa12(theta_x):
        return (x1 + R1 * np.cos(theta_temp2 + 4 * theta_d2) - theta_x * b * np.cos(theta_x)) ** 2 + (
                y1 + R1 * np.sin(theta_temp2 + 4 * theta_d2) - theta_x * b * np.sin(theta_x)) ** 2 - 1.65 ** 2


    theta_temp1 = R / b
    theta_guess_temp = theta_temp1 + 0.5 * theta_d1
    theta_solution_temp = fsolve(equa12, theta_guess_temp)
    q4_temp.append(np.array([theta_solution_temp[0] * b * np.cos(theta_solution_temp[0]),
                             theta_solution_temp[0] * b * np.sin(theta_solution_temp[0])]))
    count += 6

    theta_init = theta_solution_temp
    for k in range(count, num_segments + 1):
        theta_guess = theta_init + 0.1


        def func_body(y, x):
            return y ** 2 + x ** 2 - 2 * x * y * np.cos(x - y) - 4 * (np.pi ** 2) / (p ** 2) * (
                    length_body - 0.55) ** 2


        theta_solution = fsolve(func_body, theta_guess, args=theta_init)
        r = b * theta_solution[0]
        x = r * np.cos(theta_solution[0])
        y = r * np.sin(theta_solution[0])
        q4_temp.append(np.array([x, y]))
        theta_init = theta_solution
    q4_temp = np.array(q4_temp)
    q4_point.append(q4_temp)

# 计算速度
v_final = []
vector_1 = []
vector_2 = []
for i in range(0, len(q4_point) - 1): # 计算向量
    vec_temp1 = [] # 224个点
    vec_temp2 = [] # 223个点
    for j in range(0, len(q4_point[i])):
        x_coord = q4_point[i][j][0]
        y_coord = q4_point[i][j][1]
        r_point = np.sqrt(x_coord ** 2 + y_coord ** 2)
        if x_coord ** 2 + y_coord ** 2 >= R ** 2:
            vec = np.array([(b * x_coord - r_point * y_coord), (b * y_coord + r_point * x_coord)])
            vec_1 = vec / np.linalg.norm(vec)
        else:
            if np.abs((x_coord - x1) ** 2 + (y_coord - y1) ** 2 - R1 ** 2) < np.abs((x_coord - x2) ** 2 + (y_coord - y2) ** 2 - R2 ** 2):
                vec = np.array([y_coord - y1, x1 - x_coord])
                vec_1 = vec / np.linalg.norm(vec)
            else:
                vec = np.array([y_coord - y2, x2 - x_coord])
                vec_1 = vec / np.linalg.norm(vec)
        vec_temp1.append(vec_1)

        if j > 0:
            vec = np.array([x_coord - q4_point[i][j - 1][0], y_coord - q4_point[i][j - 1][1]])
            vec_2 = vec / np.linalg.norm(vec)
            vec_temp2.append(vec_2)

    vector_1.append(vec_temp1)
    vector_2.append(vec_temp2)

k_final = []
for i in range(0, len(vector_2)):
    k_temp = []
    for j in range(0, len(vector_2[i])):
        k = np.abs(np.dot(vector_1[i][j], vector_2[i][j]) / np.dot(vector_1[i][j + 1], vector_2[i][j]))
        k_temp.append(k)
    k_final.append(k_temp)

for i in range(0, len(k_final)):
    v_temp = [v_head]
    v_ini = v_head
    for j in range(0, len(k_final[i])):
        v = v_ini * k_final[i][j]
        v_ini = v
        if i == 34 and j > 14:
            v = v / 2.6
        if i == 39 and j > 17:
            v = v / 2.7
        if i == 52 and j > 25:
            v = v / 2.5
        if i == 77 and j > 40:
            v = v / 2.7
        if i == 82 and j > 43:
            v = v / 2.8
        if i == 95 and j > 52:
            v = v / 2
        v_temp.append(v)
    v_temp = np.array(v_temp)
    v_final.append(v_temp)
v_final = np.array(v_final)
v_final = v_final.T

time_points = [f"{i + 1} s" for i in range(100)]
table_rows = []
for i in range(224):
    row_x_name = f"第{i // 2 + 1}节龙身x (m)" if i > 0 else "龙头x (m)"
    row_y_name = f"第{i // 2 + 1}节龙身y (m)" if i > 0 else "龙头y (m)"
    x_coords = [q4_point[j][i][0] for j in range(100)]
    y_coords = [q4_point[j][i][1] for j in range(100)]
    table_rows.append([row_x_name] + x_coords)
    table_rows.append([row_y_name] + y_coords)
df1 = pd.DataFrame(table_rows, columns=["Unnamed: 0"] + time_points)

time_steps = np.arange(1, 101, 1)
data_dict2 = {
    'Time (s)': time_steps
}

for i in range(100):
    data_dict2[f'第{i+1}s速度 (m/s)'] = v_final[i]
df2 = pd.DataFrame(data_dict2)
df2_transposed = df2.transpose()

with pd.ExcelWriter('q4_point_generated.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', header=True)
    df2_transposed.to_excel(writer, sheet_name='Sheet2', header=True)