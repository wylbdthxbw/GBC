# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


class GB:
    def __init__(self, data, label):
        # 粒球内元素
        self.data = data
        # 粒球中心
        self.center = self.data.mean(0)
        # 粒球半径：粒球中心到点的最大距离
        self.radius = self.get_radius()
        self.flag = 0
        # 父粒球标记
        self.label = label
        # 元素数量
        self.num = len(data)
        self.out = 0
        # 子粒球个数(包含自己)
        self.size = 1
        self.overlap = 0 
        self.hardlapcount = 0 
        self.softlapcount = 0

    def get_radius(self):
        return max(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)


class UF:
    def __init__(self, len):
        self.parent = [0] * len
        self.size = [0] * len
        self.count = len

        for i in range(0, len):
            self.parent[i] = i
            self.size[i] = 1

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return
        if self.size[rootP] > self.size[rootQ]:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        else:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        self.count = self.count - 1

    def connected(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ

    def count(self):
        return self.count


# 粒球划分判断
def division(hb_list, n):
    '''
    对于样本点大于等于8的粒球采取自适应分裂，而小于8的粒球默认质量达标。实际上样本点小于8的粒球有很多不达标。
    Args:
        hb_list:
        n:

    Returns:

    '''
    gb_list_new = []
    # 粒球个数
    k = len(hb_list)
    # 遍历粒球
    for hb in hb_list:
        if len(hb) == 1:
            gb_list_new.append(hb)
            continue

        if len(hb) >= 8:
            # 根据距离最远的两个点作为粒球中心划分
            ball_1, ball_2 = spilt_ball(hb)
            # 计算质量
            DM_parent = get_DM(hb)
            DM_child_1 = get_DM(ball_1)
            DM_child_2 = get_DM(ball_2)
            # 计算占比
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            # 质量占比之和
            w_child = (w1 * DM_child_1 + w2 * DM_child_2)
            # 粒球数量小于8个的时候，输出质量和分类情况
            # if k < 8:
            #     print('parent', DM_parent)
            #     print('len-child_1', len(ball_1))
            #     print('child-1', DM_child_1)
            #     print('len-child_2', len(ball_2))
            #     print('child-2', DM_child_2)
            # 划分标准
            t1 = ((DM_child_1 < DM_parent) & (DM_child_2 < DM_parent))  # division standard 1
            t2 = (w_child < DM_parent)                                # division standard 2
            t3 = (len(ball_1) >= 3) & (len(ball_2) >= 3)                   # min data point control
            # 划分后，质量减小就确认划分，否则不划分
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(hb)
        # 粒球中点的个数小于8，不划分
        else:
            gb_list_new.append(hb)
    return gb_list_new


def division2(hb_list, n):
    '''
    自适应分裂
    Args:
        hb_list:
        n:
    Returns:
    '''
    gb_list_new = []
    # 粒球个数
    k = len(hb_list)
    # 遍历粒球
    for hb in hb_list:
        if len(hb) == 1:
            gb_list_new.append(hb)
            continue

        # 根据距离最远的两个点作为粒球中心划分
        ball_1, ball_2 = spilt_ball(hb)
        # 计算质量
        DM_parent = get_sparsity(hb)
        DM_child_1 = get_sparsity(ball_1)
        DM_child_2 = get_sparsity(ball_2)
        # 计算占比
        w = len(ball_1) + len(ball_2)
        w1 = len(ball_1) / w
        w2 = len(ball_2) / w
        # 质量占比之和
        w_child = (w1 * DM_child_1 + w2 * DM_child_2)
        # 划分标准
        t1 = ((DM_child_1 < DM_parent) & (DM_child_2 < DM_parent))  # division standard 1
        t2 = (w_child < DM_parent)                                # division standard 2
        t3 = (len(ball_1) >= 3) & (len(ball_2) >= 3)                   # min data point control
        # 划分后，质量减小就确认划分，否则不划分
        if t2:
            gb_list_new.extend([ball_1, ball_2])
        else:
            gb_list_new.append(hb)
    return gb_list_new


# 粒球划分
def spilt_ball(data):
    ball1 = []
    ball2 = []
    n, m = data.shape
    # 数据矩阵转置
    X = data.T
    # 数据矩阵*数据矩阵的转置
    G = np.dot(X.T, X)
    # np.diag(G)取对角线元素
    # 将数据的平方和作为一维向量纵向平铺n次形成原数据大小的矩阵
    H = np.tile(np.diag(G), (n, 1))
    """
    设data = [[a, b], [c, d]] 
    G = [[a**2 + b**2, a * c + b * d],
         [c * a + d * b, c**2 + d**2]]
    H = [[a**2 + b**2, c**2 + d**2],
         [a**2 + b**2, c**2 + d**2]]
    D为点与点之间的距离矩阵
    D = [[0, c**2 + d**2 + a**2 + b**2 - (a * c + b * d) * 2], = [[0, ((a - c)**2 + (b - d)**2)**0.5],
         [a**2 + b**2 + c**2 + d**2 - (c * a + d * b) * 2, 0]]    [((a - c)**2 + (b - d)**2)**0.5, 0]]
    """
    D = np.sqrt(H + H.T - G * 2)
    # 返回D中最大元素的行列表r和列列表c
    r, c = np.where(D == np.max(D))
    # 得到距离最远的两个点的下标
    r1 = r[1]
    c1 = c[1]
    # 根据距离这两个点的远近划分粒球
    for j in range(0, len(data)):
        if D[j, r1] < D[j, c1]:
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def get_sparsity(hb):
    num = hb.shape[0]
    dim = hb.shape[1]
    # 数据均值
    center = hb.mean(0)
    # 均值减去每个数据
    diffMat = np.tile(center, (num, 1)) - hb
    # 差的平方
    sqDiffMat = diffMat ** 2
    # 差的平方和
    sqDistances = sqDiffMat.sum(axis=1)
    # 差的平方和开根号：数据均值到每个数据的距离
    distances = sqDistances ** 0.5
    # 差的平方和开根号的最大值：最大距离
    radius = max(distances)

    if num != 1:
        sparsity = num / radius ** dim
    else:
        sparsity = radius
    return sparsity


# 平均距离作为粒球质量
def get_DM(hb):
    num = len(hb)
    # 孤立点直接返回0
    if num == 1:
        return 0
    # 数据均值
    center = hb.mean(0)
    # 均值减去每个数据
    diffMat = np.tile(center, (num, 1)) - hb
    # 差的平方
    sqDiffMat = diffMat ** 2
    # 差的平方和
    sqDistances = sqDiffMat.sum(axis=1)
    # 差的平方和开根号：数据均值到每个数据的距离
    distances = sqDistances ** 0.5
    # 差的平方和开根号的最大值：最大距离
    radius = max(distances)

    # 差的平方和开根号之和：距离之和
    sum_radius = 0
    for i in distances:
        sum_radius = sum_radius + i

    # 每个数据到数据均值的平均距离
    mean_radius = sum_radius / num
    # 数据维度
    dimension = len(hb[0])
    if mean_radius != 0:
        DM = sum_radius / num
    else:
        DM = radius
    return DM


def get_radius(hb):
    num = len(hb)
    center = hb.mean(0)
    diffMat = np.tile(center, (num, 1)) - hb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)
    return radius


def plot_dot(data,title = ''):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], s=7, c="#314300", linewidths=2, alpha=0.6, marker='o', label='data point')
    plt.title(title)
    plt.legend()


def hb_plot(gbs, noise):
    color = { 
        0: '#707afa',             
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#e96d29',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',
        25: '#ff8444'}
    label_c = {
        0: 'cluster-1',
        1: 'cluster-2',
        2: 'cluster-3',
        3: 'cluster-4',
        4: 'cluster-5',
        5: 'cluster-6',
        6: 'cluster-7',
        7: 'cluster-8',
        8: 'cluster-9',
        9: 'cluster-10',
        10: 'cluster-11',
        11: 'cluster-12',
        12: 'cluster-13',
        13: 'cluster-14',
        14: 'cluster-15',
        15: 'cluster-16',
        16: 'cluster-17',
        17: 'cluster-18',
        18: 'cluster-19',
        19: 'cluster-20',
        20: 'cluster-21',
        21: 'cluster-22',
        22: 'cluster-23',
        23: 'cluster-24',
        24: 'cluster-25'}

    plt.figure(figsize=(10,10))  
    label_num = {}
    for i in range(0, len(gbs)):
        label_num.setdefault(gbs[i].label,0)
        label_num[gbs[i].label] = label_num.get(gbs[i].label) + len(gbs[i].data)
        
    label = set()
    for key in label_num.keys():
        label.add(key)
    list = []
    for i in range(0, len(label)):
        list.append(label.pop())
   
    for i in range(0, len(list)):
        if list[i] == -1:
            list.remove(-1)
            break
    
    for i in range(0, len(list)):
        for key in gbs.keys():
            if gbs[key].label == list[i]:
                plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
                            marker='o', label=label_c[i])
                break

    for key in gbs.keys():
        for i in range(0, len(list)):
            if gbs[key].label == list[i]:
                plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
                            marker='o')
    if len(noise) > 0:
        plt.scatter(noise[:, 0], noise[:, 1], s=40, c='black', linewidths=2, alpha=1, marker='x', label='noise')

    for key in gbs.keys():
        for i in range(0, len(list)):
            if gbs[key].label == -1:
                plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=40, c='black', linewidths=2, alpha=1,
                            marker='x')

    plt.legend(loc=1, fontsize=12)
    plt.show()


# 修改后的 draw_ball 函数，现在可以显示稀疏度
def draw_ball_show_dm(hb_list, sparsity_measures, title=''):
    for data, sparsity in zip(hb_list, sparsity_measures):
        if  1< len(data) < 10:
            # 多个点是圆
            center = data.mean(0)
            radius = np.max(np.sqrt(np.sum((data - center) ** 2, axis=1)))
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, ls='-', color='black', lw=0.7)

        elif len(data)>10:
            # 多个点是圆
            center = data.mean(0)
            radius = np.max(np.sqrt(np.sum((data - center) ** 2, axis=1)))
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, ls='-', color='black', lw=0.7)
            # 在球的中心绘制红星
            plt.plot(center[0], center[1], marker='*', color='red', markersize=10)
            # 显示稀疏度
            plt.text(center[0], center[1], f'Density: {sparsity:.2f}', fontsize=8, ha='center')
        else:
            # 单个点是*
            plt.plot(data[0][0], data[0][1], marker='*', color='#0000EF', markersize=10)

    plt.plot([], [], ls='-', color='black', lw=1.2, label='hyper-ball boundary')
    # plt.scatter([], [], c="#314300", marker='o', label='data point')  # 用于显示图例的假散点图
    plt.legend(loc=1)
    plt.title(title)
    plt.show()


def draw_ball(hb_list,title=''):
    # 合并数组列表中的数组
    plt_data = np.concatenate(hb_list)
    for data in hb_list:
        if len(data) > 1:
            # 多个点是圆
            center = data.mean(0)
            radius = np.max((((data - center) ** 2).sum(axis=1) ** 0.5))
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, ls='-', color='black', lw=0.7)
            
        else:
            x = data[0][0]
            y = data[0][1]
            # 单个点是*
            plt.plot(x,y, marker='*', color='#0000EF', markersize=3)
    plt.plot(x, y, ls='-', color='black', lw=1.2, label='hyper-ball boundary')
    data = plt_data
    # plt.scatter(data[:, 0], data[:, 1], s=0.5, c="#314300", linewidths=5, alpha=0.6, marker='o', label='data point')
    plt.legend(loc=1)
    # plt.title(title)
    plt.show()
    # plt.savefig('gb_' + title + '.png')



def connect_ball(hb_list, noise, c_count):
    # 记录粒球类{0:hb1, 1:hb2, ..., n-1:hbn}
    hb_cluster = {}
    for i in range(0, len(hb_list)):
        hb = GB(hb_list[i], i)
        hb_cluster[i] = hb

    # 粒球数据半径之和
    radius_sum = 0
    # 粒球数据个数之和
    num_sum = 0
    # 粒球个数
    hb_len = 0
    # radius_sum = 0
    # num_sum = 0
    for i in range(0, len(hb_cluster)):
        if hb_cluster[i].out == 0:
            hb_len = hb_len + 1
            radius_sum = radius_sum + hb_cluster[i].radius
            num_sum = num_sum + hb_cluster[i].num

    for i in range(0, len(hb_cluster)-1):
        if hb_cluster[i].out != 1:
            # 粒球数据均值作为粒球中心
            center_i = hb_cluster[i].center
            radius_i = hb_cluster[i].radius
            for j in range(i + 1, len(hb_cluster)):
                if hb_cluster[j].out != 1:
                    center_j = hb_cluster[j].center
                    radius_j = hb_cluster[j].radius
                    # 粒球中心距离
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if (dis <= radius_i + radius_j) & ((hb_cluster[i].hardlapcount == 0) & (hb_cluster[j].hardlapcount
                                                                                            == 0)):
                        # 由于两个的点是噪声球，所以纳入重叠统计
                        hb_cluster[i].overlap = 1
                        hb_cluster[j].overlap = 1
                        hb_cluster[i].hardlapcount = hb_cluster[i].hardlapcount + 1
                        hb_cluster[j].hardlapcount = hb_cluster[j].hardlapcount + 1

    hb_uf = UF(len(hb_list))
    for i in range(0, len(hb_cluster)-1):
        if hb_cluster[i].out != 1:
            center_i = hb_cluster[i].center
            radius_i = hb_cluster[i].radius
            for j in range(i + 1, len(hb_cluster)):
                if hb_cluster[j].out != 1:
                    center_j = hb_cluster[j].center
                    radius_j = hb_cluster[j].radius
                    max_radius = max(radius_i, radius_j)
                    min_radius = min(radius_i, radius_j)
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    # 公式
                    if c_count == 1:
                        dynamic_overlap = dis < radius_i + radius_j + 1 * min_radius / (min(hb_cluster[i].hardlapcount,
                                                                                        hb_cluster[j].hardlapcount) + 1)
                    if c_count == 2:
                        dynamic_overlap = dis <= radius_i + radius_j + 1 * max_radius / \
                                          (min(hb_cluster[i].hardlapcount, hb_cluster[j].hardlapcount) + 1)
                    num_limit = ((hb_cluster[i].num > 2) & (hb_cluster[j].num > 2))
                    if dynamic_overlap & num_limit:
                        hb_cluster[i].flag = 1
                        hb_cluster[j].flag = 1
                        hb_uf.union(i, j)
                    if dis <= radius_i + radius_j + max_radius:
                        hb_cluster[i].softlapcount = 1
                        hb_cluster[j].softlapcount = 1
    # 更新最终父粒球标记对象
    for i in range(0, len(hb_cluster)):
        k = i
        if hb_uf.parent[i] != i:
            while hb_uf.parent[k] != k:
                k = hb_uf.parent[k]
        hb_uf.parent[i] = k
    # 根据标记对象结果更新粒球属性
    for i in range(0, len(hb_cluster)):
        hb_cluster[i].label = hb_uf.parent[i]
        hb_cluster[i].size = hb_uf.size[i]
    # 收集最终父粒球结果集
    label_num = set()
    for i in range(0, len(hb_cluster)):
        label_num.add(hb_cluster[i].label)
    # 结果转换为列表
    list = []
    for i in range(0, len(label_num)):
        list.append(label_num.pop())
    
    for i in range(0, len(hb_cluster)):
        if (hb_cluster[i].hardlapcount == 0) & (hb_cluster[i].softlapcount == 0):
            hb_cluster[i].flag = 0

    for i in range(0, len(list)):
        count_ball = 0
        count_data = 0 
        list1 = []
        for key in range(0, len(hb_cluster)):
            if hb_cluster[key].label == list[i]:
                count_ball += 1
                count_data += hb_cluster[key].num
                list1.append(key)
        while count_ball < 6:
            for j in range(0, len(list1)):
                hb_cluster[list1[j]].flag = 0
            break
        
    for i in range(0, len(hb_cluster)):
        distance = np.sqrt(2)            
        if hb_cluster[i].flag == 0:
            for j in range(0, len(hb_cluster)):
                if hb_cluster[j].flag == 1:
                    center = hb_cluster[i].center
                    center2 = hb_cluster[j].center
                    dis = ((center - center2)**2).sum(axis=0)**0.5 - (hb_cluster[i].radius + hb_cluster[j].radius)
                    if dis < distance:
                        distance = dis
                        hb_cluster[i].label = hb_cluster[j].label
                        hb_cluster[i].flag = 2
            for k in range(0, len(noise)):
                center = hb_cluster[i].center
                dis = ((center - noise[k]) ** 2).sum(axis=0) ** 0.5
                if dis < distance:
                    distance = dis
                    hb_cluster[i].label = -1
                    hb_cluster[i].flag = 2
                    
    label_num = set()
    for i in range(0, len(hb_cluster)):
        label_num.add(hb_cluster[i].label)
    return hb_cluster


# 划分大半径粒球
def normalized_ball(hb_list, radius_detect):
    hb_list_temp = []    
    for hb in hb_list:
        # 只有一个点的粒球
        if len(hb) < 2:
            hb_list_temp.append(hb)
        else:
            ball_1, ball_2 = spilt_ball(hb)
            # 粒球中心到点的最大距离大于2倍直接半径，划分，否则不划分
            if get_radius(hb) <= 2 * radius_detect:
                hb_list_temp.append(hb)
            else:
                hb_list_temp.extend([ball_1, ball_2])
    
    return hb_list_temp


def normalized_ball_2(hb_cluster, radius_mean, list1):
    hb_list_temp = []
    for i in range(0, len(radius_mean)):
        for key in hb_cluster.keys():
            if hb_cluster[key].label == list1[i]:
                if hb_cluster[key].num < 2:
                    hb_list_temp.append(hb_cluster[key].data)
                else:
                    ball_1, ball_2 = spilt_ball(hb_cluster[key].data)
                    if hb_cluster[key].radius <= 2 * radius_mean[i]:
                        hb_list_temp.append(hb_cluster[key].data)
                    else:
                        hb_list_temp.extend([ball_1, ball_2])
    return hb_list_temp

def load_txt_dataset(path):
    # 读取文本文件
    with open(path, 'r') as file:
        lines = file.readlines()
    # 将数据转换为 NumPy 数组
    data = np.array([[float(num) for num in line.split()] for line in lines])
    return data

def split_twice(hb_list):
    gb_list_new = []
    sparsity_list = []
    for hb in hb_list:
        center = hb.mean(0)
        r_max = max(np.linalg.norm(hb - center, axis=1))
        sparsity = r_max / len(hb)
        sparsity_list.append([sparsity,hb])

    sparsity_list.sort(key=lambda x:x[0],reverse=True)
    for idx,tmp in enumerate(sparsity_list):
        sparsity = tmp[0]
        hb = tmp[1]
        if idx < len(sparsity_list) / 10:
            # 根据距离最远的两个点作为粒球中心划分
            ball_1, ball_2 = spilt_ball(hb)
            gb_list_new.extend([ball_1, ball_2])
        else:
            gb_list_new.append(hb)
    return gb_list_new


def get_gb_division(X):
    '''
    根据GBC里面的分裂粒球方式得到粒球划分，并可视化
    Args:
        X: 特征
        y: 标签

    Returns: 划分ndarray数组

    '''
    data = X
    # y = np.array([1] * data.shape[0])

    # 特征缩放(标准化)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    hb_list_temp = [data]
    # np.shape(hb_list_temp) = (1, N(数据个数), 2)
    row = np.shape(hb_list_temp)[0]
    col = np.shape(hb_list_temp)[1]
    # 初始数据个数
    n = row * col

    # 粒球自顶向下划分至元素个数小于8或划分后质量不变小
    while 1:
        ball_number_old = len(hb_list_temp)
        hb_list_temp = division(hb_list_temp, n)
        ball_number_new = len(hb_list_temp)
        if ball_number_new == ball_number_old:
            break

    radius = []
    for hb in hb_list_temp:
        # 统计元素个数大于1的粒球半径，即粒球中心到点的最远距离
        if len(hb) >= 2:
            radius.append(get_radius(hb))

    # 半径的中位数
    radius_median = np.median(radius)
    # 半径的平均值
    radius_mean = np.mean(radius)
    # 中位数和平均值之中较大的：直接半径
    radius_detect = max(radius_median, radius_mean)

    # 划分大半径的粒球
    while 1:
        ball_number_old = len(hb_list_temp)
        hb_list_temp = normalized_ball(hb_list_temp, radius_detect)
        ball_number_new = len(hb_list_temp)
        if ball_number_new == ball_number_old:
            break

    plot_dot(data, '552')
    draw_ball(hb_list_temp, 'first')

    hb_list_temp = split_twice(hb_list_temp)

    plot_dot(data, '551')
    draw_ball(hb_list_temp, 'second')
    return hb_list_temp

def main():
    # keys=['D1','D2','D3','D4','D5','D6','D7','D8']
    keys = ['D1']
    for d in range(len(keys)):
        # df = pd.read_csv("./synthetic//" + keys[d] + ".csv", header=None)
        # data = df.values
        # # 数据去重
        # data = np.unique(data, axis=0)

        # 读取txt数据集
        path = r'F:\Python\dataset\changzhi_data_set\no_noise\t4.txt'
        data = load_txt_dataset(path)
        y = np.array([1] * data.shape[0])

        start_time = datetime.datetime.now()

        # 特征缩放(标准化)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)

        hb_list_temp = [data]
        # np.shape(hb_list_temp) = (1, N(数据个数), 2)
        row = np.shape(hb_list_temp)[0]
        col = np.shape(hb_list_temp)[1]
        # 初始数据个数
        n = row*col

        # 粒球自顶向下划分至元素个数小于8或划分后质量不变小
        i = 0
        while 1:
            i += 1
            ball_number_old = len(hb_list_temp)
            hb_list_temp = division(hb_list_temp, n)
            ball_number_new = len(hb_list_temp)

            # plot_dot(data, '583')
            # draw_ball(hb_list_temp, '{} {}'.format(i,len(hb_list_temp)))
            if ball_number_new == ball_number_old:
                break
        tt = [x for x in hb_list_temp if len(x) >= 8]
        print(len(tt))

        plot_dot(data, '583')
        draw_ball(hb_list_temp, 'first')


        radius = []
        for hb in hb_list_temp:
            # 统计元素个数大于1的粒球半径，即粒球中心到点的最远距离
            if len(hb) >= 2:
                radius.append(get_radius(hb))

        # 半径的中位数
        radius_median = np.median(radius)
        # 半径的平均值
        radius_mean = np.mean(radius)
        # 中位数和平均值之中较大的：直接半径
        radius_detect = max(radius_median, radius_mean)

        # 划分大半径的粒球
        while 1:
            ball_number_old = len(hb_list_temp)
            hb_list_temp = normalized_ball(hb_list_temp, radius_detect)
            ball_number_new = len(hb_list_temp)
            if ball_number_new == ball_number_old:
                break
        draw_ball(hb_list_temp,'548')

        end_time = datetime.datetime.now()

        consume_time = (end_time - start_time)

        print('split consume time is-----', consume_time)
        # 粒球连接
        noise = []
        hb_cluster = connect_ball(hb_list_temp, noise, 1)

        label_num = {}
        for i in range(0, len(hb_cluster)):
            label_num.setdefault(hb_cluster[i].label, 0)
            label_num[hb_cluster[i].label] = label_num.get(hb_cluster[i].label) + len(hb_cluster[i].data)
        # 统计数据个数大于10的父粒球
        label = set()
        for key in label_num.keys():
            if label_num[key] > 10:
                label.add(key)
        list1 = []
        for i in range(0, len(label)):
            list1.append(label.pop())
        # 粒球结果统计
        radius_detect = [0]*len(list1)
        count_cluster_num = [0]*len(list1)
        radius_mean = [0]*len(list1)
        for key in hb_cluster.keys():
            for i in range(0, len(list1)):
                if hb_cluster[key].label == list1[i]:
                    radius_detect[i] = radius_detect[i] + hb_cluster[key].radius
                    count_cluster_num[i] = count_cluster_num[i] + 1
        # 同类粒球平均半径
        for i in range(0, len(list1)):
            radius_mean[i] = radius_detect[i] / count_cluster_num[i]

        while 1:
            ball_number_old = len(hb_list_temp)
            hb_list_temp = normalized_ball_2(hb_cluster, radius_mean, list1)
            ball_number_new = len(hb_list_temp)
            if ball_number_new == ball_number_old:
                break
        plot_dot(data,'583')

        draw_ball(hb_list_temp,'584')
        gb_list_final = hb_list_temp
        noise = []

        gb_list_cluster = connect_ball(gb_list_final, noise, 2)  # 最终聚类结果，第5次画图

        end_time = datetime.datetime.now()

        consume_time = (end_time - start_time)

        print('all consume time is-----', consume_time)
        hb_plot(gb_list_cluster, noise)
        print("clustering complete")


if __name__ == '__main__':
    main()  # -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 17:36:56 2022

@author: hp
"""

