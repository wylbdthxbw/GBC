# -*- coding: utf-8 -*-
import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from GBC.GranularBallGeneration import get_gb_division_x

def visualize(h, color,title=''):
    if h.shape[1] > 2:
        z = TSNE(n_components=2).fit_transform(h)
    else:
        z = h
    plt.figure(figsize=(10,10))
    plt.title(title)
    z_color = np.insert(z, 2, color, axis=1)
    label_max = int(max(color))
    for label in range(0,label_max+1):
        plt.scatter(z[z_color[:,-1] == label, 0], z[z_color[:,-1] == label, 1], s=7, cmap='Set2',label = 'cluster {}'.format(label+1))
    plt.legend(loc = 'upper right')
    plt.show()

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY

def merge_sets(sets):
    '''
    合并有相同元素的集合
    Args:
        sets:

    Returns:

    '''
    uf = UnionFind()
    for set_ in sets:
        it = iter(set_)
        first_elem = next(it)
        for elem in it:
            uf.union(first_elem, elem)

    result = {}
    for set_ in sets:
        for elem in set_:
            root = uf.find(elem)
            if root not in result:
                result[root] = set()
            result[root].add(elem)

    return list(result.values())

class GB():
    X = None
    def __init__(self,idx_list):
        self.idx_list = idx_list
        self.center = np.mean(GB.X[self.idx_list,:],axis=0)

    def add_x(self,x):
        self.idx_list.append(x)
        self.center = np.mean(GB.X[self.idx_list,:],axis=0)

class GB_senior():

    S = None
    def __init__(self,idx_list):
        self.idx_list = idx_list

    @staticmethod
    def get_gb_gb_d_senior(gb1,gb2):
        max_s = 0
        for g1 in gb1.idx_list:
            for g2 in gb2.idx_list:
                max_s = max(GB_senior.S[g1,g2],max_s)
        return max_s

def get_avg_n_div_maxr(gb_list):
    '''
    用 n / maxr 当做判定噪声球的条件，用来过滤噪声粒球
    Args:
        gb_list:

    Returns:
    '''
    avg_n_div_maxr = sum([len(x) / get_radius(x)  for x in gb_list]) / len(gb_list)
    return avg_n_div_maxr

def get_radius(gb):
    '''
    返回粒球的最大半径
    Args:
        gb:
    Returns:
    '''
    center = gb.mean(0)
    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius

def get_S_matrix_drr_xia(gb_list):
    '''
    得到相似度矩阵，距离越小相似度越大
    Args:
        gb_list:

    Returns:

    '''

    means = [np.mean(gb, axis=0) for gb in gb_list]
    n = len(gb_list)
    r_list = []
    for idx,gb in enumerate(gb_list):
        points = gb
        centroid = means[idx]

        distances = np.linalg.norm(points - centroid, axis=1)

        max_distance = np.max(distances)
        r_list.append(max_distance)
    rs = np.array(r_list)
    sub1 = np.array([np.repeat(element, n) for element in rs])
    sub2 = sub1.T
    means = np.array(means)
    Means = np.linalg.norm(means[:, np.newaxis] - means, axis=2)
    d_matrix = Means - sub1 - sub2
    min_d = np.min(d_matrix)
    if min_d < 0:
        min_d = min_d * (-1)
        d_matrix += min_d * 2
    np.fill_diagonal(d_matrix, 0)

    raw_gb_gb_d_matrix = 1 / d_matrix
    raw_gb_gb_d_matrix[np.isinf(raw_gb_gb_d_matrix)] = 0
    return raw_gb_gb_d_matrix

def divide_ball_GBC_y(X,K,detaile = False):
    '''
    先根据GBC中的粒球划分生成粒球，然后按照粒球的最近邻优先的距离度量连接粒球。
    Args:
        X: 数据
        K: 聚类簇数
        detaile: 聚类细节
    Returns:
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaler = scaler.fit_transform(X)
    detaile_list = []

    minimum_ball = 2  # 一般为2，重叠时调大
    percent_avg = 0.2  # 越大忽略的越多

    start_time = datetime.datetime.now()
    gb_list = get_gb_division_x(X, False)
    end_time = datetime.datetime.now()
    consume_time = (end_time - start_time)
    print('split consume time is-----', consume_time)

    gb_list = [x for x in gb_list if len(x) != 0]
    print("gb_list len:", len(gb_list))
    # 根据粒球内的样本点数去噪声
    noise_list1 = [x for x in gb_list if x.shape[0] < minimum_ball]
    # 去噪过后的 gb_list
    gb_list = [x for x in gb_list if x.shape[0] >= minimum_ball]

    avg_n_div_maxr = get_avg_n_div_maxr(gb_list=gb_list)

    # 根据粒球的稀疏度去噪声
    noise_list2 = [x for x in gb_list if len(x) / get_radius(x) < percent_avg * avg_n_div_maxr]
    # 去噪过后的 gb_list
    gb_list = [x for x in gb_list if len(x) / get_radius(x) >= percent_avg * avg_n_div_maxr]
    noise_list = noise_list1 + noise_list2
    # visualize(np.vstack(gb_list), [0] * len(np.vstack(gb_list)), 'deoverlap..')

    noise_index = 1

    S = get_S_matrix_drr_xia(gb_list) # 粒球的相似度矩阵
    GB_senior.S = S
    # 将每个粒球都映射成一个索引
    dict_raw = {idx:gb for idx ,gb in enumerate(gb_list)}

    sets = []
    for i, x in enumerate(S):
        s_t = set()
        s_t.add(i)
        s_t.add(x.argmax())
        sets.append(s_t)

    # 得到第一次合并后的粒球集合，此时每个粒球都是高级粒球
    merged = merge_sets(sets)
    dict1 = {}
    i = 0
    for st in merged:
        dict1[i] = st
        i += 1

    GB_list = [GB_senior(list(x)) for x in merged]
    detaile_list.append(dict1)
    epoch = 0
    flag_last = False
    while len(GB_list) > K:
        # 构建新簇的相似度矩阵
        S = []
        for gb1 in GB_list:
            tmp = []
            for gb2 in GB_list:
                tmp.append(GB_senior.get_gb_gb_d_senior(gb1,gb2))
            S.append(tmp)
        S = np.array(S)
        np.fill_diagonal(S, 0)

        sets = []
        if epoch < noise_index: # 所有粒球都合并
            for i, x in enumerate(S):
                s_t = set()
                s_t.add(i)
                s_t.add(x.argmax())
                sets.append(s_t)
        else: # 仅合并 len(S) - K 个粒球
            i_argmax_val_list = []
            for i, x in enumerate(S):
                tmp = []
                tmp.append(i)
                tmp.append(x.argmax())
                tmp.append(S[i][x.argmax()])
                i_argmax_val_list.append(tmp)
            i_argmax_val_list.sort(key = lambda x:x[2],reverse=True)
            for i in range(len(S) - K):
                s_t = set()
                s_t.add(i_argmax_val_list[i][0])
                s_t.add(i_argmax_val_list[i][1])
                sets.append(s_t)
            for i in range(len(S) - K, len(S)):
                s_t = set()
                s_t.add(i_argmax_val_list[i][0])
                sets.append(s_t)

        merged = merge_sets(sets)
        # 映射回原始粒球
        merged_all = []
        for st in merged:
            t_set = set()
            for x in st:
                t_set = t_set.union(dict1[x])
            merged_all.append(t_set)
        merged = merged_all

        epoch += 1
        last_dict = dict1
        flag_last = True
        # 记录当前粒球划分
        dict1 = {}
        i = 0
        for st in merged:
            dict1[i] = st
            i += 1
        detaile_list.append(dict1)
        GB_list = [GB_senior(list(x)) for x in merged]
        if last_dict == dict1:
            break

    if flag_last == False:
        last_dict = {k:{k} for k,v in enumerate(gb_list)}

    first_GB_list_len = len(GB_list)
    if first_GB_list_len <= K:
        merged = last_dict.values()
        # 退回上一次划分
        GB_list = [GB_senior(list(x)) for x in merged]
        dict1 = last_dict
        while len(GB_list) > K:
            S = []
            for gb1 in GB_list:
                tmp = []
                for gb2 in GB_list:
                    tmp.append(GB_senior.get_gb_gb_d_senior(gb1, gb2))
                S.append(tmp)
            S = np.array(S)
            np.fill_diagonal(S, 0)
            max_index = np.argmax(S)
            max_index_2d = np.unravel_index(max_index, S.shape)
            merged = []
            # merged.append(set({max_index_2d[0], max_index_2d[1]}))
            merged.append({max_index_2d[0], max_index_2d[1]})
            one_list = [x for x in range(S.shape[0]) if x not in {max_index_2d[0], max_index_2d[1]}]
            for x in one_list:
                merged.append({x})
            merged_all = []
            for st in merged:
                t_set = set()
                for x in st:
                    t_set = t_set.union(dict1[x])
                merged_all.append(t_set)
            merged = merged_all

            last_dict = dict1
            dict1 = {}
            i = 0
            for st in merged:
                dict1[i] = st
                i += 1
            detaile_list.append(dict1)
            GB_list = [GB_senior(list(x)) for x in merged]
            print('\r{}'.format(len(GB_list)),end='')
        detaile_list.append(dict1)

    if detaile:
        out = []
        for dict1 in detaile_list:
            dict_out = {}
            for k, v in dict1.items():
                tmp = []
                for x in v:
                    tmp.extend(list(dict_raw[x]))
                dict_out[k] = tmp
            out.append(dict_out)
        return out

    # 将粒球划分映射为原始数据
    dict_out = {}
    for k, v in dict1.items():
        tmp = []
        for x in v:
            tmp.extend(list(dict_raw[x]))
        dict_out[k] = tmp

    gb_idx_lable = {}
    for key, values in dict1.items():
        for value in values:
            gb_idx_lable[value] = key

    # 噪声粒球就近分配到最近粒球所在簇
    noise_lable_list = []
    centers_gb_list = np.array([np.mean(gb, axis=0) for gb in gb_list])
    for gb_noise in noise_list:
        center_noise = np.mean(gb_noise, axis=0)
        distance = np.linalg.norm(centers_gb_list - center_noise, axis=1)
        arg_min = np.argmin(distance)
        noise_lable_list.append(gb_idx_lable[arg_min])
    for gb_noise, lable in zip(noise_list, noise_lable_list):
        dict_out[lable].extend(list(gb_noise))

    # 返回X对应的标签y
    map_y = {}
    for label, vals in dict_out.items():
        for val in vals:
            map_y[tuple(val)] = label
    y_ball = []
    for x in X_scaler:
        y_ball.append(map_y[tuple(x)])

    return y_ball

if __name__ == '__main__':

    matrix = np.array([[1,1],[1,2],[1.5,2],[2,2]])
    matrix2 = np.array([[3,3],[7,8]])
    divide_ball_GBC_y(matrix,2)
    pass