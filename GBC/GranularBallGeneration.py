import time

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
from GBC.HyperBallClustering_acceleration_note import plot_dot, draw_ball, draw_ball_show_dm
from sklearn.cluster import KMeans
from scipy.special import gamma


def calculate_center_radius(gb):
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDistances = np.sum(diffMat ** 2, axis=1)
    radius = max(sqDistances ** 0.5)
    return center, radius



def division_central_consistency(gb_list, gb_list_not):
    '''
    中心一致性划分
    Args:
        gb_list:
        gb_list_not:

    Returns:

    '''
    gb_list_new = []
    methods = '2-means'
    for gb in gb_list:
        if len(gb) > 1:
            ball_1, ball_2 = spilt_ball_k_means(gb, 2, methods)
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_not.append(gb)
                continue
            ccp, ccp_flag = get_ccp(gb)
            _, radius = calculate_center_radius(gb)
            sprase_parent = get_dm_sparse(gb)
            sprase_child1 = get_dm_sparse(ball_1)
            sprase_child2 = get_dm_sparse(ball_2)

            t1 = ccp_flag
            t4 = len(ball_2) > 2 and len(ball_1) > 2
            
            if (sprase_child1 >= sprase_parent or sprase_child2 >= sprase_parent) and (
                    len(ball_1) == 1 or len(ball_2) == 1):
                t4 = True
            if t1 and t4:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_not.append(gb)
        else:
            gb_list_not.append(gb)

    return gb_list_new, gb_list_not


def division_central_consistency_strong(gb_list,gb_list_not):
    '''
    强中心一致性划分
    Args:
        gb_list:
        gb_list_not:

    Returns:

    '''
    gb_list_new = []
    methods = '2-means'

    for gb in gb_list:
        if len(gb) > 1:
            ball_1, ball_2 = spilt_ball_k_means(gb, 2, methods)
            ccp, ccp_flag = get_ccp_strong(gb)
            t1 = ccp_flag
            t4 = len(ball_2) > 2 and len(ball_1) > 2
            if t1 and t4:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_not.append(gb)
        else:
            gb_list_not.append(gb)

    return gb_list_new,gb_list_not


def spilt_ball_k_means(data, n, methods):
    if methods == '2-means':
        
        kmeans = KMeans(n_clusters=n, random_state=0, n_init=3, max_iter=2)
        kmeans.fit(data)
        labels = kmeans.labels_
        cluster1 = [data[i].tolist() for i in range(len(data)) if labels[i] == 0]
        cluster2 = [data[i].tolist() for i in range(len(data)) if labels[i] == 1]
        ball1 = np.array(cluster1)
        ball2 = np.array(cluster2)
        return [ball1, ball2]

    elif methods == 'k-means':

        kmeans = KMeans(n_clusters=n, random_state=1)
        kmeans.fit(data)

        
        labels = kmeans.labels_

        
        clusters = [[] for _ in range(n)]
        for i in range(len(data)):
            for cluster_index in range(n):
                if labels[i] == cluster_index:
                    clusters[cluster_index].append(data[i].tolist())

        balls = [np.array(cluster) for cluster in clusters]
        
        
        return balls
    else:
        pass


def divide_gb_k(data, k):
    kmeans = KMeans(n_clusters=k, random_state=5)
    kmeans.fit(data)
    labels = kmeans.labels_
    gb_list_temp = []
    for idx in range(k):
        cluster1 = [data[i].tolist() for i in range(len(data)) if labels[i] == idx]
        gb_list_temp.append(np.array(cluster1))
    return gb_list_temp



def spilt_ball(data):
    ball1 = []
    ball2 = []
    A = pdist(data)
    d_mat = squareform(A)
    r, c = np.where(d_mat == np.max(d_mat))
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):
        if d_mat[j, r1] < d_mat[j, c1]:
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])

    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]

def get_ccp(gb):
    '''
    得到中心一致性，中心一致性指平均半径内的样本密度与最大半径内的样本密度的比值，如果比值在1~1.3就不分裂，
    否则分裂为两个粒球。密度指：样本个数 / 半径 ** 维度。
    Args:
        gb:

    Returns:

    '''
    num = len(gb)
    if num == 0:
        return 0, False
    
    dimension = len(gb[0])
    center = gb.mean(axis=0)
    
    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5

    avg_radius = np.mean(distances)
    max_radius = np.max(distances)
    points_inside_avg_radius = np.sum(distances <= avg_radius)
    density_inside_avg_radius = points_inside_avg_radius / (avg_radius ** dimension)
    density_max_radius = num / (max_radius ** dimension)
    ccp = density_inside_avg_radius / density_max_radius
    ccp_flag = ccp >= 1.30 or ccp < 1 # 分裂
    return ccp, ccp_flag

def get_ccp_strong(gb):
    '''
    强中心一致性，最大半径的四分之一半径内密度占最大半径内密度的比值
    Args:
        gb:

    Returns:

    '''
    num = len(gb)
    if num == 0:
        return 0, False
    dimension = len(gb[0])
    center = gb.mean(axis=0)
    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5

    max_radius = np.max(distances)
    quarter_radius = max_radius / 4
    points_inside_quarter_radius = np.sum(distances <= quarter_radius)
    density_inside_quarter_radius = points_inside_quarter_radius / (quarter_radius ** dimension)
    density_max_radius = num / (max_radius ** dimension)
    ccp_strong = density_inside_quarter_radius / density_max_radius
    ccp_flag_strong = ccp_strong >= 1.3 or ccp_strong < 1
    return ccp_strong, ccp_flag_strong


def get_dm_sparse(gb):
    num = len(gb)
    dim = len(gb[0])

    if num == 0:
        return 0
    center = gb.mean(0)
    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)

    
    
    

    
    sparsity = num / (radius ** dim)
    
    

    if num > 2:
        return sparsity
    else:
        return radius
        
def get_radius(gb):
    num = len(gb)
    center = gb.mean(0)
    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius

def de_sparse(gb_list):

    avg_r_div_n = sum([get_radius(x) / len(x) for x in gb_list]) / len(gb_list)
    avg_r =sum([get_radius(x)  for x in gb_list]) / len(gb_list)
    gb_list_new = []
    gb_split_list_new = []
    for gb in gb_list:
        
        r_t = get_radius(gb)
        if r_t  / len(gb) > avg_r_div_n and r_t > avg_r:
            gb_split_list_new.append(gb)
        else:
            gb_list_new.append(gb)
    for gb in gb_split_list_new:
        if len(gb) > 1:
            ball_1, ball_2 = spilt_ball(gb)
            gb_list_new.extend([ball_1, ball_2])
        else:
            gb_list_new.append(gb)
    return gb_list_new

def get_gb_division_x(data,plt_flag=False):
    '''
    根据数据集生成粒球划分
        x
        Args:
            data:
        Returns:
    '''

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    gb_list_not_temp = []
    k1 = int(np.sqrt(len(data)))
    gb_list_temp = divide_gb_k(data, k1) # 先粗划分为根号n个粒球
    if plt_flag == True:
        plot_dot(data)
        draw_ball(gb_list_temp, '')
    
    gb_list_temp,gb_list_not_temp = division_central_consistency_strong(gb_list_temp,gb_list_not_temp)
    gb_list_temp = gb_list_temp + gb_list_not_temp
    gb_list_not_temp = []
    if plt_flag == True:
        plot_dot(data)
        draw_ball(gb_list_temp, '')

    i = 0
    # 根据中心一致性进行粒球细分
    while 1:
        i += 1
        ball_number_old = len(gb_list_temp) + len(gb_list_not_temp)
        gb_list_temp, gb_list_not_temp = division_central_consistency(gb_list_temp, gb_list_not_temp)
        ball_number_new = len(gb_list_temp) + len(gb_list_not_temp)

        if ball_number_new == ball_number_old:
            gb_list_temp = gb_list_not_temp
            break

    if plt_flag == True:
        plot_dot(data)
        draw_ball(gb_list_temp + gb_list_not_temp, '')
    count = 0
    for gb in gb_list_temp:
        if len(gb) == 1:
            pass
        else:
            count += 1
    gb_list_temp = [x for x in gb_list_temp if len(x) != 0]
    
    gb_list_temp = de_sparse(gb_list_temp)
    if plt_flag == True:
        plot_dot(data)
        draw_ball(gb_list_temp, '')
    return gb_list_temp


def load_txt_dataset(path):
    
    with open(path, 'r') as file:
        lines = file.readlines()
    data = np.array([[float(num) for num in line.split()] for line in lines])
    return data


def main():

    keys = ['D1']
    for d in range(len(keys)):
        path = r'F:\Python\dataset\changzhi_data_set\no_noise\t8.txt'
        data = load_txt_dataset(path)
        y = np.array([1] * data.shape[0])

        gb_list = get_gb_division_x(data)


if __name__ == '__main__':
    main()  

