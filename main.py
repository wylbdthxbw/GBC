# _*_coding:utf-8 _*_
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import warnings
from ball_divide_cluster_final import divide_ball_GBC_y,visualize
from evaluation import evaluation
import csv

warnings.filterwarnings("ignore")

def write_csv(filename, lst):
    with open(filename, "a+", newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        '''
        data = [
        ['Name', 'Age', 'City'],
        ['Alice', 25, 'New York'],
        ['Bob', 30, 'San Francisco'],
        ['Charlie', 22, 'Los Angeles']
        ]       
        '''
        writer.writerows(lst)

def visualize_save(path, h, color, title=''):
    z = h
    plt.figure(figsize=(10, 10))
    plt.title(title)
    z_color = np.insert(z, 2, color, axis=1)
    # label 从0开始
    label_max = int(max(color))
    for label in range(0, label_max + 1):
        plt.scatter(z[z_color[:, -1] == label, 0], z[z_color[:, -1] == label, 1], s=7, cmap='Set2',
                    label='cluster {}'.format(label))  # ,cmap="Set2"
    # plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap='Set2') #,cmap="Set2"

    # 添加图例
    plt.legend(loc='upper right')
    plt.savefig(path + title + '.png')
    plt.show()

def main():
    np.random.seed(0)
    alg_name = 'GBC'
    name_list = os.listdir('./GBC15_label')
    column_wirte_list=[]
    for data_name in name_list:
        data_name = data_name[:-4]
        print(data_name)
        path = r'./GBC15_label' + '/' + data_name+ '.csv'

        df = pd.read_csv(path,header=None)
        numpy_array = df.values
        X = numpy_array[:,1:]
        y = numpy_array[:,0]
        n_cluster = len(set(y))

        st = time.time()
        y_ball = divide_ball_GBC_y(X, n_cluster, False)
        acc, nmi, ari, f1 = evaluation(y, y_ball)
        et = time.time()
        ct = et - st
        save_path = './fig_{}/'.format(alg_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        column_wirte_list.append([str(round(acc, 3))])
        column_wirte_list.append([str(round(nmi, 3))])
        column_wirte_list.append([str(round(ct, 3))])
        print("  acc nmi time:",column_wirte_list)
        write_csv('./{}_column.csv'.format(alg_name), column_wirte_list)
        column_wirte_list = []
        visualize_save(save_path,X, np.array(y_ball), 'res {}'.format(data_name))
        print("---------------")



if __name__ == '__main__':
    main()
