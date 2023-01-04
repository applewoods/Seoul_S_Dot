import os
import pandas as pd
import numpy as np

from tqdm import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt

import collections
from tslearn.clustering import TimeSeriesKMeans

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# 시계열분석
class Clustering:
    def __init__(self, dataframe, colum, path ='./', year= 2022):
        self.dataframe = dataframe
        self.path = path
        self.year = year
        self.column = colum
        self.columns = ['GRID_NUM', 'REG_DATE', self.column]

        self.dataframe = self.dataframe[self.columns]
        self.dataframe['REG_DATE'] = pd.to_datetime(self.dataframe['REG_DATE'])

    def preprocessing_cluster(self):
        self.serial_npy = list()
        self.feature_npy = list()
        
        grid_num = list(set(self.dataframe['GRID_NUM'].values))

        for idx in grid_num:
            values = self.dataframe[self.dataframe['GRID_NUM'] == idx][self.column].to_list()
            self.serial_npy.append(idx)
            self.feature_npy.append(values)

        self.date_index = self.dataframe['REG_DATE'].value_counts().index.to_list()

    def normalization(self):
        scaler = MinMaxScaler()
        scaler.fit(self.feature_npy)
        self.norm_feature_npy = scaler.transform(self.feature_npy)
        self.norm_feature_npy = np.array(self.norm_feature_npy)

    def SdotTimeSeriesKMeans(self, n_cluster= 10, metric= 'euclidean', dtw_inertia= False, max_iter= 150, n_jobs= -1):
        # n_cluster : 형성할 클러스터의 수
        # max_iter  : 단일 실행에 대한 k-평균 알고리즘의 최대 반복 횟수
        # metric    : {'euclidean', 'dtw', 'softdtw'}
        # n_jobs    : 몇개의 cpu를 사용할것인지 선택
        # 참고자료
        # https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html
        # https://machineindeep.tistory.com/36
        # https://tootouch.github.io/contest%20review/kdx_dashboard_part2/
        self.preprocessing_cluster()
        self.normalization()

        self.n_cluster = n_cluster

        km = TimeSeriesKMeans(n_clusters= self.n_cluster, metric= metric, dtw_inertia= dtw_inertia, max_iter= max_iter, n_jobs= n_jobs, random_state= self.year)
        km.fit(self.norm_feature_npy)

        self.rlt_tsm = km.predict(self.norm_feature_npy)

        labels, sizes = list(), list()

        for i in range(self.n_cluster):
            labels.append('cluster #' + str(i + 1))
            sizes.append(collections.Counter(self.rlt_tsm)[i])

        plt.figure(figsize=(15,8))
        plt.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
        plt.title("Cluster Distribution (number of serial:"+ str(len(self.serial_npy)) +")", fontsize=20)
        plt.savefig(self.path + self.column + '_Cluster_{}_Distribution_{}.png'.format(self.n_cluster, self.year))

        if metric == 'dtw' and dtw_inertia:
            return km.inertia_

    def SdotPCA(self, n_components= 2):
        pca = PCA(n_components= n_components)
        rlt_pca = pca.fit_transform(self.norm_feature_npy[~np.isnan(self.norm_feature_npy).any(axis=1)])

        plt.figure(figsize = (15, 10))
        for i in range(self.n_cluster):
            label_name = 'cluster #' + str(i + 1)
            plt.scatter(rlt_pca[[self.rlt_tsm==i]][:,0], rlt_pca[[self.rlt_tsm==i]][:,1], label=label_name)

        plt.legend()
        plt.savefig(self.path + self.column + '_PCA_Cluster_{}_Visualization_{}.png'.format(self.n_cluster, self.year))

    def show_cluster_dist(self):
        rlt_npy = list()
        date_index = [pd.to_datetime(x).strftime('%Y-%m-%d') for x in self.date_index]
        columns = ['GRID_NUM'] + date_index + ['Cluster']

        for i in tqdm(range(len(self.serial_npy))):
            tmp = list()
            tmp.append(self.serial_npy[i])
            tmp.extend(self.feature_npy[i])
            tmp.append(self.rlt_tsm[i])
            rlt_npy.append(tmp)

        result_df = pd.DataFrame(rlt_npy, columns= columns)
        result_df['Cluster'] = result_df['Cluster'].apply(lambda x: int(x) + 1)
        result_df = result_df.astype({'GRID_NUM' : np.int64})
        result_df.reset_index(inplace= True, drop= False)

        result_df[['GRID_NUM', 'Cluster']].to_csv(self.path + self.column + '_Cluster_{}_Result_{}.csv'.format(self.n_cluster, self.year), index= False, encoding= 'utf-8-sig')