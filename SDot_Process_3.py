import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from SDot_Clustering import Clustering
from SDot_Process_2 import Process_2

class Process_3(Process_2):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.file_list = os.listdir(self.data_path)
        
        self.columns = ['PM25', 'PM10', 'NOISE', 'HUMI', 'TEMP']

        # 초기화
        self.analysis_df = pd.DataFrame()

        # 파일이름 및 폴더명
        self.filter_nan_grid_num_file = '6_filtered_GRID_NUM.csv'
        self.interpolate_file = '7_Interpolate_{}.csv'.format('linear')
        self.cluster_folder = 'Clustering'

    def load_data(self):
        try:
            if self.analysis_file in self.file_list:
                self.analysis_df = pd.read_csv(self.data_path + self.analysis_file)
                self.analysis_df['REG_DATE'] = pd.to_datetime(self.analysis_df['REG_DATE'])

            else:
                raise FileNotFoundError
        
        except FileNotFoundError:
            print('설정한 파일경로에 "{}"이 있는지 확인해주세요'.format(self.analysis_file))

    # Nan 데이터의 개수 너무 많은 GRID_NUM 필터링
    def filter_nan(self, num_sigma= 2):
        filter = 0
        before = len(set(self.analysis_df['GRID_NUM'].values))

        if self.filter_nan_grid_num_file in self.file_list:
            self.analysis_df = pd.read_csv(self.data_path + self.filter_nan_grid_num_file)
            self.analysis_df['REG_DATE'] = pd.to_datetime(self.analysis_df['REG_DATE'])

        else:
            min_date = self.analysis_df['REG_DATE'].min()
            values_per_date = len(self.analysis_df[self.analysis_df['REG_DATE'] == min_date]['PM25'])

            while True:
                count_nan = self.analysis_df[self.analysis_df['REG_DATE'] == min_date]['PM25'].isna().sum()
                if (values_per_date - count_nan) > (values_per_date * 0.5):
                    break
                min_date = min_date + pd.DateOffset(hours= 1)

            # 첫 값이 nan인 값 제거
            # print(len(self.analysis_df[self.analysis_df['REG_DATE'] == min_date]))
            self.analysis_df = self.analysis_df[self.analysis_df['REG_DATE'] >= min_date]
            first_not_nan_df = self.analysis_df[self.analysis_df['REG_DATE'] == min_date].dropna()
            first_not_nan_list = first_not_nan_df['GRID_NUM'].to_list()
            self.analysis_df = self.analysis_df[self.analysis_df['GRID_NUM'].isin(first_not_nan_list)]
            # print(len(self.analysis_df[self.analysis_df['REG_DATE'] == min_date]))

            # 설정값 이상의 GRID_NUM 당 데이터의 개수 Filter
            filter_df = self.analysis_df[self.analysis_df['REG_DATE'] >= min_date]
            filter_df = filter_df.groupby('GRID_NUM')['PM25'].count()
            filter_df = pd.DataFrame(filter_df)
            filter_df.reset_index(inplace= True, drop= False)
            filter_df.rename(columns= {'PM25' : 'Values'}, inplace= True)

            mean = filter_df['Values'].mean()
            std = filter_df['Values'].std()

            filter = np.trunc(mean - (num_sigma * std))

            filter_list = filter_df[filter_df['Values'] > filter]['GRID_NUM'].to_list()

            self.analysis_df = self.analysis_df[self.analysis_df['GRID_NUM'].isin(filter_list)]
            self.analysis_df.to_csv(self.data_path + self.filter_nan_grid_num_file, index= False)

        after = len(set(self.analysis_df['GRID_NUM'].values))
        print('Start date :\t {}'.format(self.analysis_df['REG_DATE'].min()))
        print('End date :\t {}'.format(self.analysis_df['REG_DATE'].max()))
        print('Filter Num : {}'.format(filter))
        print('Before Filtering :\t {}'.format(before))
        print('After Filtering :\t {}'.format(after))

    # 보간법을 활용한 NAN 데이터 처리
    def data_interpolate(self, method= 'linear'):
        # method : interpolate 종류('linear', ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’)
        # 8종류 중 'linear'와 'slinear' 사용 예정
        self.interpolate_file = '7_Interpolate_{}.csv'.format(method)

        if self.interpolate_file not in self.file_list:
            for col in self.columns:
                self.analysis_df[col] = self.analysis_df[col].interpolate(method= method)

            self.analysis_df.to_csv(self.data_path + self.interpolate_file, index= False)

        else:
            self.analysis_df = pd.read_csv(self.data_path + self.interpolate_file)
            self.analysis_df['REG_DATE'] = pd.to_datetime(self.analysis_df['REG_DATE'])

    def cluster_analysis(self, year= 2022, min_cluster = 10, max_cluster = 11):
        # 초기화
        cluster_range = list()
        
        # Cluster 결과를 저장할 폴더 생성
        if self.cluster_folder not in self.file_list:
            os.mkdir(self.data_path + self.cluster_folder + '/')

        path = self.data_path + self.cluster_folder + '/'

        if min_cluster < max_cluster:
            cluster_range = range(min_cluster, max_cluster + 1)
        else:
            value = max(min_cluster, max_cluster)
            cluster_range = range(value, value + 1)

        for col in self.columns:
            inertias = list()
            for n_cluster in cluster_range:
                clustering = Clustering(self.analysis_df, col, path)
                km_inertia = clustering.SdotTimeSeriesKMeans(
                    n_cluster= n_cluster,
                    metric= 'dtw',
                    dtw_inertia= True
                )
                clustering.SdotPCA()
                clustering.show_cluster_dist()

                inertias.append(km_inertia)

            # Plot ks vs inertiasplt
            plt.figure(figsize=(15, 10))
            plt.plot(cluster_range, inertias, '-o')
            plt.xlabel('number of clusters, k')
            plt.ylabel('inertia')
            plt.xticks(cluster_range)
            plt.title('[{}] # of cluster and inertias'. format(col))
            plt.savefig(path + '[{}] # of cluster and inertias {}'.format(col, year))