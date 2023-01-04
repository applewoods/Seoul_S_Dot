import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from SDot_Preprocessing import RM_Outlier

class Process_1:
    def __init__(self, data_path):
        self.data_path = data_path                  # 사용자가 설정한 경로
        self.file_list = os.listdir(self.data_path)      # 설정한 파일경로 안에 있는 파일의 종류

        self.columns = ['PM25', 'PM10', 'NOISE', 'HUMI', 'TEMP']

        # 초기화
        self.group_df = pd.DataFrame()
        self.outlier_df = pd.DataFrame()

        # 파일 및 폴더명
        self.raw_data_file = '1_RAW_SDot.csv'
        self.clean_data_file = '2_Clean_RAW_SDot.csv'
        self.rm_outlier_file = '3_RM_Outlier.csv'
        
        self.save_clean_vis_folder = 'visualize_raw_data'

    def load_raw_file(self):
        # '1_RAW_SDot.csv' :
        #   * 원천 데이터에서 5개의 생활환경 데이터(초미세먼지, 미세먼지, 소음, 온도, 습도) 추출
        #   * 'REG_DATE', 'EQUI_INFO_KEY' 데이터 중 Null 값이 있는 데이터 제거

        try:
            if self.raw_data_file in self.file_list:
                raw_df = pd.read_csv(self.data_path + self.raw_data_file)
                raw_df['REG_DATE'] = pd.to_datetime(raw_df['REG_DATE'])

                # 초미세먼지, 미세먼지, 습도 데이터 중 값이 0인 경우 데이터를 Null 처리
                raw_df['PM25'] = raw_df['PM25'].apply(lambda x: np.nan if x <= 0 else x)
                raw_df['PM10'] = raw_df['PM10'].apply(lambda x: np.nan if x <= 0 else x)
                raw_df['HUMI'] = raw_df['HUMI'].apply(lambda x: np.nan if x <= 0 and x> 100 else x)
                raw_df['TEMP'] = raw_df['TEMP'].apply(lambda x: np.nan if x <= -50 and x>= 50 else x)

                # 시간과 센서를 1시간 범위로 시간 단위를 통일하고, 같은 시간대에 여러개의 센서값이 있을 경우 평균값으로 처리
                self.group_df = raw_df.set_index('REG_DATE', inplace= True)
                self.group_df = raw_df.groupby('EQUI_INFO_KEY').resample('H').mean()
                self.group_df = pd.DataFrame(self.group_df)
                self.group_df.reset_index(inplace= True, drop= False)

                print('Raw Start Date\t: {}'.format(self.group_df['REG_DATE'].min()))
                print('Raw End Date\t: {}'.format(self.group_df['REG_DATE'].max()))
                print('='*50)

                self.group_df.to_csv(self.data_path + self.clean_data_file, index= False)
                print('Raw Data Save', end= '\r')

                # 시각화
                self.vis_raw_data(self.group_df)
                print('Raw Data Visualization Finish', end= '\r')

            else:
                raise FileNotFoundError
        
        except FileNotFoundError:
            print('설정한 파일경로에 "{}"이 있는지 확인해주세요'.format(self.raw_data_file))

    def rm_outlier_values(self, set_rule= 'D', method = 'sigma', set_group= 'EQUI_INFO_KEY'):
        # 1일 단위 데이터를 만들때는 각 데이터의 이상치 제거 후 평균값 사용
        # method = ['sigma', 'box']
        if self.rm_outlier_file in self.file_list:
            self.outlier_df = pd.read_csv(self.data_path + self.rm_outlier_file)
            self.outlier_df['REG_DATE'] = pd.to_datetime(self.outlier_df['REG_DATE'])
            
        else:
            outlier = RM_Outlier(self.group_df)
            self.outlier_df = outlier.remove_outlier(set_rule= set_rule, set_group= set_group, method= method)
            self.outlier_df.to_csv(self.data_path + self.rm_outlier_file, index= False)

        print('RM_Outlier Start Date\t:{}'.format(self.outlier_df['REG_DATE'].min()))
        print('RM_Outlier End Date\t:{}'.format(self.outlier_df['REG_DATE'].max()))
        print('='*50)

        # 시각화
        self.vis_raw_data(self.outlier_df, sep= 'RM_Outlier')
        print('RM_Outlier Data Visualization Finish', end= '\r')

    def vis_raw_data(self, dataframe, sep= 'RAW'):
        # RAW_DATA와 Outlier 제거한 데이터의 생활환경 각각의 데이터를 날짜 별로 시각화
        for col in self.columns:
            # plot_df = self.outlier_df.groupby(['REG_DATE', 'EQUI_INFO_KEY'])[col].count()
            plot_df = dataframe.groupby(['REG_DATE', 'EQUI_INFO_KEY'])[col].count()
            plot_df = pd.DataFrame(plot_df)
            plot_df.reset_index(inplace= True, drop= False)
            plot_df.rename(columns={col : 'Values'}, inplace= True)
            plot_df = plot_df.groupby('REG_DATE').sum()
            plot_df.reset_index(inplace= True, drop= False)

            x_value = plot_df['REG_DATE'].to_list()
            y_value = plot_df['Values'].to_list()

            plt.figure(figsize=(40, 8))
            plt.plot(x_value, y_value, label= col)
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('# of Sensor')
            plt.xticks(pd.date_range(min(x_value), max(x_value), freq= pd.offsets.MonthBegin(1)))
            plt.title(col)

            # visualize_raw_data 폴더가 설정한 경로에 없을 경우 폴더 생성
            self.file_list = os.listdir(self.data_path)

            if self.save_clean_vis_folder not in self.file_list:
                os.mkdir(self.data_path + self.save_clean_vis_folder + '/')
                
            # 그래프저장
            plt.savefig(self.data_path + self.save_clean_vis_folder + '/{}_DATA_PLOT_{}.png'.format(sep, col))

        # 전체 RAW_DATA를 하나의 그래프에 시각화
        plt.figure(figsize=(40, 8))
        for col in self.columns:
            # plot_df = self.outlier_df.groupby(['REG_DATE', 'EQUI_INFO_KEY'])[col].count()
            plot_df = dataframe.groupby(['REG_DATE', 'EQUI_INFO_KEY'])[col].count()
            plot_df = pd.DataFrame(plot_df)
            plot_df.reset_index(inplace= True, drop= False)
            plot_df.rename(columns={col : 'Values'}, inplace= True)
            plot_df = plot_df.groupby('REG_DATE').sum()
            plot_df.reset_index(inplace= True, drop= False)

            x_value = plot_df['REG_DATE'].to_list()
            y_value = plot_df['Values'].to_list()

            plt.plot(x_value, y_value, label= col)

        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('# of Sensor')
        plt.xticks(pd.date_range(min(x_value), max(x_value), freq= pd.offsets.MonthBegin(1)))
        plt.title('Total')      
        plt.savefig(self.data_path + self.save_clean_vis_folder + '/{}_DATA_PLOT_{}.png'.format(sep, 'Total'))