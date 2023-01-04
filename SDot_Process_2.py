import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
# import modin.pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from SDot_Process_1 import Process_1
from SDot_Preprocessing import RM_Outlier

class Process_2(Process_1):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.file_list = os.listdir(self.data_path)      # 설정한 파일경로 안에 있는 파일의 종류
        self.columns = ['PM25', 'PM10', 'NOISE', 'HUMI', 'TEMP']

        # 초기화
        self.select_df = pd.DataFrame()             # 설정한 기간에 해당되는 RAW_DATA DataFrame
        self.grid_df = pd.DataFrame()               # select_df에서 GRID_NUM이 추가된 DataFrame
        self.analysis_df = pd.DataFrame()           # 최종 Cluster에 사용할 DataFrame

        # 파일 및 폴더명
        self.raw_grid_num_file = '서울시_대기오염_측정소-SDOT-그리드_통합.csv'
        self.clean_sdot_grid_num_file = '4_Add_GRIDNUM.csv'
        # self.rm_outlier_grid_num_file = '4_RM_Outlier_GRID_NUM.csv'
        self.analysis_file = '5_Analysis_Data.csv'

        self.vis_grid_folder = 'visualize_grid_data'

    # GRID_NUM 센서(EQUI_INFO_KEY) 기준으로 추가
    def add_grid_num(self):
        try:
            if self.clean_sdot_grid_num_file in self.file_list:
                # 3_Add_GRIDNUM.csv 파일이 있을 경우
                self.grid_df = pd.read_csv(self.data_path + self.clean_sdot_grid_num_file)
                self.grid_df['REG_DATE'] = pd.to_datetime(self.grid_df['REG_DATE'])

            elif self.raw_grid_num_file in self.file_list:
                self.raw_df = pd.read_csv(self.data_path + self.rm_outlier_file)
                self.raw_df['REG_DATE'] = pd.to_datetime(self.raw_df['REG_DATE'])

                # GRID_NUM 데이터 가져오기
                raw_grid_df = pd.read_csv(self.data_path + self.raw_grid_num_file)
                raw_grid_df = raw_grid_df[['시리얼', 'GRID_NUM']]
                raw_grid_df.rename(columns= {
                    '시리얼' : 'EQUI_INFO_KEY'
                }, inplace= True)

                # GRID_NUM 데이터 합치기
                self.grid_df = pd.merge(self.raw_df, raw_grid_df, how= 'inner', on= 'EQUI_INFO_KEY')
                self.grid_df.to_csv(self.data_path + self.clean_sdot_grid_num_file, index= False)

            else:
                raise FileNotFoundError

        except FileNotFoundError:
            print('설정한 파일경로에 "{}"이 있는지 확인해주세요'.format(self.raw_grid_num_file))

    # 분석에 사용할 데이터의 기간 설정
    def select_date_range_raw(self, start_date, end_date):
        # start_date : 설정 기간 시작날짜
        # end_date : 설정 기간 종료날짜
        self.select_df = self.grid_df[self.grid_df['REG_DATE'].between(start_date, end_date)]

        print('Select Data Start Date\t: {}'.format(self.select_df['REG_DATE'].min()))
        print('Select Data End Date\t: {}'.format(self.select_df['REG_DATE'].max()))
        print('='*50)

        # GRID_NUM와 시간 별로 데이터 평균으로 묶기
        self.grid_df = self.select_df.set_index('REG_DATE')
        self.grid_df = self.grid_df.groupby(['GRID_NUM']).resample('D').mean()
        self.grid_df.drop('GRID_NUM', axis= 1, inplace= True)
        self.grid_df.reset_index(inplace= True, drop= False)

    # RAW_DATA가 적은 기간의 GRID_NUM 추출 및 종류 필터링
    def select_date_range_grid_num(self, filter_num= 700):
        # filter_num : 설정한 값 이하의 센서의 종류 추출에 사용하는 변수
        # GRID_NUM이 추가된 RAW_DATA 파일을 설정한 기간(시간)과 GRID_NUM 단위로 평균 데이터를 사용해서 묶기
        if self.analysis_file in self.file_list:
            self.analysis_df = pd.read_csv(self.data_path + self.analysis_file)
            self.analysis_df['REG_DATE'] = pd.to_datetime(self.analysis_df['REG_DATE'])

        else:
            # GRID_NUM 종류 필터링
            grid_count_df = pd.DataFrame(self.grid_df['REG_DATE'].value_counts())
            grid_count_df.reset_index(inplace= True, drop= False)
            grid_count_df.rename(columns={'REG_DATE' : 'Values', 'index' : 'REG_DATE'}, inplace= True)
            grid_count_df['REG_DATE'] = pd.to_datetime(grid_count_df['REG_DATE'])
            grid_count_df = grid_count_df[grid_count_df['Values'] < filter_num]

            min_date = pd.to_datetime(grid_count_df['REG_DATE'].min())
            max_date = pd.to_datetime(grid_count_df['REG_DATE'].max())

            # GRID_NUM의 종류를 필터링한 DataFrame
            filter_df = self.grid_df[self.grid_df['REG_DATE'].between(min_date, max_date)]
            filter_list = list(set(filter_df['GRID_NUM'].values))

            self.analysis_df = self.grid_df[self.grid_df['GRID_NUM'].isin(filter_list)]
            self.analysis_df.to_csv(self.data_path + self.analysis_file, index= False)

    def vis_analysis_data(self):
        # 시간과 생활환경 데이터 별 Grid의 갯수 변화
        for col in self.columns:
            grid_vis_df = self.analysis_df.groupby('REG_DATE')[col].count()
            grid_vis_df = pd.DataFrame(grid_vis_df)
            grid_vis_df.reset_index(inplace= True, drop= False)
            grid_vis_df.rename(columns= {col : 'Values'}, inplace= True)

            x_value = grid_vis_df['REG_DATE'].to_list()
            y_value = grid_vis_df['Values'].to_list()

            plt.figure(figsize=(40, 8))
            plt.plot(x_value, y_value, label= col)
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('# of GRID')
            plt.xticks(pd.date_range(min(x_value), max(x_value), freq= pd.offsets.MonthBegin(1)))
            plt.title(col)

            # visualize_raw_data 폴더가 설정한 경로에 없을 경우 폴더 생성
            self.file_list = os.listdir(self.data_path)

            if self.vis_grid_folder not in self.file_list:
                os.mkdir(self.data_path + self.vis_grid_folder + '/')
                
            plt.savefig(self.data_path + self.vis_grid_folder + '/GRID_DATA_PLOT_{}.png'.format(col))

        # 전체 GRID_NUM_DATA를 하나의 그래프에 시각화
        x_value = list()
        plt.figure(figsize=(40, 8))
        for col in self.columns:
            grid_vis_df = self.analysis_df.groupby('REG_DATE')[col].count()
            grid_vis_df = pd.DataFrame(grid_vis_df)
            grid_vis_df.reset_index(inplace= True, drop= False)
            grid_vis_df.rename(columns= {col : 'Values'}, inplace= True)

            x_value = grid_vis_df['REG_DATE'].to_list()
            y_value = grid_vis_df['Values'].to_list()

            plt.plot(x_value, y_value, label= col)
            
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('# of GRID')
        plt.xticks(pd.date_range(min(x_value), max(x_value), freq= pd.offsets.MonthBegin(1)))
        plt.title('Total')
        plt.savefig(self.data_path + self.vis_grid_folder + '/GRID_DATA_PLOT_{}.png'.format('Total'))