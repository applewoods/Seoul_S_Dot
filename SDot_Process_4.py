import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from SDot_Process_3 import Process_3

FONT_SIZE = 35
FONT_TITLE = 45

class Process_4(Process_3):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.file_list = os.listdir(self.data_path)
        
        self.columns = ['PM25', 'PM10', 'NOISE', 'HUMI', 'TEMP']
        self.weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.season = ['Winter', 'Spring', 'Summer', 'Fall']

        # 초기화
        self.analysis_df = pd.DataFrame()       # GRID_NUM이 데이터가 있는 DataFrame

        # 파일 및 폴더 명
        self.best_k_folder = 'Best_K'

    def load_data(self):
        try:
            if self.interpolate_file in self.file_list:
                self.analysis_df = pd.read_csv(self.data_path + self.interpolate_file)
                self.analysis_df['REG_DATE'] = pd.to_datetime(self.analysis_df['REG_DATE'])

                self.month = pd.date_range(self.analysis_df['REG_DATE'].min(), self.analysis_df['REG_DATE'].max(), freq= 'M')
                self.month = self.month.month

            else:
                raise FileNotFoundError
        
        except FileNotFoundError:
            print('설정한 파일경로에 "{}"이 있는지 확인해주세요'.format(self.interpolate_file))

    def box_plot(self):
        if self.best_k_folder not in self.file_list:
            os.mkdir(self.data_path + self.best_k_folder + '/')

        path = self.data_path + self.best_k_folder + '/'
        best_k_files = os.listdir(path)

        try:
            if len(best_k_files) > 1:
                for file in best_k_files:
                    if file != '.DS_Store' and os.path.splitext(file)[1] == '.csv':
                        col_name = file.split('_')[0]
                        cluster_df = pd.read_csv(path + file)
                        plot_df = pd.merge(self.analysis_df, cluster_df, how= 'inner', on= 'GRID_NUM')
                        plot_df = plot_df[['REG_DATE', 'GRID_NUM', col_name, 'Cluster']]

                        num_cluster = sorted(list(set(plot_df['Cluster'].values)))
                        max_value = int(plot_df[col_name].max())
                        min_value = int(plot_df[col_name].min())

                        for num in num_cluster:
                            self.vis_box_plot_day(plot_df, num, col_name, min_value, max_value, path)
                            self.vis_box_plot_weekday(plot_df, num, col_name, min_value, max_value, path)
                            self.vis_box_plot_month(plot_df, num, col_name, min_value, max_value, path)
                            self.vis_box_plot_season(plot_df, num, col_name, min_value, max_value, path)
                                   
            else:
                raise FileNotFoundError

        except FileNotFoundError:
            print('최적의 K 결과를 {} 폴더에 넣어주세요'.format(self.best_k_folder))

    def line_plot(self):
        if self.best_k_folder not in self.file_list:
            os.mkdir(self.data_path + self.best_k_folder + '/')

        path = self.data_path + self.best_k_folder + '/'
        best_k_files = os.listdir(path)
        
        try:
            if len(best_k_files) > 1:
                for file in best_k_files:
                    if file != '.DS_Store' and os.path.splitext(file)[1] == '.csv':
                        col_name = file.split('_')[0]
                        cluster_df = pd.read_csv(path + file)
                        plot_df = pd.merge(self.analysis_df, cluster_df, how= 'inner', on= 'GRID_NUM')
                        plot_df = plot_df[['REG_DATE', 'GRID_NUM', col_name, 'Cluster']]

                        plot_df2 = plot_df.groupby(['REG_DATE', 'Cluster'])[col_name].mean()
                        plot_df2 = pd.DataFrame(plot_df2)
                        plot_df2.reset_index(inplace= True, drop= False)

                        num_cluster = sorted(list(set(plot_df['Cluster'].values)))
                        
                        max_value = int(plot_df[col_name].max())
                        min_value = int(plot_df[col_name].min())

                        max_value2 = int(plot_df2[col_name].max())
                        min_value2 = int(plot_df2[col_name].min())

                        # for num in num_cluster:
                        #     self.vis_line_plot(plot_df2, num, col_name, min_value2, max_value2, path)
                        #     self.vis_line_per_cluster_values(plot_df, num, col_name, min_value, max_value, path, 10)

                        x_value = 0
                        step_value = max(int((max_value2 - min_value2) / 20), 1)
                        plt.figure(figsize=(40, 30))
                        for num in num_cluster:
                            tmp_df = plot_df2[plot_df2['Cluster'] == num]

                            x_value = tmp_df['REG_DATE'].to_list()
                            y_value = tmp_df[col_name].to_list()
                            
                            # step_value = max(int((max_value - min_value) / 20), 1)

                            # plt.figure(figsize=(40, 14))
                            plt.plot(x_value, y_value, label= 'Cluster{}'.format(num))
                            plt.title('[{}] Total Cluster Mean Values'.format(col_name), fontsize= FONT_TITLE)
                        plt.xticks(pd.date_range(min(x_value), max(x_value), freq= '2W'), rotation= -45, fontsize= FONT_SIZE)
                        plt.yticks([x for x in range(min_value2, max_value2 + 1, step_value)], fontsize= FONT_SIZE)
                        # plt.yticks(fontsize= FONT_SIZE)
                        plt.xlabel('Date', fontsize= FONT_SIZE + 1)
                        plt.ylabel(col_name, fontsize= FONT_SIZE + 1)
                        plt.legend(fontsize= FONT_SIZE + 1)
                        plt.savefig(path + '{}_Cluster_LinePlot_Total.png'.format(col_name))
            else:
                raise FileNotFoundError

        except FileNotFoundError:
            print('최적의 K 결과를 {} 폴더에 넣어주세요'.format(self.best_k_folder))

    def vis_box_plot_day(self, plot_df, num_cluster, column_name, min_value, max_value, save_path):
        tmp_df = plot_df[plot_df['Cluster'] == num_cluster]
        tmp_df['Day'] = tmp_df['REG_DATE'].dt.day
        boxplot_values = [
            tmp_df[tmp_df['Day'] == x][column_name].to_list() for x in range(1, 32)
        ]
        
        step_value = max(int((max_value - min_value) / 20), 1)

        plt.figure(figsize=(20, 8))
        plt.boxplot(boxplot_values)
        plt.title('[{}] Cluster {}'.format(column_name, num_cluster), fontsize= FONT_TITLE)
        plt.xticks(fontsize= FONT_SIZE)
        plt.yticks([x for x in range(min_value, max_value + 1, step_value)], fontsize= FONT_SIZE)
        plt.xlabel('Day', fontsize= FONT_SIZE + 1)
        plt.ylabel(column_name, fontsize= FONT_SIZE + 1)
        plt.savefig(save_path + '{}_Cluster_{}_BoxPlot_Day.png'.format(column_name, num_cluster))

    def vis_box_plot_weekday(self, plot_df, num_cluster, column_name, min_value, max_value, save_path):
        tmp_df = plot_df[plot_df['Cluster'] == num_cluster]
        tmp_df['Weekday'] = tmp_df['REG_DATE'].dt.day_name()
        boxplot_values = [
            tmp_df[tmp_df['Weekday'] == x][column_name].to_list() for x in self.weekday
        ]

        step_value = max(int((max_value - min_value) / 20), 1)

        plt.figure(figsize=(20, 8))
        plt.boxplot(boxplot_values)
        plt.title('[{}] Cluster {}'.format(column_name, num_cluster), fontsize= FONT_TITLE)
        plt.xticks([x+1 for x in range(len(self.weekday))], self.weekday, fontsize= FONT_SIZE)
        plt.yticks([x for x in range(min_value, max_value + 1, step_value)], fontsize= FONT_SIZE)
        plt.xlabel('WeekDay', fontsize= FONT_SIZE + 1)
        plt.ylabel(column_name, fontsize= FONT_SIZE + 1)
        plt.savefig(save_path + '{}_Cluster_{}_BoxPlot_Weekday.png'.format(column_name, num_cluster))

    def vis_box_plot_month(self, plot_df, num_cluster, column_name, min_value, max_value, save_path):
        tmp_df = plot_df[plot_df['Cluster'] == num_cluster]
        tmp_df['Month'] = tmp_df['REG_DATE'].dt.month
        boxplot_values = [
            tmp_df[tmp_df['Month'] == x][column_name].to_list() for x in self.month
        ]

        step_value = max(int((max_value - min_value) / 20), 1)

        plt.figure(figsize=(20, 8))
        plt.boxplot(boxplot_values)
        plt.title('[{}] Cluster {}'.format(column_name, num_cluster), fontsize= FONT_TITLE)
        plt.xticks([x+1 for x in range(len(self.month))], self.month, fontsize= FONT_SIZE)
        plt.yticks([x for x in range(min_value, max_value + 1, step_value)], fontsize= FONT_SIZE)
        plt.xlabel('Month', fontsize= FONT_SIZE + 1)
        plt.ylabel(column_name, fontsize= FONT_SIZE + 1)
        plt.savefig(save_path + '{}_Cluster_{}_BoxPlot_Month.png'.format(column_name, num_cluster))

    def vis_box_plot_season(self, plot_df, num_cluster, column_name, min_value, max_value, save_path):
        tmp_df = plot_df[plot_df['Cluster'] == num_cluster]
        tmp_df['Season'] = tmp_df['REG_DATE'].dt.month // 3 % 4

        season_index = list(set(tmp_df['Season'].values))

        boxplot_values = [
            tmp_df[tmp_df['Season'] == x][column_name].to_list() for x in season_index
        ]

        step_value = max(int((max_value - min_value) / 20), 1)

        plt.figure(figsize=(20, 8))
        plt.boxplot(boxplot_values)
        plt.title('[{}] Cluster #{}'.format(column_name, num_cluster), fontsize= FONT_TITLE)
        plt.xticks([x+1 for x in range(len(season_index))], self.season, fontsize= FONT_SIZE)
        plt.yticks([x for x in range(min_value, max_value + 1, step_value)], fontsize= FONT_SIZE)
        plt.xlabel('Season', fontsize= FONT_SIZE + 1)
        plt.ylabel(column_name, fontsize= FONT_SIZE + 1)
        plt.savefig(save_path + '{}_Cluster_{}_BoxPlot_Season.png'.format(column_name, num_cluster))

    def vis_line_plot(self, plot_df, num_cluster, col_name, min_value, max_value, save_path):
        tmp_df = plot_df[plot_df['Cluster'] == num_cluster]

        x_value = tmp_df['REG_DATE'].to_list()
        y_value = tmp_df[col_name].to_list()
        
        step_value = max(int((max_value - min_value) / 20), 1)

        plt.figure(figsize=(40, 14))
        plt.plot(x_value, y_value)
        plt.title('[{}] Cluster {} Mean Values'.format(col_name, num_cluster), fontsize= FONT_TITLE)
        plt.xticks(pd.date_range(min(x_value), max(x_value), freq= '2W'), rotation= -45, fontsize= FONT_SIZE)
        plt.yticks([x for x in range(min_value, max_value + 1, step_value)], fontsize= FONT_SIZE)
        plt.xlabel('Date', fontsize= FONT_SIZE + 1)
        plt.ylabel(col_name, fontsize= FONT_SIZE + 1)
        plt.savefig(save_path + '{}_Cluster_{}_LinePlot.png'.format(col_name, num_cluster))

    def vis_line_per_cluster_values(self, plot_df, num_cluster, col_name, min_value, max_value, save_path, sample_num = 10):
        plt.figure(figsize= (40, 14))
        x_values = list()

        tmp_df = plot_df[plot_df['Cluster'] == num_cluster]
        sample_list = list(set(tmp_df['GRID_NUM']))[:sample_num]

        for sample in sample_list:
            sample_df = tmp_df[tmp_df['GRID_NUM'] == sample]

            x_values = sample_df['REG_DATE'].to_list()
            y_values = sample_df[col_name].to_list()

            plt.plot(x_values, y_values, label= 'GRID #{}'.format(sample))

        step_value = max(int((max_value - min_value) / 20), 1)

        plt.title('Cluster {}'.format(num_cluster + 1), fontsize= FONT_TITLE)
        plt.xticks(pd.date_range(min(x_values), max(x_values), freq= '2W'), rotation= -45, fontsize= FONT_SIZE)
        plt.yticks([x for x in range(min_value, max_value + 1, step_value)], fontsize= FONT_SIZE)
        plt.xlabel('Date', fontsize= FONT_SIZE + 1)
        plt.ylabel(col_name, fontsize= FONT_SIZE + 1)
        plt.savefig(save_path + '{}_Cluster_{}_LinePlot_Cluster_Values.png'.format(col_name, num_cluster))

    def basic_describe(self):
        if self.best_k_folder not in self.file_list:
            os.mkdir(self.data_path + self.best_k_folder + '/')

        path = self.data_path + self.best_k_folder + '/'
        best_k_files = os.listdir(path)

        try:
            if len(best_k_files) > 1:
                for file in best_k_files:
                    if file != '.DS_Store' and os.path.splitext(file)[1] == '.csv':
                        col_name = file.split('_')[0]
                        cluster_df = pd.read_csv(path + file)
                        describe_df = pd.merge(self.analysis_df, cluster_df, how= 'inner', on= 'GRID_NUM')
                        describe_df = describe_df[['REG_DATE', 'GRID_NUM', col_name, 'Cluster']]
                        
                        cluster_num_df = describe_df[['GRID_NUM', 'Cluster']]
                        cluster_num_df.drop_duplicates(['GRID_NUM'], keep='first', inplace= True)

                        # 월별 기초통계
                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'month', 'mean', path)
                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'month', 'std', path)

                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'month', 'mean', path)
                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'month', 'std', path)

                        # 연별 기초통계
                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'year', 'mean', path)
                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'year', 'std', path)

                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'year', 'all_mean', path)
                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'year', 'all_std', path)

                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'year', 'mean', path)
                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'year', 'std', path)
                        
                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'year', 'all_mean', path)
                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'year', 'all_std', path)

                        # 계절별 기초통계
                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'season', 'mean', path)
                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'season', 'std', path)

                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'season', 'mean', path)
                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'season', 'std', path)

                        # 요일별 기초통계
                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'day', 'mean', path)
                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'day', 'std', path)

                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'day', 'mean', path)
                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'day', 'std', path)

                        # 요일별 기초통계
                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'weekday', 'mean', path)
                        self.sdot_describe(describe_df, cluster_num_df, col_name, 'weekday', 'std', path)

                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'weekday', 'mean', path)
                        self.sdot_describe_cluster(describe_df, cluster_num_df, col_name, 'weekday', 'std', path)

            else:
                raise FileNotFoundError

        except FileNotFoundError:
            print('최적의 K 결과를 {} 폴더에 넣어주세요'.format(self.best_k_folder))

    def sdot_describe(self, describe_df, cluster_num_df, col_name, add_name, method, path):
        save_df = pd.DataFrame()
        column_name = list()

        tmp_df = describe_df[['REG_DATE', 'GRID_NUM', col_name]]
        try:
            if add_name == 'day':
                tmp_df[add_name] = tmp_df['REG_DATE'].dt.day
            
            elif add_name == 'month':
                tmp_df[add_name] = tmp_df['REG_DATE'].dt.month

            elif add_name == 'year':
                tmp_df[add_name] = tmp_df['REG_DATE'].dt.year

            elif add_name == 'season':
                tmp_df[add_name] = tmp_df['REG_DATE'].dt.month // 3 % 4
                column_name = ['GRID_NUM'] + self.season

            elif add_name == 'weekday':
                tmp_df[add_name] = tmp_df['REG_DATE'].dt.day_name()
                column_name = ['GRID_NUM'] + self.weekday

            else:
                raise NameError

        except NameError:
            pass

        try:
            if method == 'mean':
                tmp_df = tmp_df.groupby(['GRID_NUM', add_name]).mean()

            elif method == 'std':
                tmp_df = tmp_df.groupby(['GRID_NUM', add_name]).std()

            elif method == 'all_mean':
                tmp_df = tmp_df.groupby(['GRID_NUM']).mean()
            
            elif method == 'all_std':
                tmp_df = tmp_df.groupby(['GRID_NUM']).std()

            elif add_name in ['all_mean', 'all_std']:
                pass

            else:
                raise NameError

        except NameError:
            print('Please method is onlys "mean" and "std"')

        tmp_df.reset_index(inplace= True, drop= False)
        grid_num = sorted(list(set(tmp_df['GRID_NUM'].values)))

        for num in grid_num:
            tmp_df2 = tmp_df[tmp_df['GRID_NUM'] == num]
            tmp_df2.drop(['GRID_NUM'], axis= 1, inplace= True)
            tmp_df2.rename(columns={col_name : num}, inplace= True)
            tmp_df2.set_index(add_name, inplace= True)
            save_df = pd.concat([save_df, tmp_df2.T])
            
        save_df.reset_index(inplace= True, drop= False)
        if add_name == 'season':
            save_df.rename(columns={
                'index' : 'GRID_NUM',
                0 : self.season[0],
                1 : self.season[1],
                2 : self.season[2],
                3 : self.season[3]
            }, inplace= True)

        else:
            save_df.rename(columns={'index' : 'GRID_NUM'}, inplace= True)

        if len(column_name) > 0:
            save_df = save_df[column_name]

        final_save_df = pd.merge(cluster_num_df, save_df, on= 'GRID_NUM', how= 'outer')
        if method.split('_')[0] == 'all':
            final_save_df.to_excel(path + '{}_{}_{}_GRID_NUM_All.xlsx'.format(col_name, add_name, method), index= False)
        else:
            final_save_df.to_excel(path + '{}_{}_{}_GRID_NUM.xlsx'.format(col_name, add_name, method), index= False)

    def sdot_describe_cluster(self, describe_df, cluster_num_df, col_name, add_name, method, path):
        save_df = pd.DataFrame()
        tmp_df = pd.DataFrame()
        column_name = list()

        tmp_df = describe_df[['REG_DATE', 'Cluster', col_name]]
        try:
            if add_name == 'day':
                tmp_df[add_name] = tmp_df['REG_DATE'].dt.day
            
            elif add_name == 'month':
                tmp_df[add_name] = tmp_df['REG_DATE'].dt.month

            elif add_name == 'year':
                tmp_df[add_name] = tmp_df['REG_DATE'].dt.year

            elif add_name == 'season':
                tmp_df[add_name] = tmp_df['REG_DATE'].dt.month // 3 % 4
                column_name = ['Cluster'] + self.season

            elif add_name == 'weekday':
                tmp_df[add_name] = tmp_df['REG_DATE'].dt.day_name()
                column_name = ['Cluster'] + self.weekday

            elif add_name in ['all_mean', 'all_std']:
                pass

            else:
                raise NameError

        except NameError:
            pass

        try:
            if method == 'mean':
                tmp_df = tmp_df.groupby(['Cluster', add_name]).mean()

            elif method == 'std':
                tmp_df = tmp_df.groupby(['Cluster', add_name]).std()

            elif method == 'all_mean':
                tmp_df = tmp_df.groupby(['Cluster']).mean()
            
            elif method == 'all_std':
                tmp_df = tmp_df.groupby(['Cluster']).std()
                
            else:
                raise NameError

        except NameError:
            print('Please method is onlys "mean" and "std"')

        tmp_df.reset_index(inplace= True, drop= False)
        cluster_num = sorted(list(set(tmp_df['Cluster'].values)))

        for num in cluster_num:
            tmp_df2 = tmp_df[tmp_df['Cluster'] == num]
            tmp_df2.drop(['Cluster'], axis= 1, inplace= True)
            tmp_df2.rename(columns={col_name : num}, inplace= True)
            tmp_df2.set_index(add_name, inplace= True)
            save_df = pd.concat([save_df, tmp_df2.T])
            
        # print(save_df.head())
        save_df.reset_index(inplace= True, drop= False)
        if add_name == 'season':
            save_df.rename(columns={
                'index' : 'Cluster',
                0 : self.season[0],
                1 : self.season[1],
                2 : self.season[2],
                3 : self.season[3]
            }, inplace= True)

        else:
            save_df.rename(columns={'index' : 'Cluster'}, inplace= True)

        if len(column_name) > 0:
            save_df = save_df[column_name]

        # final_save_df = pd.merge(cluster_num_df, save_df, on= 'Cluster', how= 'outer')
        if method.split('_')[0] == 'all':
            save_df.to_excel(path + '{}_{}_{}_Cluster_All.xlsx'.format(col_name, add_name, method), index= False)
        else:
            save_df.to_excel(path + '{}_{}_{}_Cluster.xlsx'.format(col_name, add_name, method), index= False)