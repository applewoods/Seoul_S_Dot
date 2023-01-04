import warnings
warnings.filterwarnings('ignore')

from SDot_Process_1 import Process_1        # Raw Data Cleaning
from SDot_Process_2 import Process_2        # Add GRID_NUM and Make Analysis Data
from SDot_Process_3 import Process_3        # DATA Clustering
from SDot_Process_4 import Process_4        # Result Visualized BoxPlot and Lineplot

def main(data_path):
    print('Process 1 시작')
    p1 = Process_1(data_path)
    p1.load_raw_file()                                                              # '1_RAW_SDot.csv'
    p1.rm_outlier_values(set_rule= 'D', method = 'sigma')                                                          # '2_RM_Outlier.csv

    print('Process 2 시작')
    p2 = Process_2(data_path)
    p2.add_grid_num()                                                               # merge RAW_DATA and GRID_NUM
    p2.select_date_range_raw(
        start_date= '2020.11.19', end_date= '2021.09.02'
    )                                                                               # Select Date range                                                                 
    p2.select_date_range_grid_num(filter_num= 700)                                  # Filtering GRID_NUM and save '3_Analysis_Data.csv'
    p2.vis_analysis_data()                                                          # Visualized Analysis_Data

    print('Process 3 시작')
    p3 = Process_3(data_path)
    p3.load_data()
    p3.filter_nan(num_sigma= 2)                                                      # GRID_NUM의 데이터의 갯수가 filter 보다 작을 경우 버림, '4_filtered_GRID_NUM.csv'
    p3.data_interpolate(method= 'linear')                                           # 결측치 데이터를 보간법을 이용해 NAN 값 처리
    p3.cluster_analysis(year= 2022, min_cluster = 2, max_cluster = 20)              # SDot Cluster 분석

    print('Process 4 시작')
    p4 = Process_4(data_path)
    p4.load_data()
    p4.line_plot()
    p4.box_plot()
    p4.basic_describe()

if __name__ == '__main__':
    main(data_path= './Data/')