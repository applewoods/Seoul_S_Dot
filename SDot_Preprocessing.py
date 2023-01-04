import pandas as pd
import numpy as np

class RM_Outlier:
    def __init__(self, dataframe):
        self.df = dataframe

        self.df['REG_DATE'] = pd.to_datetime(self.df['REG_DATE'])
        self.df.set_index('REG_DATE', inplace= True)

    def remove_outlier(self, set_rule= 'D', set_group= 'EQUI_INFO_KEY', method= 'sigma'):
        try:
            if method.lower() == 'sigma':
                result_df = self.df.groupby(set_group).resample(set_rule).apply(self.remover_sigma)
                result_df = pd.DataFrame(result_df)
                # result_df.drop(set_group, axis= 1, inplace= True)
                result_df.reset_index(drop= False, inplace= True)
                return result_df

            elif method.lower() == 'box':
                result_df = self.df.groupby(set_group).resample(set_rule).apply(self.remover_box)
                result_df = pd.DataFrame(result_df)
                # result_df.drop(set_group, axis= 1, inplace= True)
                result_df.reset_index(drop= False, inplace= True)
                return result_df
            
            else:
                raise NameError("['sigma', 'box']")
        
        except NameError as e:
            print('Please Select {}'.format(e))

    def remover_box(self, num, rev_range= 1.5):
        # rev_range : 제거 범위 조절 변수
        # 참고자료 : https://lifelong-education-dr-kim.tistory.com/16
        quantile_value = 0.25
        level_1q = num.quantile(quantile_value)
        level_3q = num.quantile(1 - quantile_value)
        IQR = level_3q - level_1q

        result_value = num[num.between(level_1q - (rev_range * IQR), level_3q + (rev_range * IQR))]
        return np.nanmean(result_value)

    def remover_sigma(self, num, num_sigma= 1):
        std_value = np.nanstd(num)
        mean_value = np.nanmean(num)
        sigma = num_sigma * std_value

        result_value = num[num.between(mean_value - sigma, mean_value + sigma)]

        return np.nanmean(result_value)

