import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

class MakeData2Graph(object):
    FILE_PATH = 'E:\\data\\bitcoin_data\\'

    @classmethod
    def classfy_data_range_unit(cls, csv_path):
        '''

        :param csv: str형태로 들어오게해서, 파일이 6개라면 함수가 6번 호출될 수 있도록 하자.

        ----------

        refine_table = Nan값을 0으로 교체한다.
        ----------

        :return: shape -> ( ? , 3 )
        '''
        table = pd.read_csv(MakeData2Graph.FILE_PATH + csv_path)
        refine_table = table[['Close','High','Low','Volume_(BTC)']].dropna()
        return refine_table  #

    @classmethod
    def table_to_data_with_normalize(cls, table):
        newdf = (table - table.min()) / (table.max() - table.min())
        return newdf - newdf.min()

    @classmethod
    def section_data_normalize(cls, data):
        newdf = (data - np.mean(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0)+1e-9)
        return newdf - np.min(newdf, axis=0)

    @classmethod
    def make_label_data(cls, df, idx, end):
        try:
            pre = df[idx+30,0] # t
            post = df[idx+35,0] # t+5


            if idx+31 < df.shape[0]:
                if  pre <= post:  #비트코인가격이 올랐다.
                    return 'up'
                elif pre > post:
                    return 'down'

            elif idx+31 >= df.shape[0]:
                if  pre <= post:  #비트코인가격이 올랐다.
                    return 'up'
                elif pre > post:
                    return 'down'
        except IndexError:
            pass

    @classmethod
    def make_y_data(cls, df, idx, end):
        if idx+31 < df.shape[0]:
            return df[idx:idx+30,1:4]
        elif idx+31 >= df.shape[0]:
            return df[idx:end,1:4]  #나머지 잔반처리랄까.



####################################################################

def execute_data_handling():
    table = MakeData2Graph.classfy_data_range_unit('bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv')
    y_data = []
    label = []
    length = int(table.shape[0])
    for i in range(0,length,30):
        temp = table.values
        label_data = MakeData2Graph.make_label_data(temp,i,length)  # label값 생성
        label.append(label_data)
        # y_val = MakeData2Graph.make_y_data(temp,i,length)
        # y_data.append(y_val)  # 30개씩, 94000...개 정도 들어가 있음. (한 csv당)
    return label


#################### 라벨 생성기################

labels = execute_data_handling()
with open('C:\\Users\\WonJuDangbi\\Documents\\label\\label.csv','w') as w:
    for i in labels:
        w.write(str(i)+'\n')

