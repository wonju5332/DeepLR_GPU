import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math
import re
import csv
import os


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
    def make_label_data(cls, df, idx):
        try:
            pre = df[idx+30,0] # t
            post = df[idx+35,0] # t+5


            if idx+31 < df.shape[0]:
                return 'up' if np.log(post/pre) > 0 else 'down'

            elif idx+31 >= df.shape[0]:
                return 'up' if np.log(post / pre) > 0 else 'down'
        except IndexError:
            pass

    @classmethod
    def make_y_data(cls, df, idx, end):
        if idx+31 < df.shape[0]:
            return df[idx:idx+30,1:4]
        elif idx+31 >= df.shape[0]:
            return df[idx:end,1:4]  #나머지 잔반처리랄까.
    @classmethod
    def save_data_to_fig(cls,y_data):
        SAVE_PATH = 'C:\\Users\\WonJuDangbi\\Documents\\fig\\fig_new\\'
        for i in range(0,55029):
            new = MakeData2Graph.section_data_normalize(y_data[i])
            fig = plt.figure(i, frameon=False, figsize=[5, 5], dpi=10, facecolor='w')
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            ax.clear()
            plt.plot(new[:, 0], color='red', lw=5)
            plt.plot(new[:, 1], color='green', lw=5)
            plt.plot(new[:, 2], color='blue', lw=5)
            plt.savefig(SAVE_PATH + str(i + 1) + '.png', transparant=True, pad_inches=0)
            plt.cla()
            plt.close()

####################################################################

class DataToMatrix(object):
    def __init__(self):
        self.IMAGE_PATH = 'C:\\Users\\WonJuDangbi\\Documents\\fig\\fig\\'
        self.__LABEL_PATH = 'C:\\Users\\WonJuDangbi\\Documents\\label\\'
        self.__train_labels = None
        self.__labels = {'up':1,'down':0}
        self.__image_data = []
        self.__rgb_cnt = 0
        self.__save_cnt = 0

    #################### 라벨 생성기################
    def execute_data_handling(self):
        table = MakeData2Graph.classfy_data_range_unit('bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv')
        label = []
        y_data = []
        temp = table.values
        length = int(table.shape[0])
        for i in range(0,length,30):
            label_data = MakeData2Graph.make_label_data(temp,i)  # label값 생성
            label.append(label_data)
            y_val = MakeData2Graph.make_y_data(temp,i,length)  # 변수 3개데이터를 리스트에 추가하는 작업.
            y_data.append(y_val)  # 30개씩, 94000...개 정도 들어가 있음. (한 csv당)
        return label, y_data



    def write_label_csv(self):
        labels , _ = self.execute_data_handling()
        with open(self.__LABEL_PATH+'label_last.csv','w') as w:
            for i in labels:
                w.write(str(i)+'\n')


    def _setLabel(self):
        file = open(self.__LABEL_PATH+'label.csv', 'r')
        reader = csv.reader(file, delimiter=',')
        list = ['dummy']
        for r in reader:
            list.append(r[0])
        return list

    ####################################################

    def png_to_matrix_with_label(self):
        self.__train_labels = self._setLabel()
        for name in [filename for filename in os.listdir(self.IMAGE_PATH)]:
            try:
                img = mpimg.imread(self.IMAGE_PATH + str(name))
                img = img[...,:3]  # PNG는 RGBA값으로 나온다. A를 제외한 RGB만 따로 뽑을 것.
                # print(img.flatten().shape)  # 10000,0
                f = re.split('[.]', name)
                label = self.__train_labels[int(f[0])]   #up
                label_num = self.__labels[label]  # label=1이면
                rgb_value = ','.join([str(i) for i in img.flatten().tolist()])

                self.__image_data.append([rgb_value, label_num])
            except OSError as e:
                print(str(name) + ', 이미지 식별 불가능', e)
                continue

            self.__rgb_cnt += 1
            if self.__rgb_cnt % 1000 == 0:
                self.__save_cnt += 1
                self._data_to_file()
                self.__image_data.clear()
        self.__save_cnt += 1
        self._data_to_file()
        self.__image_data.clear()
        self.__rgb_cnt = 0


    def _data_to_file(self):
        '''
            rgb 정보와 label을 파일로 기록하는 함수.
        '''
        print('데이터를 저장하는 중입니다.')

        for data in self.__image_data:
            with open(self.__LABEL_PATH + 'image_data' + str(self.__save_cnt) + '.csv', 'a', encoding='utf-8') as f:
                f.write(data[0] + ',')
                f.write(str(data[1]) + '\n')
        print('데이터 저장이 완료되었습니다.')


    def img_to_csv(self):
        # 썸네일로 바꾼 이미지를 바꾸어 csv로 저장.
        print('rgb to gray start.')
        self.png_to_matrix_with_label()
        print('rgb to gray end.')



# data = shffule_total_data()
# (train_x, train_y) , (test_x,testy) = classify_data_set(data)
# print(train_x.shape, train_y.shape) (30000, 7500) (30000,)


crawler = DataToMatrix()
# crawler.write_label_csv()
# _, y_data = crawler.execute_data_handling()
crawler.img_to_csv()
# MakeData2Graph.save_data_to_fig(y_data)