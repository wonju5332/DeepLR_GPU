import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import ImageGrab
from bitcoin.refer_code.CNN_model import *
from drawnow import drawnow



def one_hot_incoder(df):
    return [[1, 0] if y_ == 0 else [0, 1] for y_ in df]


def load_total_data(df):
    '''

    :param df: 매개변수로 받은 메모리형태의 numpy array를 훈련데이터와 검증데이터로 나뉠 데이터
    :return: 훈련데이터 44,000 / 10000
    '''
    np.random.shuffle(df)  #
    train_x, train_y = df[:4000, :-1].reshape([-1,50,50,3]), df[:4000, -1]
    test_x, test_y = df[4000:5000, :-1].reshape([-1,50,50,3]), df[4000:5000, -1]

    # train_x, train_y = df[:43000, :-1].reshape([-1, 50, 50, 3]), df[:43000, -1]
    # test_x, test_y = df[43000:55000, :-1].reshape([-1, 50, 50, 3]), df[43000:55000, -1]
    train_y = one_hot_incoder(train_y)
    test_y = one_hot_incoder(test_y)

    return (train_x, train_y), (test_x, test_y)


def classify_data_set():
    PATH = 'C:\\Users\\WonJuDangbi\\Documents\\image_data\\'
    print('데이터를 불러오고 있습니다.')
    start_time = time.time()
    stack = np.loadtxt(PATH + 'image_data' + '1' + '.csv', delimiter=',')  # csv
    length = 6

    for i in range(2, length + 1):
        data = np.loadtxt(PATH + 'image_data' + str(i) + '.csv', delimiter=',')  # csv
        stack = np.vstack([stack, data])
        print('데이터 로딩 상태 = ', round(i / length * 100, 2), '% 진행 완료')
    last_time = time.time()
    print('총', (last_time - start_time)/60, '분 이 소요되었습니다.')  # 메모리 총 사용량 1.8gb (확인)
    return stack





def image_screeshot():
    im = ImageGrab.grab()
    im.show()

# monitoring 관련 parameter
num_models = 5
mon_epoch_list = []
mon_cost_list = [[] for m in range(num_models)]
mon_color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
mon_label_list = ['model'+str(m+1) for m in range(num_models)]
######################################################################################################
######################################################################################################
######################################################################################################


class RunCNN(object):
    def __init__(self,sess):
        self.models = {}
        self.epochs = 0
        self.last_epochs = None
        self.avg_cost_list = None
        self.mon_cost_list = None     #
        self.mon_epoch_list = []
        self.mon_label_list = None
        self.mon_color_list = None
        self.sess = sess
        self.saver = None
        self.diff = 0
        self.early_stop_count = 0
        self.latest_cost_list = None
        self.ensemble_accuracy = 0.
        self.ensemble_accuracy_old = 0.
        self.model_accuracy = [0., 0., 0., 0., 0.]
        self.batch_try_cnt = 0.
        self.ens_conf_mat = np.zeros((2, 2))
        self.cnt = 0.

    def make_model_list(self,num,name):
        '''

        :param sess: sess을 input으로 받는다.
        :param num: 모델의 갯수를 지정한다.
        :param name: 모델의 이름을 지정한다. ex) model['train'], model['test']
        :return: model생성 및 monitoring 요소 갱신
        '''

        self.models[name] = [Model(self.sess, 'model' + str(i)) for i in range(num)]
        # self.get_set_monitoring_factor(self.models[name])


    def learning_start(self,x, y, batch_size=100):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        print('Learning Started! -- batch_size = {}, X.shape = {}'.format(batch_size,x.shape))
        while True:
            epoch_start_time = time.time()
            self.epochs += 1  # 첫 에폭을 1로 시작한다.
            self.avg_cost_list = np.zeros(len(self.models['train']))
            self.input_batch_unit(x,y,batch_size,name='train')
            # self.cost_monitor_list()
            # drawnow(self.monitor_train_cost) # 그래프 생성
            epoch_end_time = time.time()
            self.early_stopping()
            if self.early_stop_count== 3 or self.epochs > 70:
                self.last_epochs = self.epochs - 1
                print('Epoch: ', '%04d' % (self.epochs), 'cost =', self.avg_cost_list, ' - ', self.diff, ',',epoch_end_time - epoch_start_time)
                print('Learning Finished!')
                break
            else:
                self.latest_cost_list = self.avg_cost_list  #early_stopping을 위해 마지막 코스트값을 할당해고 다음 순환으로 넘어감.
                print('Epoch: ', '%04d' % (self.epochs), 'cost =', self.avg_cost_list, ' - ', self.diff, ',',epoch_end_time - epoch_start_time)

    def input_batch_unit(self, x, y ,batch_size,name):
        length = x.shape[0]


        for start_idx in range(0, length, batch_size):

            x_batch = x[start_idx:start_idx + batch_size]
            y_batch = y[start_idx:start_idx + batch_size]
        #초기화 해놓기 - 검증용 변수
            predictions = np.zeros((batch_size, 2, 3))
            predictions[:, :, 0] = range(0, 2)
            model_result = np.zeros(batch_size, dtype='int32')

            for idx, m in enumerate(self.models[name]):  #train을 name으로 해서 test로 넣을 수 있도록 하자.
                if name =='train':
                    c, *_ = m.train(x_batch, y_batch)
                    self.avg_cost_list[idx] += c / batch_size


                elif name =='test':
                    self.model_accuracy[idx] += m.get_accuracy(x_batch, y_batch)
                    p = m.predict(x_batch)
                    model_result[:] = np.argmax(p, axis=1)  # 5개의 모델의 예측값들 중에서 가장 큰 값을 모델결과에 집어넣는다. (1개당 100개의 예측값이 있으니, 500개를 종합하면 얘네도 100개의 결과)
                    predictions[:, :, 2] += p
                    for idx2, result in enumerate(model_result):  # 100개들을 가지고 순환을 하는데, 이건 뭔뜻이냐..
                        predictions[idx2, result, 1] += 1
            if name=='test':
                self.cnt += 1
                print(self.cnt,'번째 배치입력 하였습니다.')
                self.ensemble_vote_predict(predictions,y_batch)


    def testing_start(self, x, y,batch_size=100, last_epoch = None):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, 'log/epoch_' + str(last_epoch) + '.ckpt')
        self.saver.restore(self.sess, 'C:\\Users\\WonJuDangbi\\PycharmProjects\\DeepLR_GPU\\bitcoin\\refer_code\\log\\epoch_' + str(48) + '.ckpt')
        print('Testing Started!')
        self.input_batch_unit(x, y, batch_size, 'test')


#모니터링 소스
    # def cost_monitor_list(self):
    #     self.mon_epoch_list.append(self.epochs+1)
    #     for idx, cost in enumerate(self.avg_cost_list):
    #         self.mon_cost_list[idx].append(cost)
    #
    #
    # def get_set_monitoring_factor(self,model):
    #     self.mon_cost_list = [[] for _ in range(len(model))]
    #     self.mon_label_list = ['model' + str(m + 1) for m in range(len(model))]
    #     self.mon_color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

    # def monitor_train_cost(self):
    #     for cost, color, label in zip(self.mon_cost_list,
    #                                   self.mon_color_list[0:len(self.mon_label_list)],
    #                                   self.mon_label_list):
    #
    #         plt.plot(mon_epoch_list, cost, c=color, lw=2, ls="--", marker="o", label=label)
    #     plt.title('Epoch per Cost Graph')
    #     plt.legend(loc=1)
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Cost')
    #     plt.grid(True)
    #

    def early_stopping(self):
        self.saver.save(self.sess, 'log/epoch_' + str(self.epochs) + '.ckpt')
        if self.epochs >= 2: # 에폭수가 2개이상일 땐, 비교대상이 생기니까, 비교하는 것이다.
            diff = np.sum(self.latest_cost_list < self.avg_cost_list)  # 5개 vs 5개 를 np.sum한 결과가 2개 이상이라면, early_stop_count +=1이다.
            if diff > 2:
                print('early stopping - epoch({})'.format(self.epochs), ' - ', diff)
                self.early_stop_count += 1  # 최근코스트가 큰 모델이 2개 이상이라면, 카운트1을 증가한다.
                for m in self.models['train']:
                    m.early_stop_count = self.early_stop_count
                    m.epoch = self.epochs





    def ensemble_vote_predict(self, prd,test_y_batch):
        predictions_old = np.array(prd[:, :, 1])  # for get vote-only Accuracy

        # sort
        prd.view('i8, i8, i8').sort(order=['f1', 'f2'], axis=1)
        res = np.array(prd[:, -1, 0], dtype='int64')

        # get Accuracy
        ensemble_correct_prediction = tf.equal(res, tf.argmax(test_y_batch, 1))
        self.ensemble_accuracy += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

        # get vote-only Accuracy
        ensemble_correct_prediction_old = tf.equal(tf.argmax(predictions_old, 1), tf.argmax(test_y_batch, 1))
        self.ensemble_accuracy_old += tf.reduce_mean(tf.cast(ensemble_correct_prediction_old, tf.float32))
        self.ens_conf_mat = tf.add(tf.contrib.metrics.confusion_matrix(labels=tf.argmax(test_y_batch, 1),
                                                                            predictions=res,
                                                                            num_classes=2, dtype='int32',
                                                                            name='confusion_matrix'), self.ens_conf_mat)
    def draw_confusion_matrix(self):
        for i in range(len(self.model_accuracy)):
            print('Model ' + str(i) + ' : ', self.model_accuracy[i]/self.batch_try_cnt)
        print('Ensemble Accuracy : ', self.sess.run(self.ensemble_accuracy)/self.batch_try_cnt)
        print('Testing Finished!')
        print('####### Confusion Matrix #######')
        print(self.sess.run(self.ens_conf_mat))


#메인 실행절
data = classify_data_set()
(train_x, train_y), (test_x, test_y) = load_total_data(data)

with tf.Session() as sess:
    # 시작 시간 체크
    stime = time.time()
     # 학습된 상태를 저장하는 클래스 인스턴스화
    traning_CNN = RunCNN(sess)
    traning_CNN.make_model_list(num = 5, name = 'train')  # RunCNN.models[train] = models_list
    traning_CNN.learning_start(train_x,train_y, batch_size=100)
    etime = time.time()
    print('consumption time : ', round(etime-stime, 6))
tf.reset_default_graph()


with tf.Session() as sess:
    test_CNN = RunCNN(sess)
    test_CNN.make_model_list(num=5,name='test')
    test_CNN.testing_start(test_x,test_y,50)
    test_CNN.draw_confusion_matrix()
    image_screeshot()