import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import ImageGrab
from bitcoin.refer_code.model_factorization import *
from drawnow import drawnow



def one_hot_incoder(df):
    return [[1, 0] if y_ == 0 else [0, 1] for y_ in df]


def load_total_data(df):
    '''

    :param df: 매개변수로 받은 메모리형태의 numpy array를 훈련데이터와 검증데이터로 나뉠 데이터
    :return: 훈련데이터 44,000 / 10000
    '''
    np.random.shuffle(df)  #
    train_x, train_y = df[:44000, :-1].reshape([-1,50,50,3]), df[:44000, -1]
    test_x, test_y = df[44000:, :-1].reshape([-1,50,50,3]), df[44000:, -1]
    train_y = one_hot_incoder(train_y)
    test_y = one_hot_incoder(test_y)

    return (train_x, train_y), (test_x, test_y)


def classify_data_set():
    PATH = 'C:\\Users\\WonJuDangbi\\Documents\\image_data\\'
    stack = np.loadtxt(PATH + 'image_data' + '1' + '.csv', delimiter=',')  # csv
    length = 34

    print('데이터를 불러오고 있습니다.')
    start_time = time.time()

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
        self.avg_cost_list = None
        # self.mon_cost_list = None     #
        # self.mon_epoch_list = []
        # self.mon_label_list = None
        # self.mon_color_list = None
        self.sess = sess
        self.saver = None
        self.diff = 0
        self.early_stop_count = 0
        self.latest_cost_list = None
        self.ensemble_accuracy = 0.
        self.ensemble_accuracy_old = 0.
        self.model_accuracy = [0., 0., 0., 0., 0.]

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
        print('Learning Started!')
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
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
            if self.early_stop_count== 3 or self.epochs > 3:
                print('Epoch: ', '%04d' % (self.epochs), 'cost =', self.avg_cost_list, ' - ', self.diff, ',',epoch_end_time - epoch_start_time)
                print('Learning Finished!')
                break
            else:
                print('Epoch: ', '%04d' % (self.epochs), 'cost =', self.avg_cost_list, ' - ', self.diff, ',',epoch_end_time - epoch_start_time)


    def input_batch_unit(self, x, y ,batch_size,name):
        length = x.shape[0]
        # cnt = 0
        # ensemble_confusion_mat = np.zeros((10, 10))
        # model_result = np.zeros(test_size, dtype='int32')

        for start_idx in range(0, length, batch_size):
            x_batch = x[start_idx:start_idx + batch_size]
            y_batch = y[start_idx:start_idx + batch_size]

            for idx, m in enumerate(self.models[name]):  #train을 name으로 해서 test로 넣을 수 있도록 하자.
                if name =='train':
                    c, *_ = m.train(x_batch, y_batch)
                    self.avg_cost_list[idx] += c / batch_size


                elif name =='test':
                    self.model_accuracy[idx] += m.get_accuracy(x_batch,y_batch)
                    p = m.predict(x_batch)
                    model_result[:] = np.argmax(p, 1)
                    predictions = np.zeros((test_size, 10, 3))
                    predictions[:, :, 0] = range(0, 10)
                    for idx, result in enumerate(model_result):
                        predictions[idx, result, 1] += 1
                    predictions[:, :, 2] += p

    #
    #
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
        if self.epochs < 2:
            self.latest_cost_list = self.avg_cost_list #코스트 값이 1개뿐이라면, 이 변수에 할당한 후,

        elif self.epochs >= 2: # 에폭수가 2개이상일 땐, 비교대상이 생기니까, 비교하는 것이다.
            diff = np.sum(self.latest_cost_list < self.avg_cost_list)  # 5개 vs 5개 를 np.sum한 결과가 2개 이상이라면, early_stop_count +=1이다.
            if diff > 2:
                print('early stopping - epoch({})'.format(self.epochs), ' - ', diff)
                self.early_stop_count += 1  # 최근코스트가 큰 모델이 2개 이상이라면, 카운트1을 증가한다.
                for m in self.models['train']:
                    m.early_stop_count = self.early_stop_count
                    m.epoch = self.epochs
            else:
                self.latest_cost_list = self.avg_cost_list  #비교한 후 다시 현 코스트값을 최근값으로 할당한다.




    def testing_start(self,x,y,epochs=30, batch_size=100):
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, 'log/epoch_' + str(traning_CNN.epochs - 1) + '.ckpt')
        print('Testing Started!')
        self.input_batch_unit(x,y,batch_size, epochs)









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
    # saver = tf.train.Saver()
       test_CNN = RunCNN(sess)
    test_CNN.make_model_list(num= 5 ,name='test')
    test_CNN.testing_start(test_x,test_y,)
    cnt = 0
    ensemble_confusion_mat = np.zeros((10, 10))


    for start_idx in range(0, 1000, batch_size):
        test_x_batch, test_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
        test_size = len(test_y_batch)

        predictions = np.zeros((test_size, 10, 3))  # (100,10,3)
        predictions[:, :, 0] = range(0, 10)

        model_result = np.zeros(test_size, dtype='int32')  # 0 , 0, 0. ... 0 (100,)

        for idx, m in enumerate(models):
            model_accuracy[idx] += m.get_accuracy(test_x_batch, test_y_batch)
            p = m.predict(test_x_batch)
            model_result[:] = np.argmax(p, axis=1)  # 5개의 모델의 예측값들 중에서 가장 큰 값을 모델결과에 집어넣는다. (1개당 100개의 예측값이 있으니, 500개를 종합하면 얘네도 100개의 결과)
            for idx, result in enumerate(model_result):  #100개들을 가지고 순환을 하는데, 이건 뭔뜻이냐..
                predictions[idx, result, 1] += 1
            predictions[:, :, 2] += p

        predictions_old = np.array(predictions[:, :, 1])  # for get vote-only Accuracy

        # sort
        predictions.view('i8, i8, i8').sort(order=['f1', 'f2'], axis=1)
        res = np.array(predictions[:, -1, 0], dtype='int64')

        # get Accuracy
        ensemble_correct_prediction = tf.equal(res, tf.argmax(test_y_batch, 1))
        ensemble_accuracy += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

        # get vote-only Accuracy
        ensemble_correct_prediction_old = tf.equal(tf.argmax(predictions_old, 1), tf.argmax(test_y_batch, 1))
        ensemble_accuracy_old += tf.reduce_mean(tf.cast(ensemble_correct_prediction_old, tf.float32))
        ensemble_confusion_mat = tf.add(tf.contrib.metrics.confusion_matrix(labels=tf.argmax(test_y_batch, 1),
                                                                            predictions=res,
                                                                            num_classes=10, dtype='int32',
                                                                            name='confusion_matrix'),
                                        ensemble_confusion_mat)
        cnt += 1
    for i in range(len(model_accuracy)):
        print('Model ' + str(i) + ' : ', model_accuracy[i] / cnt)
    print('Ensemble Accuracy : ', sess.run(ensemble_accuracy) / cnt)
    print('Testing Finished!')

    print('####### Confusion Matrix #######')
    print(sess.run(ensemble_confusion_mat))