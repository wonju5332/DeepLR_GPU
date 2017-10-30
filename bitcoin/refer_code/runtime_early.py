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


def shffule_total_data(df):
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

################################################################################################################
## ▣ Train Monitoring - Created by 박상범
##  - 실시간으로 train cost 값을 monitoring 하는 기능
################################################################################################################
def monitor_train_cost():
    for cost, color, label in zip(mon_cost_list, mon_color_list[0:len(mon_label_list)], mon_label_list):
        plt.plot(mon_epoch_list, cost, c=color, lw=2, ls="--", marker="o", label=label)
    plt.title('Epoch per Cost Graph')
    plt.legend(loc=1)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)


########################################################################################################################
## ▣ Data Training
##  - train data : 50,000 개 (10클래스, 클래스별 5,000개)
##  - epoch : 20, batch_size : 100, model : 5개
########################################################################################################################
data = classify_data_set()
(train_x, train_y), (test_x, test_y) = shffule_total_data(data)

with tf.Session() as sess:
    # 시작 시간 체크
    stime = time.time()
    models = []
    num_models = 5

    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('Learning Started!')

    # early stopping 관련 parameter

    early_stopping_list = []
    last_epoch = -1
    early_stop_count = 0
    epoch = 0
    batch_size = 100

    train_size = train_x.shape[0] #총 데이터 갯수
    ep_check = 0

    # training -
    while True:
        sstime = time.time()
        avg_cost_list = np.zeros(len(models))
        for idx in range(0, train_size, batch_size):
            train_x_batch, train_y_batch = train_x[idx:idx + batch_size],\
                                           train_y[idx:idx + batch_size]
            for idx, m in enumerate(models):
                c, _ = m.train(train_x_batch, train_y_batch)
                avg_cost_list[idx] += c / batch_size


        mon_epoch_list.append(epoch + 1)
        for idx, cost in enumerate(avg_cost_list):
            mon_cost_list[idx].append(cost)
        drawnow(monitor_train_cost)


        saver.save(sess, 'log/epoch_' + str(epoch + 1) + '.ckpt')
        early_stopping_list.append(avg_cost_list)
        diff = 0
        if len(early_stopping_list) >= 2:
            temp = np.array(early_stopping_list)
            last_epoch = epoch
            diff = np.sum(temp[0] < temp[1])
            if diff > 2:
                print('Epoch: ', '%04d' % (epoch + 1), 'cost =', avg_cost_list, ' - ', diff)
                print('early stopping - epoch({})'.format(epoch + 1))
                ep_check += 1
            early_stopping_list.pop(0)
        epoch += 1
        if epoch == 2:
            break
        eetime = time.time()
        print('Epoch: ', '%04d' % (epoch), 'cost =', avg_cost_list, ' - ', diff, ', epoch{} time'.format(epoch),
              round(eetime - sstime, 2), ', ep_check', ep_check)
    print('Learning Finished!')

    image_screeshot()
    # 종료 시간 체크
    etime = time.time()
    print('consumption time : ', round(etime - stime, 6))

    saver.save(sess, 'log/bitcoin_predict.ckpt')
tf.reset_default_graph()

########################################################################################################################
## ▣ Data Test
##  - test data : 10,000 개
##  - batch_size : 100, model : 5개
########################################################################################################################
with tf.Session() as sess:
    models = []
    num_models = 5
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, 'log/bitcoin_predict.ckpt')

    print('Testing Started!')

    ensemble_accuracy = 0.
    model_accuracy = [0., 0., 0., 0., 0.]
    cnt = 0
    ensemble_confusion_mat = np.zeros((2, 2))
    length = test_x.shape[0]

    for start_idx in range(0, length, batch_size):
        test_x_batch, test_y_batch = test_x[start_idx:start_idx + batch_size],\
                                     test_y[start_idx:start_idx + batch_size]
        test_size = len(test_y_batch)  #100
        predictions = np.zeros(test_size * 2).reshape(test_size, 2)  # (,200) -> (2,100)
        model_predict_result = np.zeros(test_size * 2, dtype=np.int).reshape(test_size, 2) #(,200) -> (2,100)
        model_predict_result[:, 0] = range(0, test_size)  # 0~100까지의 값을 0번째 컬럼에 넣는다.

        for idx, m in enumerate(models):
            print('정확도를 얻는 중입니다.')
            model_accuracy[idx] += m.get_accuracy(test_x_batch, test_y_batch)
            p = m.predict(test_x_batch)
            model_predict_result[:, 1] = np.argmax(p, 1)  #각 테스트데이터의 예측결과값(0또는1)을 넣는다.
            for result in model_predict_result:
                print('결과값에 1씩 더하고 있습니다.')
                predictions[result[0], result[1]] += 1
        cnt += 1
        print('cnt +1 작업이 완료되었습니다.')
        ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y_batch, 1))
        ensemble_accuracy += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
        ensemble_confusion_mat = tf.add(tf.contrib.metrics.confusion_matrix(labels=tf.argmax(test_y_batch, 1),
                                                                            predictions=tf.argmax(predictions, 1),
                                                                            num_classes=2, dtype='int32',
                                                                            name='confusion_matrix'),ensemble_confusion_mat)


    for i in range(len(model_accuracy)):
        print(cnt)
        print('Model ' + str(i) + ' : ', model_accuracy[i] / cnt)
    print('Ensemble Accuracy : ', sess.run(ensemble_accuracy) / cnt)
    print('Testing Finished!')
    print('####### Confusion Matrix #######')
    print(sess.run(ensemble_confusion_mat))
    # print(sess.run(tf.contrib.metrics.confusiusion_matrix(labels=tf.arg_max(train_y, dimension=1), predictions=tf.arg_max(m.predict(total_x), dimension=1), num_classes=10, dtype='int32', name='confusion_matrix')))