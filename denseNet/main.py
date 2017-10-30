import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageGrab
from denseNet.dense_with_selu import Model
from drawnow import drawnow
from tflearn import datasets
# from denseNet.densenet_origin import Model

# datasets.cifar10.load_data('/')





epochs = 200
batch_size = 100
value_num = 3
test_title = 'something'
train_file_list = ['C:\\Users\\WonJuDangbi\\tensorGPUenv_3_5_3\\Scripts\\DeepML\\cifar10\\train_data_' + str(i) + '.csv' for i in range(1, 46)]  # train data 45,000 ��
test_file_list = ['C:\\Users\\WonJuDangbi\\tensorGPUenv_3_5_3\\Scripts\\DeepML\\cifar10\\test_data_' + str(i) + '.csv' for i in range(1, 11)]  # test data 10,000 ��
validation_file_list = ['C:\\Users\\WonJuDangbi\\tensorGPUenv_3_5_3\\Scripts\\DeepML\\cifar10\\validation_data_' + str(i) + '.csv' for i in range(1, 6)]  # validation data 5,000 ��

def scaler(d):
    '''

    :param d: 입력값을 표준정규분포 (m=0, sigma=1)의 형태로 스케일링하는 함수
    :return:
    '''
    d_mean = np.mean(d,axis=0)
    d_std = np.std(d, axis=0)
    d_scale = (d-d_mean)/d_std
    return d_scale



def data_setting(data):
    # x : ������, y : ��
    x = (np.array(data[:, 0:-1]) / 255).tolist()
    y_tmp = np.zeros([len(data), 10])
    for i in range(0, len(data)):
        label = int(data[i][-1])
        y_tmp[i, label - 1] = 1
    y = y_tmp.tolist()

    return scaler(x), y

def read_data(filename):
    ####################################################################################################################
    ## �� Data Loading
    ##  - ������ ���Ͽ� ���� load �� ��ó���� ����
    ####################################################################################################################
    data = np.loadtxt(filename, delimiter=',')
    np.random.shuffle(data)
    return data_setting(data)

def image_screeshot():
    im = ImageGrab.grab()
    im.show()

# monitoring ���� parameter
mon_epoch_list = []
mon_value_list = [[] for _ in range(value_num)]
mon_color_list = ['blue', 'yellow', 'red', 'cyan', 'magenta', 'green', 'black']
mon_label_list = ['loss', 'train_acc', 'val_acc']


def write_csv(title,val_list):
    import pandas as pd
    data = np.array(val_list).transpose()
    p = pd.DataFrame(data)
    p.to_csv(title+'.csv', header=['cost', 'tra_acc', 'valid_acc','time'], encoding='utf-8')


def monitor_train_cost():
    for cost, color, label in zip(mon_value_list, mon_color_list[0:len(mon_label_list)], mon_label_list):
        plt.plot(mon_epoch_list, cost, c=color, lw=2, ls="--", marker="o", label=label)
    plt.title('DenseNet-BC-SELU on CIFAR-10')
    plt.legend(loc=1)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)

# ��� �������� ���� ����ϴ� �Լ�
def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

# ���� ���¸� restore �ϴ� �Լ�
def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}  # inputs : �ش� operation �� �Է� �����͸� ǥ���ϴ� objects
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

########################################################################################################################
## �� Data Training
##  - train data : 50,000 �� (10Ŭ����, Ŭ������ 5,000��)
##  - epoch : 100, batch_size : 100, model : 1��
########################################################################################################################
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# tf.Session(config = config)

with tf.Session() as sess:        #
    # ���� �ð� üũ
    stime = time.time()
    m = Model(sess, 40)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    best_loss_val = np.infty
    check_since_last_progress = 0  # early stopping ������ �������� ���� Ƚ��
    max_checks_without_progress = 100  # Ư�� Ƚ�� ��ŭ ������ �������� ���� ���
    best_model_params = None  # ���� ���� ���� parameter ���� �����ϴ� ����

    print('Learning Started!')

    for epoch in range(epochs):
        epoch_stime = time.time()
        train_accuracy = []
        validation_accuracy = []
        validation_loss = 0.

        '''train part'''
        for index in range(0, len(train_file_list)):
            total_x, total_y = read_data(train_file_list[index])
            for start_idx in range(0, 1000, batch_size):
                train_x_batch, train_y_batch = total_x[start_idx:start_idx+batch_size], total_y[start_idx:start_idx+batch_size]
                if epoch+1 in (50, 75):  # dynamic learning rate
                    m.learning_rate = m.learning_rate/10
                a, _ = m.train(train_x_batch, train_y_batch)
                train_accuracy.append(a)

        '''validation part'''
        for index in range(0, len(validation_file_list)):
            total_x, total_y = read_data(validation_file_list[index])
            for start_idx in range(0, 1000, batch_size):
                validation_x_batch, validation_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
                l, a = m.validation(validation_x_batch, validation_y_batch)
                validation_loss += l / batch_size
                validation_accuracy.append(a)

        '''early stopping condition check'''
        if validation_loss < best_loss_val:
            best_loss_val = validation_loss
            check_since_last_progress = 0
            best_model_params = get_model_params()
            saver.save(sess, 'log/densenet_cifar10_v2.ckpt')
        else:
            check_since_last_progress += 1

        # monitoring factors
        train_acc = np.mean(np.array(train_accuracy))*100
        valid_acc = np.mean(np.array(validation_accuracy))*100

        mon_epoch_list.append(epoch + 1)
        mon_value_list[0].append(validation_loss)
        mon_value_list[1].append(train_acc)
        mon_value_list[2].append(valid_acc)

        epoch_etime = time.time()
        print('epoch :', epoch+1, ', loss :', validation_loss, ', train_accuracy :', np.mean(np.array(train_accuracy)),
              ', validation_accuracy :', np.mean(np.array(validation_accuracy)), ', time :', round(epoch_etime-epoch_stime, 6))
        drawnow(monitor_train_cost)


        if check_since_last_progress > max_checks_without_progress:
            print('Early stopping!')
            break

    print('Learning Finished!')

    # ���� �ð� üũ
    etime = time.time()
    print('consumption time : ', round(etime-stime, 6))

    print('\nTesting Started!')

    if best_model_params:
        restore_model_params(best_model_params)

    test_accuracy = []

    for index in range(0, len(test_file_list)):
        total_x, total_y = read_data(test_file_list[index])
        for start_idx in range(0, 1000, batch_size):
            test_x_batch, test_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
            a = m.get_accuracy(test_x_batch, test_y_batch)
            test_accuracy.append(a)
    test_acc = np.mean(np.array(test_accuracy))
    print('Test Accuracy : ', test_acc)
    print('Testing Finished!')

    write_csv(test_title, mon_value_list)  # 훈련 결과를 csv로 저장