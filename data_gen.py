import numpy as np
import scipy.io as sio
import random

# TODO: 这里是一处涉及到路径且可能需要根据实际需要修改的地方
# TODO: 将数据放在data目录下就行，会自动split数据
data_path_prefix = './data'


def load_data(data_sign):
    if data_sign == "Indian":
        data = sio.loadmat('%s/Indian_pines_corrected.mat' % data_path_prefix)['indian_pines_corrected']
        labels = sio.loadmat('%s/Indian_pines_gt.mat' % data_path_prefix)['indian_pines_gt']
    elif data_sign == "Pavia":
        data = sio.loadmat('%s/PaviaU.mat' % data_path_prefix)['paviaU']
        labels = sio.loadmat('%s/PaviaU_gt.mat' % data_path_prefix)['paviaU_gt']
    elif data_sign == "Honghu":
        data = sio.loadmat('%s/WHU_Hi_HongHu.mat' % data_path_prefix)['WHU_Hi_HongHu']
        labels = sio.loadmat('%s/WHU_Hi_HongHu_gt.mat' % data_path_prefix)['WHU_Hi_HongHu_gt']
    return data, labels


def gen(data_sign, train_num_per_class=10, max_percent=0.5):
    data, labels = load_data(data_sign)
    h, w, c = data.shape
    class_num = labels.max()
    class2data = {}
    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:
                if labels[i, j] in class2data:
                    class2data[labels[i, j]].append([i, j])
                else:
                    class2data[labels[i, j]] = [[i, j]]

    TR = np.zeros_like(labels)
    TE = np.zeros_like(labels)
    for cl in range(class_num):
        class_index = cl + 1
        ll = class2data[class_index]
        all_index = list(range(len(ll)))
        real_train_num = train_num_per_class
        if len(all_index) <= train_num_per_class:
            real_train_num = int(len(all_index) * max_percent)
        select_train_index = set(random.sample(all_index, real_train_num))
        for index in select_train_index:
            item = ll[index]
            TR[item[0], item[1]] = class_index
    TE = labels - TR
    target = {}
    target['TE'] = TE
    target['TR'] = TR
    target['input'] = data
    return target


def gen_as_precent(data_sign, train_num_percent=0.1, lowest_data=5):
    data, labels = load_data(data_sign)
    h, w, c = data.shape
    class_num = labels.max()
    class2data = {}
    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:
                if labels[i, j] in class2data:
                    class2data[labels[i, j]].append([i, j])
                else:
                    class2data[labels[i, j]] = [[i, j]]

    TR = np.zeros_like(labels)
    TE = np.zeros_like(labels)
    for cl in range(class_num):
        class_index = cl + 1
        ll = class2data[class_index]
        all_index = list(range(len(ll)))
        pre_num = int(len(ll) * train_num_percent)
        real_train_num = pre_num if pre_num >= lowest_data else lowest_data
        select_train_index = set(random.sample(all_index, real_train_num))
        for index in select_train_index:
            item = ll[index]
            TR[item[0], item[1]] = class_index
    TE = labels - TR
    target = {}
    target['TE'] = TE
    target['TR'] = TR
    target['input'] = data
    return target


def generate_data(data_sign, train_class=10):
    if train_class < 1:
        target = gen_as_precent(data_sign, train_class)
    else:
        target = gen(data_sign, int(train_class))
    train_num_str = str(train_class)
    # TODO: 这里是一处涉及到路径且可能需要根据实际需要修改的地方
    save_path = './data/%s/%s_%s_split.mat' % (data_sign, data_sign, train_num_str)
    sio.savemat(save_path, target)
    return True
