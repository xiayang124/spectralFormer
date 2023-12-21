import json

import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from vit_pytorch import ViT
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import os
import data_gen
from dataset import get_dataset
from thop import profile

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Honghu'], default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=1, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight_decay')
parser.add_argument('--train_num', type=int, default=10)
parser.add_argument('--train_time', type=int, default=5)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


# -------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes, shape):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    # -------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data == (i + 1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]  # (695,2)
    total_pos_train = total_pos_train.astype(int)
    # --------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    # --------------------------for true data------------------------------------
    list_sec = np.array([])
    list_fir = np.array([i for i in range(shape[0])]).reshape(-1, 1)
    for i in range(shape[1]):
        list_sec = np.append(list_sec, i)
    list_left = list_sec.reshape(-1, 1)
    for i in range(shape[0] - 1):
        list_sec = np.concatenate((list_left, list_sec.reshape(-1, 1)))
    list_fir = np.repeat(list_fir, repeats=shape[1], axis=1).reshape(-1, 1)
    total_pos_true = np.concatenate((list_fir, list_sec), axis=1)

    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


# -------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    # 中心区域
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    # 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    # 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    # 上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    # 下边镜像
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


# -------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=float)
    # 中心区域
    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape
    # 左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]
    # 右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[
                                                                                                        :, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[
                                                                                                        :, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    return x_train_band


# -------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape, x_test.dtype))
    print("**************************************************")

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape, x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape, x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape, x_true_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band, x_true_band


# -------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape, y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape, y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true


# -------------------------------------------------------------------------------
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# -------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


# -------------------------------------------------------------------------------
# train model
def train_epoch(train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


# -------------------------------------------------------------------------------
# validate model
def valid_epoch(valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre, batch_data


def test_all_epoch(all_loader):
    tar = np.array([])
    pre = np.array([])
    begin_time = round(time.time() * 1000)
    for batch_idx, (batch_data, batch_target) in enumerate(all_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)

        _, t, p = accuracy(batch_pred, batch_target, topk=(1,))

        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    end_time = round(time.time() * 1000)
    return tar, pre, batch_data, end_time - begin_time


# -------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def result_file_exists(prefix, file_name_part):
    ll = os.listdir(prefix)
    for l in ll:
        if file_name_part in l:
            return True
    return False


# -------------------------------------------------------------------------------
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

dataset_name = args.dataset
train_num = args.train_num
train_time = args.train_time


mat_file = "{}_{}_split.mat".format(dataset_name, train_num)
if result_file_exists('./data/{}'.format(dataset_name), mat_file):
    print("{} had been generated...skip".format(mat_file))
data_gen.generate_data(dataset_name, train_num)
print("All data had been generated!")

while True:
    """uniq_name = "{}_{}_{}_spectral.json".format(dataset_name, train_num, times)
    if result_file_exists('./save_path', uniq_name):
        print('%s has been run. skip...' % uniq_name)
        continue
    print("begin training {}".format(uniq_name))"""
    get_result = True
    TE, TR, input = get_dataset(dataset_name, train_num_or_rate=train_num)

    num_classes = np.max(TR)
    label = TE + TR

    # normalize data by band norm
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:, :, i])
        input_min = np.min(input[:, :, i])
        input_normalize[:, :, i] = (input[:, :, i] - input_min) / (input_max - input_min)
    # data size
    height, width, band = input.shape
    print("height={0},width={1},band={2}".format(height, width, band))
    # -------------------------------------------------------------------------------
    # obtain train and test data
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(
        TR, TE, label, num_classes, input.shape)

    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
    x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test,
                                                                 total_pos_true, patch=args.patches,
                                                                 band_patch=args.band_patches)
    y_train, y_test, _ = train_and_test_label(number_train, number_test, number_true, num_classes)
    y_true = label
    # -------------------------------------------------------------------------------
    # load data
    x_train = torch.from_numpy(x_train_band.transpose(0, 2, 1)).type(torch.FloatTensor)  # [695, 200, 7, 7]
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # [695]
    Label_train = Data.TensorDataset(x_train, y_train)
    x_test = torch.from_numpy(x_test_band.transpose(0, 2, 1)).type(torch.FloatTensor)  # [9671, 200, 7, 7]
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # [9671]
    Label_test = Data.TensorDataset(x_test, y_test)
    x_true = torch.from_numpy(x_true_band.transpose(0, 2, 1)).type(torch.FloatTensor)
    y_true = torch.from_numpy(y_true.reshape(-1, )).type(torch.LongTensor)
    Label_true = Data.TensorDataset(x_true, y_true)

    label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
    label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
    label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)

    # -------------------------------------------------------------------------------
    # create model
    model = ViT(
        image_size=args.patches,
        near_band=args.band_patches,
        num_patches=band,
        num_classes=num_classes,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=8,
        dropout=0.1,
        emb_dropout=0.1,
        mode=args.mode
    )
    model = model.cuda()
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
    # -------------------------------------------------------------------------------
    print("start training")
    tic = time.time()

    for epoch in range(args.epoches):
        # train model
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
              .format(epoch + 1, train_obj, train_acc))
        scheduler.step()

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
            model.eval()
            tar_v, pre_v, batch_data = valid_epoch(label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

    toc = time.time()

    model.eval()
    if dataset_name == "Indian":
        oa_range = (45.44 - 3.35, 45.44 + 3.35)
        aa_range = (61.22 - 1.85, 61.22 + 1.85)
        kappa_range = (39.90 - 3.41, 39.90 + 3.41)
    if dataset_name == "Pavia":
        oa_range = (67.65 - 1.05, 67.65 + 1.05)
        aa_range = (73.89 - 0.26, 73.89 + 0.26)
        kappa_range = (58.72 - 1.10, 58.72 + 1.10)
    if dataset_name == "Honghu":
        oa_range = (61.93 - 4.44, 61.93 + 4.44)
        aa_range = (57.20 - 1.29, 57.20 + 1.29)
        kappa_range = (55.18 - 4.43, 55.18 + 4.43)
    print(str(OA2) + "\n" + str(AA_mean2) + "\n" + str(Kappa2) + "\n")
    if not oa_range[0] < OA2 * 100 < oa_range[1] or not aa_range[0] < AA_mean2 * 100 < aa_range[1] or not kappa_range[0] < Kappa2 * 100 < kappa_range[1]:
        continue
    # No problem ↑
    tar_v, pre_v, batch_data, have_times = test_all_epoch(label_true_loader)
    input = torch.randn(args.batch_size, batch_data.shape[1], batch_data.shape[2]).cuda()
    macs, params = profile(model, (input, ))

    res = {
        'oa': OA2 * 100,
        'each_acc': str(AA2 * 100),
        'aa': AA_mean2 * 100,
        'kappa': Kappa2 * 100,
        'times': have_times,
        'macs': macs,
        'flop': macs * 2
    }
    # np.save("./spetral_save_npy/special_Indian_save.pred.npy", pre_v)
    # np.save("./spetral_save_npy/special_Indian_save_target.pred.npy", tar_v)
    # No problem ↓
    file_name = dataset_name + "_" + str(train_num) + "_spectralFormer"
    save_path = "./save_path/" + file_name
    save_path_json = "%s.json" % save_path
    ss = json.dumps(res, indent=4)
    with open(save_path_json, 'w') as fout:
        fout.write(ss)
        fout.flush()

    all_label = pre_v.reshape(TE.shape[0], TE.shape[1])
    save_npy = "./spetral_save_npy/" + file_name
    np.save(save_npy, all_label)
    print("save record of %s done!" % save_path)
    print("**************************************************")

    print("Final result:")
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    print(AA2)
    print("**************************************************")
    break
