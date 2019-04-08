# -*- coding:utf-8 -*-
__author__ = 'XF'

import jieba.posseg
import json
import os.path
import re
import time
if __name__ == "__main__":
    """
 
    fullMode = jieba.lcut(str1, cut_all = True, HMM=True)
    print(type(fullMode))
    print("fullMode:", fullMode)
    precisionMode = jieba.lcut(str1, cut_all = False)
    print(type(precisionMode))
    print("precisionMode:", precisionMode)
   
    # json.dumps()
    data = {
        'name': 'ACM',
        'day': '2018-11',
        'place': 'beijing'
    }
    json_str  = json.dumps(data)
    print(json_str)
    print(type(json_str))

    # json.load()

    with open('conf.json', 'r') as f:
        temp = json.load(f)
        print(temp)  # 打印
        print(type(temp))  # 取出特定键的值
"""
    # filesName = os.listdir("F:\\巨杉数据库\\a.txt")
    # print(filesName)
    """
     try:
        with open(r"stopWords.txt", 'r', encoding="utf-8") as f:
            content = f.readlines()
            i = 0
            for word in content:
                content[i] = word.strip()
                i = i + 1
            if "。".encode(encoding="gbk").decode(encoding="gbk") in content:
                print("yes")
    except IOError:
        print("IOError")

    print("。".encode().decode())
"""
    """
    str1 = "沈阳 1 0.002127659574468085"
    list1 = str1.split(" ")
    print(list1)
    print(type(list1))
#! --*--coding:utf8--*--

import numpy as np
from sklearn import svm
from Data import Origin, Label
# from feature import Model, FeatureCalculator
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import sklearn
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import random


def cal_distance(p, default):
    d = np.float(0)
    if p and default:
        for key, val in default.items():
            d = d + (val - p[key]) ** 2
        d = np.sqrt(d)
        return d
    return 0.0


def get_traning_src(src, k=8):
    m, n = src.shape
    for x in range(0, m - k):
        for y in range(0, n - k):
            flag = True
            for i in range(x, x + k):
                if not flag:
                    break
                for j in range(y, y + k):
                    if src[i, j] == 255:
                        flag = False
                        break
            if flag:
                return src[x: x + k, y: y + k]
    return None


meanf = {
    'mean': 31.9097977107,
    'stddv': 15.8324457838,
    'skewness': 0.837149647467,
    'kurtosis': 0.975928584471,
    'smooth': 0.994324675294,
    'energy': 0.0022990047125089,
    'entropy': 9.397126203719155,
    'relative': -0.0315046037301272,
    'ncj': 0.1770933256578241,
    'threehug': 405.98356877616584,
    'entro': 0.029649774727759273,
    'evenness': 0.002720645040439284
}

meanf_tumour = {
    'energy': 0.00712605302964,
    'relative': -0.151798749511,
    'ncj': 0.173470297299,
    'entropy': 8.25561962566,
}

F9 = ('skewness', 'energy', 'ncj', 'kurtosis', 'threehug', 'smooth', 'entropy', 'contrast', 'mean')


def compare(a, b):
    if a - b > 0.0:
        return 1
    return -1


def cal_thresh(distances, percent):
    res = sorted(distances, cmp=compare)
    return res[int(len(res) * percent - 1)]


def count(src, gray=255):
    counter = 0
    for i in src:
        for j in i:
            if j == gray:
                counter = counter + 1
    return counter


def trainData(filename, start, end, deli=","):
    dataset = np.loadtxt(filename, delimiter=deli)
    X = dataset[:, start:end]  # 仅仅是特征值
    y = dataset[:, -1]  # 对应于X中样本的判定值
    return X, y


def train(model, X, y):
    model.fit(X, y)  # 训练

    def judge(data):
        predicted = model.predict([[data[key] for key in F9]])
        if predicted[0]:
            return True
        return False

    return judge


class Classify:
    def __init__(self, perc=None, judgefunc=None, filestr=None, dislist=None, center=None):
        '''

        :param perc: 阈值
        :param judgefunc: 判断函数
        :param filestr: 距离文件，应与阈值一同出现
        :param dislist: 距离列表，应与阈值一同出现
        :param center: 中心点，应与阈值一同出现
        '''
        if perc:
            self.getjudgeline(perc, filestr, dislist)
        if judgefunc:
            self.judge = judgefunc
        if center:
            self.mf = center

    def getjudgeline(self, perc, filestr=None, dislist=None):
        if not dislist:
            dislist = np.loadtxt(filestr)
        self.judgeline = cal_thresh(dislist, perc)
        return self.judgeline

    def judge(self, mf1):
        dis = cal_distance(mf1, self.mf)
        if dis < self.judgeline:
            return True
        return False

    def handleimg(self, img, u=4, w=4, rely=None, bg=0):
        left = 105
        top = 80
        width = 288
        height = 322
        dst = np.zeros((512, 512), np.uint8)
        fill = 255
        if bg == 255:
            fill = 0
            for i in range(512):
                for j in range(512):
                    dst[i][j] = 255
        for i in range(top, top + height, u):
            for j in range(left, left + width, u):
                src = rely[i:i + w, j:j + w]
                if count(src) >= w * w / 2:
                    continue
                src = img[i:i + w, j:j + w]
                # if dst[i][j] != 255:
                #     continue
                mdl = FeatureCalculator(src, 2)
                if not self.judge(mdl.all):
                    dst[i:i + w, j:j + w] = fill
                else:
                    dst[i:i + w, j:j + w] = src
                print (i * width + j) * 100 // (width * height)

        return dst


def cal_traning_thresh(filename, select, mf, w=8):
    # 255-肿瘤
    # 128-壁
    with open(filename, 'w+') as fwrite:
        perc = 0
        k = 1
        n = 0
        imgrange = range(1, 2001)
        if select == 255:
            imgrange = np.loadtxt('tumour-index.txt', np.int)
        for i in imgrange[301:601]:
            mm = Model(i, select)
            cut = get_traning_src(mm.src, w)
            try:
                cutm = FeatureCalculator(cut, 2)
                v = cal_distance(cutm.all, mf)
            except Exception, e:
                print e
                print i, mm.src
                continue
            fwrite.write(str(v))
            fwrite.write('\n')
            n = n + 1
            if k != n // 3:
                k = n // 3
                perc = perc + 1
                print "%d completed." % perc
        fwrite.close()


def newTrainData(tumour, pgb, deli=",", percent=0.7):
    tumour_all = np.loadtxt(tumour, delimiter=deli)
    t_row, col = tumour_all.shape
    t_length = t_row
    t_row = int(t_row * percent)
    pgb_all = np.loadtxt(pgb, delimiter=deli)
    p_length = len(pgb_all)
    p_row = int(p_length * percent)

    train_data = np.zeros((t_row + p_row, col), np.float)
    test_length_t = t_length - t_row
    test_length_p = p_length - p_row
    test_data = np.zeros((test_length_t + test_length_p, col), np.float)
    t_position = random.randint(1, t_length-1)
    p_position = random.randint(1, p_length-1)

    temp = t_length - t_position
    if t_position > t_length / 2:
        train_data[0:temp, :] = tumour_all[t_position:t_length, :]
        train_data[temp:t_row, :] = tumour_all[0:t_row - temp, :]
        test_data[0:test_length_t, :] = tumour_all[t_row - temp:test_length_t + t_row - temp, :]
    else:
        train_data[0:temp, :] = tumour_all[0:temp, :]
        train_data[temp:t_row, :] = tumour_all[t_length - t_row + temp:t_length, :]
        #print(test_length_t, temp)
        test_data[0:test_length_t, :] = tumour_all[temp:test_length_t + temp, :]

    temp = p_length - p_position
    if p_position > p_length / 2:
        train_data[t_row:t_row + temp, :] = pgb_all[p_position:p_length, :]
        train_data[t_row + temp:t_row + p_row, :] = pgb_all[0:p_row - temp, :]
        test_data[test_length_t:test_length_t + test_length_p, :] = pgb_all[p_row - temp:test_length_p + p_row - temp,
                                                                    :]
    else:
        train_data[t_row:t_row + temp, :] = pgb_all[0:temp, :]
        train_data[t_row + temp:t_row + p_row, :] = pgb_all[p_length - p_row + temp:p_length, :]
        test_data[test_length_t:test_length_p + test_length_t, :] = pgb_all[temp:test_length_p + temp]

    return train_data, test_data


def newTrain(X, y, model):
    model.fit(X, y)
    with open("SVC_model.txt", "w") as f:
        f.write(str(model))
    f.close()
    return None


#def crossTest(train_start, test_start, num):



    
    temp1 = np.loadtxt("all-PGB-13.txt", delimiter=",")
    temp2 = np.loadtxt("all-tumour-13.txt", delimiter=",")
    temp3 = np.zeros([2*num, 13], np.float)
    temp4 = np.zeros([2*num, 1], np.int)
    temp3[0:num, 0:13] = temp1[train_start:train_start+num, 0:13]
    temp3[num:2*num, 0:13] = temp2[train_start:train_start+num, 0:13]
    temp4[0:num, 0] = temp1[train_start:train_start+num, -1]
    temp4[num:2 * num, 0] = temp2[train_start:train_start + num, -1]

    temp5 = np.zeros([2*num, 13], np.float)
    temp6 = np.zeros([2*num, 1], np.int)
    temp5[0:num, 0:13] = temp1[test_start:test_start+num, 0:13]
    temp5[num:2*num, 0:13] = temp2[test_start:test_start+num, 0:13]
    temp6[0:num, 0] = temp1[test_start:test_start+num, -1]
    temp6[num:2 * num, 0] = temp2[test_start:test_start + num, -1]
    return temp3, temp4, temp5, temp6


if __name__ == '__main__':
    # callist = np.loadtxt('tumour-distances.txt')
    # print cal_thresh(callist, 0.3)

    # cal_traning_thresh('tumour-distances.txt', 255, meanf_tumour, 4)

    import cv2

    # n = 0
    # for i in range(100, 2001, 100):
    #     classify = Classify(0.2, filestr='tumour-distances.txt')
    #     res = classify.handleimg(Origin(i).src, 2, 4)
    #     cv2.imshow(str(i), res)
    #     cv2.imwrite('%d.png'%i,res)
    #     n = n+1
    #     print str(n)+" completed."
    # print "completed.."
    # classify = Classify(0.8, filestr='tumour-distances.txt')
    # res = classify.handleimg(Origin(922).src, meanf_tumour, 2, 4)
    # cv2.imshow('1', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # X, y = trainData('tumour-13-train.txt', 0, 1)
    # print(type(y))
    # ab = Classify(judgefunc=train(SVC(), X, y))
    # label = Label(351)
    # label.threshold(methods=cv2.THRESH_BINARY_INV)
    # res = ab.handleimg(Origin(351).src, 1, 8, label.src)
    # cv2.imshow('1', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # img = Origin(1).src
    # cv2.imshow("IMG", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # X, y = newTrainData("tumour-13-train.txt", "PGB-13-train.txt", 0, 13)
    # X1 = np.loadtxt("PGB-13-test.txt", delimiter=",")
    # #newTrain(X, y, SVC())
    # predicted = model.predict(X)
    # print(model)

    #train_data, test_data = newTrainData("all-tumour-13.txt", "all-PGB-13.txt")
    # train_x, train_y, test_x, test_y = crossTest(0, 1, 7000)
    # model = KNeighborsClassifier()
    # model.fit(train_x, train_y)
    # p = model.predict(test_x)
    # print(sklearn.metrics.classification_report(test_y, p, labels=[1, 0], target_names=["tumour", "pgb"], digits=4))
    # print(metrics.confusion_matrix(test_y, p))
    #用所有的样本做训练

    # data1 = np.loadtxt("all-PGB-13.txt", delimiter=",")
    # data2 = np.loadtxt("all-tumour-13.txt", delimiter=",")

    # train_x = np.zeros([5000+5000, 13], np.float)
    # y = np.zeros([5000+5000], np.int)
    # train_x[0:len(data1), :] = data1[:, 0:13]
    # y[0:len(data1)] = data1[:, -1]
    # train_x[len(data1):len(data1)+len(data2), :] = data2[:, 0:13]
    # y[len(data1):len(data1)+len(data2)] = data2[:, -1]
    # model = SVC()
    # model.fit(train_x, y)
    # p = model.predict(train_x)
    # print(model)
    # print(sklearn.metrics.classification_report(y, p, labels=[1, 0], target_names=["tumour", "pgb"], digits=4))
    # print(metrics.confusion_matrix(y, p))
    # test = train_x[0, :]
    # model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)
    # p = model.decision_function(test)
    # print(p)
    data = np.loadtxt("all-features.txt", delimiter=",")
    X = data[:, 0:13]
    y = data[:, -1]
    y = np.transpose(y)
    for i in range(1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model = SVC()
        model.fit(X_train, y_train)
        print(model)
        p = model.predict(X_test)
        print(sklearn.metrics.classification_report(y_test, p, labels=[1, 0], target_names=["tumour", "pgb"], digits=4))
        print(metrics.confusion_matrix(y_test, p))

    # model = GaussianNB()
    # cv1 = ShuffleSplit(n_splits=5, test_size=.3, random_state=0)
    # scores = cross_val_score(model, data, y, cv=cv1)
    # print(scores)
    # print(scores.mean())
"""
    list1 = [1,2,1,1,1,1,2,2]
    list1.remove(1)
    print(list1)
    print(list1[0])









