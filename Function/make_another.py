import scipy.io
import numpy as np
from sklearn import preprocessing

def window_slice(data, time_steps):#dataをtimestepsの長さずつに一つずつデータをずらして小分けする
    data_ = data.copy()
    data_ = np.transpose(data_, (1, 0, 2)).reshape(-1, 310) # (n, 310)
    
    # ここで全体に対して標準化+0.5の処理を行う
    # data_をreshapeして一列にして、これに処理を行い、(n,　310)に戻す
    data_st = data_.reshape(data_.shape[0] * data_.shape[1], ) #(n*310, )
    data_st = preprocessing.scale(data_st)
    data_st += 0.5
    data_ = data_st.reshape(data_.shape[0], data_.shape[1]) #(n, 310)
    
    xs = []
    for i in range(int(data_.shape[0] / time_steps)):
        k = i*time_steps
        xs.append(data_[k: k + time_steps, :])
    
    xs = np.concatenate(xs).reshape((len(xs), -1, 310))
    return xs

# def setData(de_, label_data, time_steps=9):
#     X = []
#     y = []   

#     for data, label_ in zip(de_, list(label_data)):
#         X.append(window_slice(data, time_steps))
#         y_np = np.array([label_] * len(X[-1]))
#         y.append(y_np)
    
#     #y = np.array(y)
#     return X, y #Xはlistで映画15個のデータを入れてある
#     #return np.concatenate(X), y

def window_slice_m(data, time_steps, num):#dataをtimestepsの長さずつに一つずつデータをずらして小分けする
    data_ = data.copy()
    data_ = np.transpose(data_, (1, 0, 2)).reshape(-1, 310) # (n, 310)
    
    # ここで全体に対して標準化+0.5の処理を行う
    # data_をreshapeして一列にして、これに処理を行い、(n,　310)に戻す
    data_st = data_.reshape(data_.shape[0] * data_.shape[1], ) #(n*310, )
    data_st = preprocessing.scale(data_st)
    data_st += 0.5
    data_ = data_st.reshape(data_.shape[0], data_.shape[1]) #(n, 310)
    
    xs = []
    for i in range(int((data_.shape[0]-9)/(time_steps-num) + 1)):
        if i == 0:
            k = 0
        else:
            k = k + (time_steps-num)
        xs.append(data_[k: k + time_steps, :])
    
    xs = np.concatenate(xs).reshape((len(xs), -1, 310))
    return xs

def setData(de_, label_data, time_steps=9):
    X = []
    y = []   

    for data, label_ in zip(de_, list(label_data)):
        X.append(window_slice_m(data, time_steps, 0))
        y_np = np.array([label_] * len(X[-1]))
        y.append(y_np)
    
    #y = np.array(y)
    return X, y #Xはlistで映画15個のデータを入れてある
    #return np.concatenate(X), y


def mkdata2d():
    data_1_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/1_20131030.mat")
    data_2_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/2_20140413.mat")
    data_3_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/3_20140611.mat")
    data_4_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/4_20140702.mat")
    data_5_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/5_20140418.mat")
    data_6_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/6_20131016.mat")
    data_7_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/7_20131030.mat")
    data_8_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/8_20140514.mat")
    data_9_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/9_20140627.mat")
    data_10_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/10_20131204.mat")
    data_11_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/11_20140625.mat")
    data_12_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/12_20131201.mat")
    data_13_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/13_20140603.mat")
    data_14_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/14_20140615.mat")
    data_15_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/15_20131016.mat")

    label_data = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/label.mat")
    label = label_data['label']
    label = label.reshape(15,)
    label += 1

    de_1_1 = []
    de_2_1 = []
    de_3_1 = []
    de_4_1 = []
    de_5_1 = []
    de_6_1 = []
    de_7_1 = []
    de_8_1 = []
    de_9_1 = []
    de_10_1 = []
    de_11_1 = []
    de_12_1 = []
    de_13_1 = []
    de_14_1 = []
    de_15_1 = []
    for i in range(1, 16):
        de_1_1.append(data_1_1['de_LDS' + str(i)])
        de_2_1.append(data_2_1['de_LDS' + str(i)])
        de_3_1.append(data_3_1['de_LDS' + str(i)])
        de_4_1.append(data_4_1['de_LDS' + str(i)])
        de_5_1.append(data_5_1['de_LDS' + str(i)])
        de_6_1.append(data_6_1['de_LDS' + str(i)])
        de_7_1.append(data_7_1['de_LDS' + str(i)])
        de_8_1.append(data_8_1['de_LDS' + str(i)])
        de_9_1.append(data_9_1['de_LDS' + str(i)])
        de_10_1.append(data_10_1['de_LDS' + str(i)])
        de_11_1.append(data_11_1['de_LDS' + str(i)])
        de_12_1.append(data_12_1['de_LDS' + str(i)])
        de_13_1.append(data_13_1['de_LDS' + str(i)])
        de_14_1.append(data_14_1['de_LDS' + str(i)])
        de_15_1.append(data_15_1['de_LDS' + str(i)])

    x1, y1 = setData(de_1_1, label)
    x2, y2 = setData(de_2_1, label)
    x3, y3 = setData(de_3_1, label)
    x4, y4 = setData(de_4_1, label)
    x5, y5 = setData(de_5_1, label)
    x6, y6 = setData(de_6_1, label)
    x7, y7 = setData(de_7_1, label)
    x8, y8 = setData(de_8_1, label)
    x9, y9 = setData(de_9_1, label)
    x10, y10 = setData(de_10_1, label)
    x11, y11 = setData(de_11_1, label)
    x12, y12 = setData(de_12_1, label)
    x13, y13 = setData(de_13_1, label)
    x14, y14 = setData(de_14_1, label)
    x15, y15 = setData(de_15_1, label)

    EEG_data = []
    EEG_data.append(x1)
    EEG_data.append(x2)
    EEG_data.append(x3)
    EEG_data.append(x4)
    EEG_data.append(x5)
    EEG_data.append(x6)
    EEG_data.append(x7)
    EEG_data.append(x8)
    EEG_data.append(x9)
    EEG_data.append(x10)
    EEG_data.append(x11)
    EEG_data.append(x12)
    EEG_data.append(x13)
    EEG_data.append(x14)
    EEG_data.append(x15)
    EEG = EEG_data

    label_all = []
    label_all.append(y1)
    label_all.append(y2)
    label_all.append(y3)
    label_all.append(y4)
    label_all.append(y5)
    label_all.append(y6)
    label_all.append(y7)
    label_all.append(y8)
    label_all.append(y9)
    label_all.append(y10)
    label_all.append(y11)
    label_all.append(y12)
    label_all.append(y13)
    label_all.append(y14)
    label_all.append(y15)
    label = label_all

    return EEG, label_all


def mkdata3d():
    data_1_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/1_20131107.mat")
    data_2_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/2_20140419.mat")
    data_3_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/3_20140629.mat")
    data_4_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/4_20140705.mat")
    data_5_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/5_20140506.mat")
    data_6_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/6_20131113.mat")
    data_7_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/7_20131106.mat")
    data_8_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/8_20140521.mat")
    data_9_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/9_20140704.mat")
    data_10_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/10_20131211.mat")
    data_11_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/11_20140630.mat")
    data_12_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/12_20131207.mat")
    data_13_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/13_20140610.mat")
    data_14_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/14_20140627.mat")
    data_15_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/15_20131105.mat")

    label_data = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/label.mat")
    label = label_data['label']
    label = label.reshape(15,)
    label += 1

    de_1_1 = []
    de_2_1 = []
    de_3_1 = []
    de_4_1 = []
    de_5_1 = []
    de_6_1 = []
    de_7_1 = []
    de_8_1 = []
    de_9_1 = []
    de_10_1 = []
    de_11_1 = []
    de_12_1 = []
    de_13_1 = []
    de_14_1 = []
    de_15_1 = []
    for i in range(1, 16):
        de_1_1.append(data_1_1['de_LDS' + str(i)])
        de_2_1.append(data_2_1['de_LDS' + str(i)])
        de_3_1.append(data_3_1['de_LDS' + str(i)])
        de_4_1.append(data_4_1['de_LDS' + str(i)])
        de_5_1.append(data_5_1['de_LDS' + str(i)])
        de_6_1.append(data_6_1['de_LDS' + str(i)])
        de_7_1.append(data_7_1['de_LDS' + str(i)])
        de_8_1.append(data_8_1['de_LDS' + str(i)])
        de_9_1.append(data_9_1['de_LDS' + str(i)])
        de_10_1.append(data_10_1['de_LDS' + str(i)])
        de_11_1.append(data_11_1['de_LDS' + str(i)])
        de_12_1.append(data_12_1['de_LDS' + str(i)])
        de_13_1.append(data_13_1['de_LDS' + str(i)])
        de_14_1.append(data_14_1['de_LDS' + str(i)])
        de_15_1.append(data_15_1['de_LDS' + str(i)])

    x1, y1 = setData(de_1_1, label)
    x2, y2 = setData(de_2_1, label)
    x3, y3 = setData(de_3_1, label)
    x4, y4 = setData(de_4_1, label)
    x5, y5 = setData(de_5_1, label)
    x6, y6 = setData(de_6_1, label)
    x7, y7 = setData(de_7_1, label)
    x8, y8 = setData(de_8_1, label)
    x9, y9 = setData(de_9_1, label)
    x10, y10 = setData(de_10_1, label)
    x11, y11 = setData(de_11_1, label)
    x12, y12 = setData(de_12_1, label)
    x13, y13 = setData(de_13_1, label)
    x14, y14 = setData(de_14_1, label)
    x15, y15 = setData(de_15_1, label)

    EEG_data = []
    EEG_data.append(x1)
    EEG_data.append(x2)
    EEG_data.append(x3)
    EEG_data.append(x4)
    EEG_data.append(x5)
    EEG_data.append(x6)
    EEG_data.append(x7)
    EEG_data.append(x8)
    EEG_data.append(x9)
    EEG_data.append(x10)
    EEG_data.append(x11)
    EEG_data.append(x12)
    EEG_data.append(x13)
    EEG_data.append(x14)
    EEG_data.append(x15)
    EEG = EEG_data

    label_all = []
    label_all.append(y1)
    label_all.append(y2)
    label_all.append(y3)
    label_all.append(y4)
    label_all.append(y5)
    label_all.append(y6)
    label_all.append(y7)
    label_all.append(y8)
    label_all.append(y9)
    label_all.append(y10)
    label_all.append(y11)
    label_all.append(y12)
    label_all.append(y13)
    label_all.append(y14)
    label_all.append(y15)
    label = label_all

    return EEG, label_all


def mkdata_lab():
    data_1_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/1_20131027.mat")
    data_2_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/2_20140404.mat")
    data_3_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/3_20140603.mat")
    data_4_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/4_20140621.mat")
    data_5_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/5_20140411.mat")
    data_6_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/6_20130712.mat")
    data_7_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/7_20131027.mat")
    data_8_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/8_20140511.mat")
    data_9_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/9_20140620.mat")
    data_10_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/10_20131130.mat")
    data_11_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/11_20140618.mat")
    data_12_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/12_20131127.mat")
    data_13_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/13_20140527.mat")
    data_14_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/14_20140601.mat")
    data_15_1 = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/15_20130709.mat")

    label_data = scipy.io.loadmat(r"../SEED dataset/SEED/SEED/ExtractedFeatures/label.mat")
    label = label_data['label']
    label = label.reshape(15,)
    label += 1

    de_1_1 = []
    de_2_1 = []
    de_3_1 = []
    de_4_1 = []
    de_5_1 = []
    de_6_1 = []
    de_7_1 = []
    de_8_1 = []
    de_9_1 = []
    de_10_1 = []
    de_11_1 = []
    de_12_1 = []
    de_13_1 = []
    de_14_1 = []
    de_15_1 = []
    for i in range(1, 16):
        de_1_1.append(data_1_1['de_LDS' + str(i)])
        de_2_1.append(data_2_1['de_LDS' + str(i)])
        de_3_1.append(data_3_1['de_LDS' + str(i)])
        de_4_1.append(data_4_1['de_LDS' + str(i)])
        de_5_1.append(data_5_1['de_LDS' + str(i)])
        de_6_1.append(data_6_1['de_LDS' + str(i)])
        de_7_1.append(data_7_1['de_LDS' + str(i)])
        de_8_1.append(data_8_1['de_LDS' + str(i)])
        de_9_1.append(data_9_1['de_LDS' + str(i)])
        de_10_1.append(data_10_1['de_LDS' + str(i)])
        de_11_1.append(data_11_1['de_LDS' + str(i)])
        de_12_1.append(data_12_1['de_LDS' + str(i)])
        de_13_1.append(data_13_1['de_LDS' + str(i)])
        de_14_1.append(data_14_1['de_LDS' + str(i)])
        de_15_1.append(data_15_1['de_LDS' + str(i)])

    x1, y1 = setData(de_1_1, label)
    x2, y2 = setData(de_2_1, label)
    x3, y3 = setData(de_3_1, label)
    x4, y4 = setData(de_4_1, label)
    x5, y5 = setData(de_5_1, label)
    x6, y6 = setData(de_6_1, label)
    x7, y7 = setData(de_7_1, label)
    x8, y8 = setData(de_8_1, label)
    x9, y9 = setData(de_9_1, label)
    x10, y10 = setData(de_10_1, label)
    x11, y11 = setData(de_11_1, label)
    x12, y12 = setData(de_12_1, label)
    x13, y13 = setData(de_13_1, label)
    x14, y14 = setData(de_14_1, label)
    x15, y15 = setData(de_15_1, label)

    EEG_data = []
    EEG_data.append(x1)
    EEG_data.append(x2)
    EEG_data.append(x3)
    EEG_data.append(x4)
    EEG_data.append(x5)
    EEG_data.append(x6)
    EEG_data.append(x7)
    EEG_data.append(x8)
    EEG_data.append(x9)
    EEG_data.append(x10)
    EEG_data.append(x11)
    EEG_data.append(x12)
    EEG_data.append(x13)
    EEG_data.append(x14)
    EEG_data.append(x15)
    EEG = EEG_data

    label_all = []
    label_all.append(y1)
    label_all.append(y2)
    label_all.append(y3)
    label_all.append(y4)
    label_all.append(y5)
    label_all.append(y6)
    label_all.append(y7)
    label_all.append(y8)
    label_all.append(y9)
    label_all.append(y10)
    label_all.append(y11)
    label_all.append(y12)
    label_all.append(y13)
    label_all.append(y14)
    label_all.append(y15)
    label = label_all

    return EEG, label_all



















