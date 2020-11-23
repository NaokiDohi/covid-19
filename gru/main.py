import os
import datetime as dt#デフォはUTCの国際基準時間
import pytz#日本時間に変換
import time
import sys

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

import statsmodels.api as sm # version 0.8.0以上
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_error,r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout,GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras import regularizers
from keras.models import load_model
from keras import backend as K
from keras.utils import plot_model
from IPython.display import Image

from preprocess import split, normalized, sequence_creator
from forecast import past_predict, test_predict, long_predict, eval_func

def main():
    args = sys.argv
    ARG_NUM = 5
    if(len(sys.argv) < ARG_NUM):
        print("Error")
        sys.exit(0)

    df = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    df = df[df['Country/Region']=='Japan']
    df = df.iloc[:,4:].copy()
    data_at_japan = df.iloc[0,:]
    data_at_japan.index = pd.to_datetime(data_at_japan.index)
    #print(data_at_japan)
    plt.figure(figsize=(10,5))
    plt.plot(data_at_japan)
    plt.title('COVID-19 confilmed at Japan', y = -0.2)
    plt.xlabel("Date")
    plt.ylabel("Person infected (people)")
    plt.grid(True)
    #plt.show()
    #ファイル保存
    fname_1 ='original.png'
    plt.savefig(fname_1)
    plt.close()

    data_at_japan_diff = data_at_japan - data_at_japan.shift(1) # 階差系列データの作成
    data_at_japan_diff = data_at_japan_diff.dropna()
    data_at_japan_diff = data_at_japan_diff['2020-01-23':'2020-10-28']#10-28
    #print(data_at_japan_diff)
    plt.figure(figsize=(10,5))
    plt.plot(data_at_japan_diff)
    plt.title('COVID-19 confilmed at Japan', y=-0.2)
    plt.xlabel("Date")
    plt.ylabel("Person infected (people)")
    plt.grid(True)
    #plt.show()
    #ファイル保存
    fname_2 ='diff.png'
    plt.savefig(fname_2)
    plt.close()

    res = sm.tsa.seasonal_decompose(data_at_japan_diff)#データを分解
    original = data_at_japan_diff # オリジナルデータ
    trend_original = res.trend # トレンドデータ
    seasonal_original = res.seasonal # 季節性データ
    residual = res.resid # 残差データ
    plt.figure(figsize=(10, 20)) # グラフ描画枠作成、サイズ指定
    plt.subplot(411) # グラフ4行1列の1番目の位置（一番上）
    plt.plot(original)
    plt.title('COVID-19 confilmed(Original) at Japan', y=-0.17)
    plt.xlabel("Date")
    plt.ylabel("Person infected (people)")
    plt.grid(True)
    # trend データのプロット
    plt.subplot(412) # グラフ4行1列の2番目の位置
    plt.plot(trend_original)
    plt.title('COVID-19 confilmed(Trend) at Japan', y=-0.17)
    plt.xlabel("Date")
    plt.ylabel("Person infected (people)")
    plt.grid(True)
    # seasonalデータ のプロット
    plt.subplot(413) # グラフ4行1列の3番目の位置
    plt.plot(seasonal_original)
    plt.title('COVID-19 confilmed(Seasonality) at Japan', y=-0.17)
    plt.xlabel("Date")
    plt.ylabel("Person infected (people)")
    plt.grid(True)
    # residual データのプロット
    plt.subplot(414) # グラフ4行1列の4番目の位置（一番下）
    plt.plot(residual)
    plt.title('COVID-19 confilmed(Residuals) at Japan', y=-0.17)
    plt.xlabel("Date")
    plt.ylabel("Person infected (people)")
    plt.grid(True)
    plt.tight_layout() # グラフの間隔を自動調整
    fname_3 ='decompose.png'
    plt.savefig(fname_3)


    y = data_at_japan_diff.values.astype(float)
    test_size = 7# test_size

    train_original_data, test_original_data = split(y)
    train_normalized = normalized(train_original_data)

    window = 7# 学習時のウィンドウサイズ
    study_data, correct_data  = sequence_creator(train_normalized, window)


    n_in_out = 1
    n_hidden = args[1]
    drop_out = args[2]
    tf.random.set_seed(0)

    # parameters = {
    #               'n_hidden': [16, 32, 64, 128, 256, 512, 1024]
    #               'dropout': [0, 0.2, 0.4, 0.5, 0.6],
    # }

    # model = KerasClassifier(build_fn=gru,
    #                         verbose=0)
    # gridsearch = GridSearchCV(estimator=model, param_grid=parameters)
    # gridsearch.fit(study_data, correct_data)
    # print('Best params are: {}'.format(gridsearch.best_params_))

    gru = gru(n_in_out, n_hidden, drop_out)
    print(gru.summary())

    filename = 'gru_'+str(n_hidden)+'_'+str(drop_out)+'.png'
    plot_model(gru, show_shapes=True, show_layer_names=True, to_file=filename)
    Image(retina=False, filename=filename)

    epochs = 1
    start_time = time.time()
    history = gru.fit(study_data, correct_data, batch_size=1, epochs=epochs, validation_split=0.1, verbose=1, callbacks=[])#lr_decay,
    print("学習時間:",time.time() - start_time)

    # === 学習推移の可視化 ===
    # mse
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(np.arange(len(train_loss)), train_loss, label="train_loss")
    plt.plot(np.arange(len(val_loss)), val_loss, label="val_loss")
    plt.title('Training and Validation loss')
    plt.ylim((0, 0.04))#add
    plt.legend()
    # plt.show()

    # === 学習推移の可視化 ===
    # mae
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    plt.plot(np.arange(len(train_mae)), train_mae, label="train_mae")
    plt.plot(np.arange(len(val_mae)), val_mae, label="val_mae")
    plt.title('Training and Validation mae')
    plt.ylim((0, 0.2))#add
    plt.legend()
    #plt.show()

    train_inverse = past_predict(study_data)

    upcoming_future=7
    predictions_infected_pepole = test_predict(upcoming_future)

    x_all =np.arange('2020-01-23','2020-10-29', dtype='datetime64[D]').astype('datetime64[D]')
    x_past_predict = np.arange('2020-01-30','2020-10-22', dtype='datetime64[D]').astype('datetime64[D]')#23-26
    x_train = np.arange('2020-01-23','2020-10-22', dtype='datetime64[D]').astype('datetime64[D]')
    x_test = np.arange('2020-10-22', '2020-10-29', dtype='datetime64[D]').astype('datetime64[D]')

    sns.set()
    COVID = plt.figure(figsize=(20,8))
    plt.title("COVID-19 in Japan", y=-0.15)
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Nunber of Person infected with corona virus (people)")
    plt.plot(x_all,data_at_japan_diff,'g',lw=3,label='daily_at_japan')
    # plt.plot(x_train,train_original_data,label='train_data')
    # plt.plot(x_test,test_original_data,label='test_data')
    plt.plot(x_past_predict,train_inverse,color='b', ls='-',lw=3,alpha=0.7, label='past_predict')#+8かも
    plt.plot(x_test, predictions_infected_pepole, 'r',lw=3,alpha=0.7,label='upcoming_future')
    plt.legend(loc='upper left')
    #plt.show()

    sns.set()
    COVID = plt.figure(figsize=(20,8))
    plt.title("COVID-19 in Japan", y=-0.15)
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Nunber of Person infected with corona virus　(people)")
    plt.plot(x_test,test_original_data,color='b', ls='-',lw=3,alpha=0.7, label='past_predict')
    plt.plot(x_test, predictions_infected_pepole, 'r',lw=3,alpha=0.7,label='upcoming_future')
    #plt.show()

    train_mae, train_mse, train_rmse, train_r2, test_mae, test_mse, test_rmse, test_r2 = eval_func(train_inverse, test_original_data, predictions_infected_pepole)

if __name__ == '__main__':
    main()
