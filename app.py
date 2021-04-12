if __name__ == '__main__':
    import csv
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.callbacks import History
    import sklearn
    from sklearn.preprocessing import MinMaxScaler
    # print(csv.__version__)
    # print(np.__version__)
    # print(pd.__version__)
    # print(tf.__version__)
    # print(tf.keras.__version__)
    # print(sklearn.__version__)
    dataset_train = pd.read_csv('training.csv',header = None)  # 讀取訓練集
    training_set = dataset_train.iloc[0:1488, 0:1].values  # 取「Open」欄位值
    # Feature Scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []   #預測點的前 30 天的資料
    y_train = []   #預測點
    for i in range(30, 1488):  
        X_train.append(training_set_scaled[i-30:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train) 
    # print(X_train[0])
    # print(y_train[0])
    # print(X_train.shape)
    # print(y_train.shape)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # print(X_train.shape[0])   //1458
    # print(X_train.shape[1])   //30
    # print(X_train.shape[2])   //1
    # # LSTM
    # regressor = Sequential()
    # # Adding the first LSTM layer and some Dropout regularisation
    # regressor.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    # regressor.add(Dropout(0.2))

    # # Adding a second LSTM layer and some Dropout regularisation
    # regressor.add(LSTM(units = 64, return_sequences = True))
    # regressor.add(Dropout(0.2))

    # # Adding a third LSTM layer and some Dropout regularisation
    # regressor.add(LSTM(units = 32, return_sequences = True))
    # regressor.add(Dropout(0.2))

    # # Adding a fourth LSTM layer and some Dropout regularisation
    # regressor.add(LSTM(units = 16))
    # regressor.add(Dropout(0.2))

    # regressor.add(Dense(units = 1))
    # # Compiling
    # regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error')

    # # 進行訓練
    # his = History()
    # regressor.fit(X_train, y_train, epochs = 50, batch_size = 32,callbacks=[his])
    # print(his.history)
    # regressor.save('app_model')
    # del regressor
    model = load_model('app_model')
    dataset1 = pd.read_csv('training.csv',header = None)  # 讀取訓練集
    testing_set1 = dataset1.iloc[1458:1489, 0:1].values
    dataset2 = pd.read_csv('testing.csv',header = None)
    testing_set2 = dataset2.iloc[:, 0:1].values  
    data = []
    for i in range(len(testing_set1)):
        data.append(testing_set1[i][0])
    for j in range(len(testing_set2)):
        data.append(testing_set2[j][0])
    # with open('test_data.csv', 'w', newline='',encoding="utf-8") as csvfile1:
    #     # 建立 CSV 檔寫入器
    #     writer = csv.writer(csvfile1)
    #     for k in range(len(data)):
    #         s = [data[k]]
    #         writer.writerow(s)
    dataset_train = pd.read_csv('training.csv',header = None)  # 讀取訓練集
    dataset_test = pd.read_csv('test_data.csv')
    real_stock_price = dataset_test.iloc[:, 0:1].values
    # for i in range(30, 50): 
    #     X_test.append(real_stock_price[i-30:i, 0])
    # X_test = np.array(X_test)
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 
    # predicted_stock_price = model.predict(X_test)
    # predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    # print(predicted_stock_price)
    flag = 0
    k = 0
    buy_price = 0.0
    buy_list = []
    for i in range(30, 49): #每次只讀前30天去預測下一天的動作
        X_test = []
        X_test.append(real_stock_price[i-30:i, 0])
        today = X_test[0][29]
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        new_predicted = predicted_stock_price[0][0]*100000 - (int(predicted_stock_price[0][0]*100)*1000)
        new_predicted = round(new_predicted,1)
        if(k == 1):
            buy_price = today
            k = 0
        if(flag == 0):
            if(new_predicted - today):
                buy_list.append(1)
                k = 1
                flag = 1
            else:
                buy_list.append(0)
        else:
            if(new_predicted - buy_price > 0):
                buy_list.append(0)
            else:
                buy_list.append(-1)
                flag = 0
    print(buy_list)
    with open('output.csv', 'w', newline='',encoding="utf-8") as csvfile1:   #Y_test
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile1)
            for i in range(len(buy_list)):
                writer.writerow([buy_list[i]])