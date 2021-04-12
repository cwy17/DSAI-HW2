import csv
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
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
X_test = []
for i in range(30, 50): 
    X_test.append(real_stock_price[i-30:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshape 成 3-dimension
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)
# for i in range(30, 50): #每次只讀30天去預測下一天的動作
#     X_test = real_stock_price[i-30:i, 0]
#     X_test = np.array(X_test)
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#     predicted_stock_price = model.predict(X_test)
#     print(predicted_stock_price)