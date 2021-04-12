#開啟 CSV 檔案
    # with open('training.csv', newline='',encoding="utf-8") as csvfile:
    #     #讀取 CSV 檔案內容
    #     rows = csv.reader(csvfile)
    #     rows2 = list(csv.reader(csvfile))
        # with open('X_train.csv', 'w', newline='',encoding="utf-8") as csvfile1:   #X_train
        #     # 建立 CSV 檔寫入器
        #     writer = csv.writer(csvfile1)
        #     time = 0
        #     for i in range(1100):
        #         A = []
        #         for j in range(len(rows2)):
        #             if(j >= i and j < i + 30):
        #                 s = rows2[j]
        #                 s[0] = float(s[0])
        #                 s[1] = float(s[1])
        #                 s[2] = float(s[2])
        #                 s[3] = float(s[3])
        #                 A.append(s)
        #         writer.writerow(A)

        # with open('X_test.csv', 'w', newline='',encoding="utf-8") as csvfile1:   #X_test
        #     # 建立 CSV 檔寫入器
        #     writer = csv.writer(csvfile1)
        #     time = 0
        #     for i in range(1101,1441):
        #         A = []
        #         for j in range(len(rows2)):
        #             if(j >= i and j < i + 30):
        #                 s = rows2[j]
        #                 s[0] = float(s[0])
        #                 s[1] = float(s[1])
        #                 s[2] = float(s[2])
        #                 s[3] = float(s[3])
        #                 A.append(s)
        #         writer.writerow(A)

        # with open('Y_train.csv', 'w', newline='',encoding="utf-8") as csvfile1:   #Y_train
        #     # 建立 CSV 檔寫入器
        #     writer = csv.writer(csvfile1)
        #     k = 0 
        #     time = 0
        #     flag = 0
        #     for row in rows:
        #         if(k == 30):
        #             flag = 1
        #         if (flag == 1):
        #             time = time + 1
        #             if(time <= 1100):
        #                 writer.writerow(row[0:1])
        #         k = k + 1

        # with open('Y_test.csv', 'w', newline='',encoding="utf-8") as csvfile1:   #Y_test
        #     # 建立 CSV 檔寫入器
        #     writer = csv.writer(csvfile1)
            
        #     for j in range(len(rows2)):
        #         A=[]
        #         if(j>1100 and j<=1440):
        #             s = rows2[j]
        #             s[0] = float(s[0])
        #             A.append(s[0])
        #             writer.writerow(A)