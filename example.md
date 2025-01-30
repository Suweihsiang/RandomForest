實作
==========================================================================================
## 1.資料集使用breast_cancer.csv，模型使用Decision_tree，不更改參數設定  
### (1)參數：criterion = gini,max_depth = 2147483647,min_sample_split = 2,min_sample_leaf = 1,ccp_alpha = 0,min_impurity_decrease = 0  
### (2)執行時間：422ms  
### (3)決策樹：  
  
**node number = 0**  
node parent number = -1  
node layer = 1  
node depth = 7  
Split threashold : x[20] <=16.795  
node criteria = 0.46753  
node sample size = 569  
node label count : [357,212]  
  
  
**node number = 1**  
node parent number = 0  
node layer = 2  
node depth = 6  
Split threashold : x[27] <=0.1358  
node criteria = 0.15898  
node sample size = 379  
node label count : [346,33]  
  
  
**node number = 3**  
node parent number = 1  
node layer = 3  
node depth = 5  
Split threashold : x[29] <=0.055125  
node criteria = 0.0295791  
node sample size = 333  
node label count : [328,5]  
  
  
**node number = 7**  
node parent number = 3  
node layer = 4  
node depth = 0  
node criteria = 0  
node sample size = 1  
node label count : [0,1]  
  
  
**node number = 8**  
node parent number = 3  
node layer = 4  
node depth = 4  
Split threashold : x[13] <=38.605  
node criteria = 0.0238061  
node sample size = 332  
node label count : [328,4]  
  
  
**node number = 17**  
node parent number = 8  
node layer = 5  
node depth = 3  
Split threashold : x[14] <=0.003294  
node criteria = 0.0124606  
node sample size = 319  
node label count : [317,2]  
  
  
**node number = 35**  
node parent number = 17  
node layer = 6  
node depth = 1  
Split threashold : x[27] <=0.100985  
node criteria = 0.244898  
node sample size = 7  
node label count : [6,1]  
  
  
**node number = 71**  
node parent number = 35  
node layer = 7  
node depth = 0  
node criteria = 0  
node sample size = 6  
node label count : [6,0]  
  
  
**node number = 72**  
node parent number = 35  
node layer = 7  
node depth = 0  
node criteria = 0  
node sample size = 1  
node label count : [0,1]  
  
  
**node number = 36**  
node parent number = 17  
node layer = 6  
node depth = 2  
Split threashold : x[21] <=33.27  
node criteria = 0.00638971  
node sample size = 312  
node label count : [311,1]  
  
  
**node number = 73**  
node parent number = 36  
node layer = 7  
node depth = 0  
node criteria = 0  
node sample size = 292  
node label count : [292,0]  
  
  
**node number = 74**  
node parent number = 36  
node layer = 7  
node depth = 1  
Split threashold : x[21] <=33.56  
node criteria = 0.095  
node sample size = 20  
node label count : [19,1]  
  
  
**node number = 149**  
node parent number = 74  
node layer = 8  
node depth = 0  
node criteria = 0  
node sample size = 1  
node label count : [0,1]  
  
  
**node number = 150**  
node parent number = 74  
node layer = 8  
node depth = 0  
node criteria = 0  
node sample size = 19  
node label count : [19,0]  
  
  
**node number = 18**  
node parent number = 8  
node layer = 5  
node depth = 2  
Split threashold : x[28] <=0.2069  
node criteria = 0.260355  
node sample size = 13  
node label count : [11,2]  
  
  
**node number = 37**  
node parent number = 18  
node layer = 6  
node depth = 0  
node criteria = 0  
node sample size = 1  
node label count : [0,1]  
  
  
**node number = 38**  
node parent number = 18  
node layer = 6  
node depth = 1  
Split threashold : x[27] <=0.11665  
node criteria = 0.152778  
node sample size = 12  
node label count : [11,1]  
  
  
**node number = 77**  
node parent number = 38  
node layer = 7  
node depth = 0  
node criteria = 0  
node sample size = 11  
node label count : [11,0]  
  
  
**node number = 78**  
node parent number = 38  
node layer = 7  
node depth = 0  
node criteria = 0  
node sample size = 1  
node label count : [0,1]  
  
  
**node number = 4**  
node parent number = 1  
node layer = 3  
node depth = 3  
Split threashold : x[21] <=25.67  
node criteria = 0.476371  
node sample size = 46  
node label count : [18,28]  
  
  
**node number = 9**  
node parent number = 4  
node layer = 4  
node depth = 2  
Split threashold : x[23] <=810.3  
node criteria = 0.33241  
node sample size = 19  
node label count : [15,4]  
  
  
**node number = 19**  
node parent number = 9  
node layer = 5  
node depth = 1  
Split threashold : x[24] <=0.17855  
node criteria = 0.124444  
node sample size = 15  
node label count : [14,1]  
  
  
**node number = 39**  
node parent number = 19  
node layer = 6  
node depth = 0  
node criteria = 0  
node sample size = 14  
node label count : [14,0]  
  
  
**node number = 40**  
node parent number = 19  
node layer = 6  
node depth = 0  
node criteria = 0  
node sample size = 1  
node label count : [0,1]  
  
  
**node number = 20**  
node parent number = 9  
node layer = 5  
node depth = 1  
Split threashold : x[24] <=0.13525  
node criteria = 0.375  
node sample size = 4  
node label count : [1,3]  
  
  
**node number = 41**  
node parent number = 20  
node layer = 6  
node depth = 0  
node criteria = 0  
node sample size = 1  
node label count : [1,0]  
  
  
**node number = 42**  
node parent number = 20  
node layer = 6  
node depth = 0  
node criteria = 0  
node sample size = 3  
node label count : [0,3]  
  
  
**node number = 10**  
node parent number = 4  
node layer = 4  
node depth = 2  
Split threashold : x[7] <=0.0541  
node criteria = 0.197531  
node sample size = 27  
node label count : [3,24]  
  
  
**node number = 21**  
node parent number = 10  
node layer = 5  
node depth = 1  
Split threashold : x[21] <=28.545  
node criteria = 0.5  
node sample size = 6  
node label count : [3,3]  
  
  
**node number = 43**  
node parent number = 21  
node layer = 6  
node depth = 0  
node criteria = 0  
node sample size = 3  
node label count : [3,0]  
  
  
**node number = 44**  
node parent number = 21  
node layer = 6  
node depth = 0  
node criteria = 0  
node sample size = 3  
node label count : [0,3]  
  
  
**node number = 22**  
node parent number = 10  
node layer = 5  
node depth = 0  
node criteria = 0  
node sample size = 21  
node label count : [0,21]  
  
  
**node number = 2**  
node parent number = 0  
node layer = 2  
node depth = 4  
Split threashold : x[21] <=19.91  
node criteria = 0.109086  
node sample size = 190  
node label count : [11,179]  
  
  
**node number = 5**  
node parent number = 2  
node layer = 3  
node depth = 1  
Split threashold : x[27] <=0.1454  
node criteria = 0.49827  
node sample size = 17  
node label count : [9,8]  
  
  
**node number = 11**  
node parent number = 5  
node layer = 4  
node depth = 0  
node criteria = 0  
node sample size = 9  
node label count : [9,0]  
  
  
**node number = 12**  
node parent number = 5  
node layer = 4  
node depth = 0  
node criteria = 0  
node sample size = 8  
node label count : [0,8]  
  
  
**node number = 6**  
node parent number = 2  
node layer = 3  
node depth = 3  
Split threashold : x[24] <=0.08798  
node criteria = 0.0228541  
node sample size = 173  
node label count : [2,171]  
  
  
**node number = 13**  
node parent number = 6  
node layer = 4  
node depth = 0  
node criteria = 0  
node sample size = 1  
node label count : [1,0]  
  
  
**node number = 14**  
node parent number = 6  
node layer = 4  
node depth = 2  
Split threashold : x[26] <=0.17975  
node criteria = 0.0115603  
node sample size = 172  
node label count : [1,171]  
  
  
**node number = 29**  
node parent number = 14  
node layer = 5  
node depth = 1  
Split threashold : x[28] <=0.2423  
node criteria = 0.375  
node sample size = 4  
node label count : [1,3]  
  
  
**node number = 59**  
node parent number = 29  
node layer = 6  
node depth = 0  
node criteria = 0  
node sample size = 1  
node label count : [1,0]  
  
  
**node number = 60**  
node parent number = 29  
node layer = 6  
node depth = 0  
node criteria = 0  
node sample size = 3  
node label count : [0,3]  
  
  
**node number = 30**  
node parent number = 14  
node layer = 5  
node depth = 0  
node criteria = 0  
node sample size = 168  
node label count : [0,168]  
  
### (4)使用sklearn之DecisionTreeClassifier之決策樹：  
可以與使用C++產生之決策樹之Split threashold、node sample size、node label count比對，但Split threashold可能因逐項比對特徵時因順序之不同而有異，但由於分割之criteria相同，故分割之結果(sample size、label count)亦相同  
![image](https://github.com/Suweihsiang/RandomForest/blob/main/image/breast_cancer1.png)  
  
## 2.資料集使用breast_cancer.csv，模型使用Decision_tree：  
### (1)參數設定如下：criterion = gini,max_depth = 2147483647,min_sample_split = 85,min_sample_leaf = 1,ccp_alpha = 0.006,min_impurity_decrease = 0.01  
### (2)執行時間：419ms  
### (3)決策樹：  
  
**node number = 0**  
node parent number = -1  
node layer = 1  
node depth = 2  
Split threashold : x[20] <=16.795  
node criteria = 0.46753  
node sample size = 569  
node label count : [357,212]  
  
  
**node number = 1**  
node parent number = 0  
node layer = 2  
node depth = 1  
Split threashold : x[27] <=0.1358  
node criteria = 0.15898  
node sample size = 379  
node label count : [346,33]  
  
  
**node number = 3**  
node parent number = 1  
node layer = 3  
node depth = 0  
node criteria = 0.0295791  
node sample size = 333  
node label count : [328,5]  
  
  
**node number = 4**  
node parent number = 1  
node layer = 3  
node depth = 0  
node criteria = 0.476371  
node sample size = 46  
node label count : [18,28]  
  
  
**node number = 2**  
node parent number = 0  
node layer = 2  
node depth = 1  
Split threashold : x[21] <=19.91  
node criteria = 0.109086  
node sample size = 190  
node label count : [11,179]  
  
  
**node number = 5**  
node parent number = 2  
node layer = 3  
node depth = 0  
node criteria = 0.49827  
node sample size = 17  
node label count : [9,8]  
  
  
**node number = 6**  
node parent number = 2  
node layer = 3  
node depth = 0  
node criteria = 0.0228541  
node sample size = 173  
node label count : [2,171]  
  
    
### (4)使用sklearn之DecisionTreeClassifier之決策樹：  
可以與使用C++產生之決策樹之Split threashold、node sample size、node label count比對，但Split threashold可能因逐項比對特徵時因順序之不同而有異，但由於分割之criteria相同，故分割之結果(sample size、label count)亦相同  
![image](https://github.com/Suweihsiang/RandomForest/blob/main/image/breast_cancer2.png)  
  
## 3.資料集使用breast_cancer.csv，模型使用RandomForest  
### (1)參數設定如下：：nEstimators = 100,criterion = gini,fit_samples = 400,max_depth = 2147483647,min_sample_split = 85,min_sample_leaf = 1,ccp_alpha = 0.006,min_impurity_decrease = 0.01  
### (2)執行時間：28498ms  
### (3)使用原資料集隨機抽170筆資料進行測試，模型分數為0.923077  