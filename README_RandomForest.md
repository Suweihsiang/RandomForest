RandomForest.h  RandomForest.cpp  
使用C++實現RandomForest  
===========================================================================  
  
Parameters
===========================================================================  
**nEstimators：int**  
由幾棵決策樹組成隨機森林，預設為100  
  
**criterion：string**  
不純度計算方法，可為gini不純度或信息熵，預設為gini不純度  

**fit_samples：int**  
隨機挑選幾個樣本進行訓練  
    
**max_depth：int**  
每棵決策樹的最大深度  
  
**min_sample_split：int**  
每棵決策樹節點能夠進行分枝的最小資料數，預設為2  
  
**min_sample_leaf：int**  
每棵決策樹節點之子節點的最小資料數，預設為1  
  
**ccp_alpha：double**  
Cost Complicity Pruning路徑可接受之最小alpha值其對應之不純度，並以此進行剪枝，預設為0  
  
**min_impurity_decrease：double**  
節點分枝後可接受之最小不純度減少幅度，小於該數值則不執行分枝，預設為0  
  
Attributes  
===========================================================================  
**trees：vector\<Decision_Tree\>**  
儲存每棵決策樹  

    
Methods  
===========================================================================  
===========================================================================  
**RandomForest()**  
RandomForest建構式  
  
===========================================================================  
**RandomForest(int nEstimators, string criterion, int fit_samples, int max_depth, int min_sample_split, int min_sample_leaf, double ccp_alpha, double min_impurity_decrease)**  
**Parameters:**  
nEstimators：int  
   *由幾棵決策樹組成隨機森林，預設為100*  
criterion：string  
   *不純度計算方式，可為gini或entropy，預設為gini*  
fit_samples：int  
   *隨機挑選幾個樣本進行訓練*  
max_depth：int  
   *決策樹最大深度*  
min_sample_split：int  
   *決策樹節點能夠進行分枝的最小資料數，預設為2*  
min_sample_leaf：int  
   *決策樹節點之子節點的最小資料數，預設為1*  
ccp_alpha：double  
   *Cost Complicity Pruning路徑可接受之最小alpha值，預設為0*  
min_impurity_decrease：double  
   *節點分枝後可接受之最小不純度減少幅度，小於該數值則不執行分枝，預設為0*  
  
  
===========================================================================  
**~RandomForest()**  
RandomForest解構式  
    
===========================================================================  
**void fit(vector\<pair\<vector\<double\>,int\>\>& datas)**  
進行隨機森林演算法之配適    
**parameters:**  
datas：vector\<pair\<vector\<double\>,int\>\>  
   *訓練資料集*  
  
===========================================================================  
**vector\<int\> predict(vector\<vector\<double\>\> x)**  
**parameters:**  
x：vector\<vector\<double\>\>  
   *輸入之預測資料集*  
**return:**  
y_pred：vector\<int\>  
   *預測結果*  
  
===========================================================================  
**double score(vector\<vector\<double\>\> &x, vector\<int\> &y)**  
**parameters:**  
x：vector\<vector\<double\>\>  
   *輸入之測試資料集*  
y：vector\<int\>  
   *真正的目標變數*  
**return:**  
score：double  
   *模型預測之分數*  