DecisionTree.h  DecisionTree.cpp  
使用C++實現DecisionTree  
===========================================================================  
  
Node  
===========================================================================  
Parameters
===========================================================================  
**data_index：vector< int >**  
節點上所有資料之索引值  
  
**number：int**  
節點編號  
  
**parent_number：int**  
父節點編號  
  
**feature_index：int**  
作為劃分標準之特徵編號  
  
**threshold：double**  
閾值  
  
**criteria：double**  
節點之不純度  
  
**samples：int**  
節點上之資料數量  
  
**values：vector< int >**  
各類別之數量  
  
**isLeaf：bool**  
是否為葉節點  
  
**node_layer：int**  
節點所在層數  
  
**node_depth：int**  
節點下的層數  
  
**parent：Node\***  
父節點  
  
**left：Node\***  
左子節點  
  
  **right：Node\***  
右子節點  
  
  
DecisionTree  
===========================================================================  
Parameters
===========================================================================  
**criterion：string**  
不純度計算方法，可為gini不純度或信息熵，預設為gini不純度  
  
**max_depth：int**  
決策樹的最大深度  
  
**min_sample_split：int**  
決策樹節點能夠進行分枝的最小資料數，預設為2  
  
**min_sample_leaf：int**  
決策樹節點之子節點的最小資料數，預設為1  
  
**ccp_alpha：double**  
Cost Complicity Pruning路徑可接受之最小alpha值其對應之不純度，並以此進行剪枝，預設為0  
  
**min_impurity_decrease：double**  
節點分枝後可接受之最小不純度減少幅度，小於該數值則不執行分枝，預設為0  
  
Attributes  
===========================================================================  
**Root：Node**  
決策樹之根  
  
**depth：int**  
決策樹深度，預設為1  
  
**classes：set\<int\>**  
資料之分類標籤  
  
Methods  
===========================================================================  
===========================================================================  
**Decision_Tree()**  
Decision_Tree建構式  
  
===========================================================================  
**Decision_Tree(string criterion, int max_depth, int min_sample_split, int min_sample_leaf, double ccp_alpha, double min_impurity_decrease)**  
**Parameters:**   
criterion：string  
   *不純度計算方式，可為gini或entropy，預設為gini*  
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
**~Decision_Tree()**  
Decision_Tree解構式  
  
===========================================================================  
**void set_params(unordered_map<string, double>params)**  
設定決策樹演算法之參數  
**parameters:**  
params：unordered_map<string,double>  
   *{參數名稱：參數設定值}，設定max_depth、min_sample_split、min_sample_leaf、ccp_alpha、min_impurity_decrease、criterion(1為gini)*  
  
===========================================================================  
**void get_params()**  
印出決策樹演算法之參數  
  
===========================================================================  
**void fit(vector\<vector\<double\>\> &x, vector\<int\> &y)**  
進行決策樹演算法之配適    
**parameters:**  
x：vector\<vector\<double\>\>  
   *訓練資料集*  
y：vector\<int\>  
   *目標變數*  
  
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
  
===========================================================================  
**vector\<pair\<double,double\>\> cost_complexity_pruning_path()**  
**return:**  
ccp_path：vector\<pair\<double,double\>\>  
   *各個ccp_alpha及其對應之不純度所組成之路徑*  
  
===========================================================================  
**void export_tree()**  
以文字形式打印出決策樹  
  
===========================================================================  
**int get_classes_size()**  
**return:**  
classes_size：int  
   *目標標籤之個數*  
  
===========================================================================  
**vector\<int\> label_count(vector\<int\> & data)**  
**parameters:**  
data：vector\<int\>  
   *資料集之目標標籤*  
**return:**  
counts：vector\<int\>  
   *各個標籤之資料數量*  
  
===========================================================================  
**double gini_impurity(vector\<int\> & data, vector\<int\> &label_counts)**  
**parameters:**  
data：vector\<int\>  
   *資料集之目標標籤*  
label_counts：vector\<int\>  
   *各個標籤之資料數量*  
**return:**  
gini：double  
   *gini不純度*  
  
===========================================================================  
**double calc_entropy(vector\<int\> & data, vector\<int\> &label_counts)**  
**parameters:**  
data：vector\<int\>  
   *資料集之目標標籤*  
label_counts：vector\<int\>  
   *各個標籤之資料數量*  
**return:**  
entropy：double  
   *信息熵*  
  
===========================================================================  
**void split(Node\* node, vector\<pair\<vector\<double\>,int\>\> &data)**  
在當前的節點上決定最佳分割所使用之特徵及閾值  
**parameters:**  
node：Node*  
   *當前的節點*  
data：vector\<pair\<vector\<double\>,int\>\>  
   *資料集的輸入樣本及其對應之目標標籤*  
  
===========================================================================  
**void sortX(vector\<pair\<vector\<double\>,int\>\> &data, int &col)**  
依照資料的某個特徵行由小到大排列  
**parameters:**  
data：vector\<pair\<vector\<double\>,int\>\>  
   *資料集的輸入樣本及其對應之目標標籤*  
col：int  
   *特徵行數*  
  
===========================================================================  
**double threshold(Node\* node, int &col, double &crit, vector\<pair\<vector\<double\>,int\>\> &data)**  
**parameters:**  
node：Node*  
   *當前的節點*  
col：int  
   *此特徵之行數*  
crit：double  
   *執行本特徵分割前最小的不純度*  
data：vector\<pair\<vector\<double\>,int\>\>  
   *資料集的輸入樣本及其對應之目標標籤*  
**return:**  
crit：double  
   *執行本特徵分割後最小的不純度*  
  
===========================================================================  
**void predict_node(Node\* node, vector\<vector\<double\>\> &x, vector\<int\> &y_pred)**  
若當前的節點為葉節點，則將預測結果儲存於y_pred。若否，則依當前節點之分割標準進行分割後，繼續在其子節點進行預測。  
**parameters:**  
node：Node*  
   *當前的節點*  
x：vector\<vector\<double\>\>  
   *預測資料集*  
y_pred：vector\<int\>  
   *執行預測後之結果儲存在此*  
  
===========================================================================  
**void copy_tree(Node\* tree, Node\* new_tree)**  
**parameters:**  
tree：Node*  
   *當前樹之根節點*  
new_tree：Node*  
   *複製樹之根節點*  
  
===========================================================================  
**void construct_treemap(Node\* node, map\<int, vector\<double\>\> &tree_map)**  
treemap = {節點編號,{父節點編號,是否為葉節點,節點不純度,節點樣本數量,節點下之深度}}，提供cost complicity pruning使用  
**parameters:**  
node：Node*  
   *當前的節點*  
tree_map：map\<int, vector\<double\>\>  
   *儲存生成之treemap*  
  
===========================================================================  
**map\<int, vector\<double\>\> cost_complexity_pruning(Node\* node, map\<int, vector\<double\>\> &best_tree_map, map\<int, vector\<double\>\> &tree_map, vector\<pair\<double,double\>\> &path,int iters)**  
**parameters:**  
node：Node*  
   *當前的節點*  
best_tree_map：map\<int, vector\<double\>\>  
   *目前執行cost complicity pruning中的最佳treemap*  
tree_map：map\<int, vector\<double\>\>  
   *目前產生之treemap*  
path：vector\<pair\<double,double\>\>  
   *用以儲存ccp_path*  
iters：int  
   *目前ccp_path的元素數量*  
**return:**  
best_tree_map：map\<int, vector\<double\>\>  
   *目前執行cost complicity pruning中的最佳treemap*  
  
===========================================================================  
**int findparent(int number, map\<int, vector\<double\>\> &tree_map,int target)**  
藉由treemap來尋找某節點截至目標截點為止之最高層父節點  
**parameters:**  
number：int  
   *當前節點編號*  
tree_map：map\<int, vector\<double\>\>  
   *目前產生之treemap*  
target：int  
   *目標節點編號*  
**return:**  
parent_number：int  
   *截至目標截點為止之最高層父節點編號*  
  
===========================================================================  
**void construct_pruning_tree(Node &tree,map\<int, vector\<double\>\> &tree_map)**  
依照treemap來修剪目前之決策樹
**parameters:**  
tree：Node  
   *目前之決策樹*  
tree_map：map\<int, vector\<double\>\>  
   *目前產生之treemap*  
    
===========================================================================  
**void traverse_node(Node\* node, map\<int,vector\<double\>\> &tree_map,bool &pruning)**  
找尋需要剪枝之節點並剪枝
以索引值之順序進行排列  
**parameters:**  
node：Node*  
   *目前之節點*  
tree_map：map\<int,vector\<double\>\>  
   *目前產生之treemap*  
pruning：bool  
   *目前之節點是否需要剪枝*  
  
===========================================================================  
**void export_node(Node\* node)**  
以DFS依序打印決策樹節點內容  
**parameters:**  
node：Node*  
   *目前之節點*  
    
===========================================================================  
**void print(Node\* node)**  
打印當前節點之內容  
**parameters:**  
node：Node*  
   *當前節點*  
  
===========================================================================  
**void adjust_parent_depth(Node\* node)**  
剪枝後調整被剪枝節點其父節點以上之節點深度  
**parameters:**  
node：Node*  
   *當前節點*  
  
===========================================================================  
**void pruning_min_impurity_decrease(Node\* node)**  
目前節點分割後若不純度減少幅度小於min_impurity_decrease，則不執行分割，當前節點為葉節點  
**parameters:**  
node：Node*  
   *當前節點*  