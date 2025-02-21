Label.h  Label.cpp  
使用C++實現sklearn之LabelEncoder  
===========================================================================  
Attributes
===========================================================================  
**data：vector\<pair\<string,vector\<string\>\>\>**  
各項特徵與其標籤後之資料  
  
**classes：map\<int,set\<string\>\>**  
特徵之行數與其所有種類  
  
**encoder：map\<int,map\<string,string\>\>**  
特徵之行數與其所有種類對應之標籤{特徵行數,{名稱,標籤}}  
      
Methods  
===========================================================================  
===========================================================================  
**Label()**  
Label建構式  
  
===========================================================================  
**~Label()**  
Label解構式  
  
===========================================================================  
**void fit(vector\<vector\<string\>\> &raw_data, vector\<int\> cols)**  
將原始資料之選定特徵行進行標籤  
**parameters:**  
raw_data：vector\<vector\<string\>\>  
   *原始資料*  
cols：vector\<int\>  
   *欲進行標籤之特徵行數*  
  
===========================================================================  
**void transform(vector\<vector\<string\>\> &raw_data, vector\<int\> cols)**  
將原始資料之選定特徵行資料轉為標籤  
raw_data：vector\<vector\<string\>\>  
   *原始資料*  
cols：vector\<int\>  
   *欲轉換為標籤之特徵行數*  
  
===========================================================================  
**void inverse_transform(vector\<vector\<string\>\> & raw_data, vector\<int\> cols)**  
將原始資料之選定特徵行資料自標籤轉為原始資料  
raw_data：vector\<vector\<string\>\>  
   *以轉換為標籤之資料*  
cols：vector\<int\>  
   *欲自標籤轉為原始資料之特徵行數*  
  
===========================================================================  
**vector\<pair\<string,vector\<string\>\>\> get_data()**  
**return:**  
data：vector\<pair\<string,vector\<string\>\>\>  
   *各項特徵與其標籤後之資料*  
  
===========================================================================  
**map\<int,set\<string\>\> get_classes()**  
**return:**  
classes：map\<int,set\<string\>\>  
   *特徵之行數與其所有種類*  
  
===========================================================================  
**map\<int,map\<string,string\>\> get_encoder()**  
**return:**  
encoder：map\<int,map\<string,string\>\>  
   *特徵之行數與其所有種類對應之標籤{特徵行數,{名稱,標籤}}*
  
===========================================================================  
**string get_key(string &value,int &col)**  
取得某特徵之標籤其對應之原始資料  
**parameters:**  
value：string  
   *標籤*  
col：int  
   *特徵行數*  
**return:**  
key：string  
   *標籤對應之原始資料*  
  
