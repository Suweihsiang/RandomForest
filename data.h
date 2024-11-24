#ifndef DATA_H
#define DATA_H
#include<iostream>
#include<eigen3/Eigen/Dense>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<algorithm>
#include<unordered_map>
#include<set>

using std::unordered_map;
using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using std::ios;
using std::istringstream;
using std::set;
using std::pair;
using std::cout;
using std::endl;
using std::to_string;
using std::transform;
using std::back_inserter;


using Eigen::Matrix;
using Eigen::MatrixX;
using Eigen::Dynamic;
using Eigen::RowVectorX;
using Eigen::VectorX;
using Eigen::Map;

template<typename T>
class Data {
public:
	Data();
	Data(unordered_map<string, vector<string>>m, bool IndexFirst, bool sortIndex = false);
	Data(string path,bool IndexFirst,bool isThousand,bool sortIndex = false);
	~Data();
	Data<T> operator[](vector<string> fts);
	vector<string> split(string s, char dec);
	void setDataname(string _data_name);
	string getDataname() const;
	int getRows() const;
	int getColumns() const;
	Matrix<T, Dynamic, Dynamic> getMatrix() const;
	vector<string> getFeatures() const;
	vector<string>getIndexs() const;
	vector<string> camma_remove(string data_string);
	void setIndex(string index);
	void removeRow(int RowToRemove);
	void removeRow(string idx);
	void removeRows(vector<string>idxs);
	void merge(Data df2);
	void addRows(vector<vector<string>>rows);
	void addColumns(unordered_map<string, vector<string>>m);
	void removeColumn(int ColToRemove);
	void removeColumns(vector<string>fts);
	void renameColumns(unordered_map<string, string>names);
	void sortbyIndex(bool ascending);
	void sortby(string feature,bool ascedning);
	Data<T> groupby(string feature, string operate);
	void dropna(int axis);
	void print();
	void to_csv(string path);
private:
	string data_name;
	int rows = 0;
	int columns = 0;
	bool hasIndexRows;
	string index_name;
	vector<string>features;
	vector<string>indexes;
	vector<vector<int>>nullpos;
	Matrix<T,Dynamic,Dynamic>mat;
	void setMatrix(unordered_map<string, vector<string>>::iterator it);
	void setMatrix(int r, int c, vector<string> cols);
};

#endif