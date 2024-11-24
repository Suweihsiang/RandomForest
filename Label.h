#ifndef LABEL_H
#define LABEL_H
#include<unordered_map>
#include<map>
#include<set>
#include<vector>
#include<string>
#include<iostream>
#include<fstream>
#include<sstream>

using std::vector;
using std::set;
using std::unordered_map;
using std::map;
using std::string;
using std::ifstream;
using std::istringstream;
using std::to_string;


class Label {
public:
	Label();
	~Label();
	void fit(vector<vector<string>>&raw_data,vector<int>cols);
	void transform(vector<vector<string>>&raw_data, vector<int>cols);
	void inverse_transform(vector<vector<string>>& raw_data, vector<int>cols);
	unordered_map<string, vector<string>> get_data() const;
	map<int, set<string>> get_classes() const;
	map<int, map<string, string>>get_encoder() const;
private:
	string get_key(string &value,int &col);
	unordered_map<string, vector<string>>data;
	map<int, set<string>>classes;
	map<int, map<string,string>>encoder;
};

#endif