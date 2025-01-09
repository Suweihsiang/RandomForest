#ifndef LABEL_H
#define LABEL_H
#include<map>
#include<set>
#include<vector>
#include<string>
#include<iostream>
#include<fstream>
#include<sstream>

using std::vector;
using std::set;
using std::pair;
using std::map;
using std::string;
using std::ifstream;
using std::istringstream;
using std::to_string;


class Label {
public:
	Label();																	//constructor
	~Label();																	//destructor
	void fit(vector<vector<string>>&raw_data,vector<int>cols);					//set pair of labels and their number of features
	void transform(vector<vector<string>>&raw_data, vector<int>cols);			//transform labels to numbers
	void inverse_transform(vector<vector<string>>& raw_data, vector<int>cols);  //transform numbers to labels
	vector<pair<string, vector<string>>> get_data() const;
	map<int, set<string>> get_classes() const;
	map<int, map<string, string>>get_encoder() const;
private:
	string get_key(string &value,int &col);										//get the value's label
	vector<pair<string, vector<string>>>data;									//feature : data
	map<int, set<string>>classes;												//feature : labels
	map<int, map<string,string>>encoder;										//feature : { label : number }
};

#endif