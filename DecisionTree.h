#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include<iostream>
#include<eigen3/Eigen/Dense>
#include<string>
#include<vector>
#include<map>
#include<unordered_map>
#include<set>
#include<cmath>
#include<algorithm>

using std::cout;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::vector;
using std::map;
using std::unordered_map;
using std::set;
using std::sort;
using std::max;

struct Node {
	MatrixXd data;
	string feature;
	double threshold;
	double criteria;
	int samples;
	map<int, int>values;
	bool isLeaf = false;
	Node* left;
	Node* right;
};

class Decision_Tree {
public:
	Decision_Tree();
	~Decision_Tree();
	void set_params(unordered_map<string, double>params);
	void get_params();
	void fit(MatrixXd& x, VectorXd& y, map<int, set<string>>classes, vector<string>features);
	VectorXd predict(MatrixXd x, vector<string>features);
	double score(MatrixXd x, VectorXd y,vector<string>features);
private:
	Node Root;
	int depth = 1;
	string criterion = "gini";
	int max_depth = INT_MAX;
	double min_impurity_decrease = 0.0;
	vector<string>classes;
	map<int, int>label_count(VectorXd& data);
	double gini_impurity(VectorXd& data, map<int, int>label_counts);
	double calc_entropy(VectorXd& data, map<int, int>label_counts);
	double threshold(VectorXd& x, VectorXd& y, int col, double gini, vector<string>features, Node* node);
	void split(Node* node, vector<string>features);
	void predict_node(Node* node, MatrixXd& x, VectorXd& y, vector<string>features);
};

#endif