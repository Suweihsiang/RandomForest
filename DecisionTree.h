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
using std::pair;
using std::unordered_map;
using std::set;
using std::sort;
using std::max;

struct Node {
	MatrixXd data;
	int number;
	int parent_number;
	string feature;
	double threshold;
	double criteria;
	int samples;
	map<int, int>values;
	bool isLeaf = false;
	int node_layer;
	int node_depth = 0;
	Node* parent;
	Node* left;
	Node* right;
};

class Decision_Tree {
public:
	Decision_Tree();
	~Decision_Tree();
	void set_params(unordered_map<string, double>params);
	void get_params();
	Node get_root() ;
	void fit(MatrixXd& x, VectorXd& y, map<int, set<string>>classes, vector<string>features);
	VectorXd predict(MatrixXd x, vector<string>features);
	double score(MatrixXd x, VectorXd y,vector<string>features);
	vector<pair<double, double>>cost_complexity_pruning_path();
	void export_tree(Node* node);
private:
	Node Root;
	int depth = 1;
	string criterion = "gini";
	int max_depth = INT_MAX;
	int min_sample_split = 2;
	int min_sample_leaf = 1;
	double ccp_alpha = 0.0;
	double min_impurity_decrease = 0.0;
	vector<string>classes;
	map<int, int>label_count(VectorXd& data);
	double gini_impurity(VectorXd& data, map<int, int>label_counts);
	double calc_entropy(VectorXd& data, map<int, int>label_counts);
	void split(Node* node, vector<string>features);
	void sortX(MatrixXd& x, int col);
	double threshold(Node* node, VectorXd& y,int col, double crit, vector<string>features);
	void predict_node(Node* node, MatrixXd& x, VectorXd& y, vector<string>features);
	void copy_tree(Node* tree, Node* new_tree);
	void construct_treemap(Node *node,map<int, vector<double>>&tree_map);
	map<int, vector<double>> cost_complexity_pruning(Node *node, map<int, vector<double>> &best_tree_map, map<int, vector<double>> &tree_map, vector<pair<double, double>>&path,int iters);
	int findparent(int number, map<int, vector<double>>& tree_map,int target);
	void construct_pruning_tree(Node &tree,map<int, vector<double>>& tree_map);
	void traverse_node(Node* node, map<int, vector<double>>& tree_map,bool &pruning);
	void print(Node* node);
	void adjust_parent_depth(Node* node);
	void pruning_min_impurity_decrease(Node* node);
};

#endif