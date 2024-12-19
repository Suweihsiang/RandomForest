#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include<iostream>
#include<string>
#include<vector>
#include<map>
#include<unordered_map>
#include<set>
#include<cmath>
#include<algorithm>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::map;
using std::pair;
using std::unordered_map;
using std::set;
using std::make_pair;
using std::sort;
using std::max;

struct Node {
	vector<int>data_index;
	int number;
	int parent_number;
	int feature_index;
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
	Decision_Tree(string criterion, int max_depth, int min_sample_split, int min_sample_leaf, double ccp_alpha, double min_impurity_decrease);
	~Decision_Tree();
	void set_params(unordered_map<string, double>params);
	void get_params();
	void fit(vector<vector<double>>& x, vector<double>& y, map<int, set<string>>&classes);
	vector<double> predict(vector<vector<double>> x);
	double score(vector<vector<double>> &x, vector<double> &y);
	vector<pair<double, double>>cost_complexity_pruning_path();
	void export_tree();
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
	map<int, int>label_count(vector<double>& data);
	double gini_impurity(vector<double>& data, map<int, int>&label_counts);
	double calc_entropy(vector<double>& data, map<int, int>&label_counts);
	void split(Node* node, vector<pair<vector<double>, double>>& data);
	void sortX(vector<pair<vector<double>, double>>& data, int &col);
	double threshold(Node* node, int &col, double &crit, vector<pair<vector<double>, double>>& data);
	void predict_node(Node* node, vector<vector<double>>& x, vector<double>& y_pred);
	void copy_tree(Node* tree, Node* new_tree);
	void construct_treemap(Node* node, map<int, vector<double>>& tree_map);
	map<int, vector<double>> cost_complexity_pruning(Node *node, map<int, vector<double>> &best_tree_map, map<int, vector<double>> &tree_map, vector<pair<double, double>>&path,int iters);
	int findparent(int number, map<int, vector<double>>& tree_map,int target);
	void construct_pruning_tree(Node &tree,map<int, vector<double>>& tree_map);
	void traverse_node(Node* node, map<int, vector<double>>& tree_map,bool &pruning);
	void export_node(Node* node);
	void print(Node* node);
	void adjust_parent_depth(Node* node);
	void pruning_min_impurity_decrease(Node* node);
};

#endif