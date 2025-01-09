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
	vector<int>data_index;//datas of the Node
	int number;			  //Node number
	int parent_number;	  //Node's parent number
	int feature_index;	  //split feature's index
	double threshold;	  //split feature's threshold
	double criteria;	  //gini or entropy
	int samples;		  //samples amount of the Node
	vector<int>values;	  //samples amount of each label
	bool isLeaf = false;
	int node_layer;		  //the layer of this Node
	int node_depth = 0;	  //the depth under this Node
	Node* parent;
	Node* left;
	Node* right;
};

class Decision_Tree {
public:
	Decision_Tree();										   //constructor
	Decision_Tree(string criterion, int max_depth, int min_sample_split, int min_sample_leaf, double ccp_alpha, double min_impurity_decrease);
	~Decision_Tree();										   //destructor
	void set_params(unordered_map<string, double>params);      //set parameters
	void get_params();									       //show parameters
	void fit(vector<vector<double>>& x, vector<int>& y);
	vector<int> predict(vector<vector<double>> x);
	double score(vector<vector<double>> &x, vector<int> &y);   //accuracy of this prediction
	vector<pair<double, double>>cost_complexity_pruning_path();//post_pruning method
	void export_tree();										   //show the decision tree
	int get_classes_size() const;							   //how many labels
private:
	Node Root;
	int depth = 1;
	string criterion = "gini";										  //gini or entropy
	int max_depth = INT_MAX;										  //pre_pruning method
	int min_sample_split = 2;										  //pre_pruning method
	int min_sample_leaf = 1;										  //pre_pruning method
	double ccp_alpha = 0.0;											  //post_pruning method
	double min_impurity_decrease = 0.0;								  //post_pruning method
	set<int>classes;												  //category of datas
	vector<int>label_count(vector<int>& data);						  //count datas' labels
	double gini_impurity(vector<int>& data, vector<int>&label_counts);//calculate gini impurity
	double calc_entropy(vector<int>& data, vector<int>&label_counts); //calculate entropy
	void split(Node* node, vector<pair<vector<double>, int>>& data);  //decide node split norm
	void sortX(vector<pair<vector<double>, int>>& data, int &col);    //sort data ascending
	double threshold(Node* node, int &col, double &crit, vector<pair<vector<double>, int>>& data);//decide split threshold
	void predict_node(Node* node, vector<vector<double>>& x, vector<int>& y_pred);
	void copy_tree(Node* tree, Node* new_tree);
	void construct_treemap(Node* node, map<int, vector<double>>& tree_map);
	map<int, vector<double>> cost_complexity_pruning(Node *node, map<int, vector<double>> &best_tree_map, map<int, vector<double>> &tree_map, vector<pair<double, double>>&path,int iters);
	int findparent(int number, map<int, vector<double>>& tree_map,int target);//find node parent that above target
	void construct_pruning_tree(Node &tree,map<int, vector<double>>& tree_map);//follow tree_map to construct the pruning tree
	void traverse_node(Node* node, map<int, vector<double>>& tree_map,bool &pruning);
	void export_node(Node* node);
	void print(Node* node);
	void adjust_parent_depth(Node* node);								//adjust lineal node of this pruning node
	void pruning_min_impurity_decrease(Node* node);						//post_pruning method
};

#endif