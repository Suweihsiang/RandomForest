#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include"DecisionTree.h"
#include<string>
#include<vector>
#include<thread>
#include<mutex>
#include<condition_variable>
#include<random>
#include<algorithm>

using std::string;
using std::vector;
using std::thread;
using std::mutex;
using std::unique_lock;
using std::condition_variable;
using std::random_device;
using std::mt19937;
using std::shuffle;

class RandomForest {
public:
	RandomForest();												//constructor
	RandomForest(int nEstimators, string criterion, int fit_samples, int max_depth, int min_sample_split, int min_sample_leaf, double ccp_alpha, double min_impurity_decrease);
	~RandomForest() {}											//destructor
	void fit(vector<pair<vector<double>, int>>& datas);
	vector<int> predict(vector<vector<double>>& x);
	double score(vector<vector<double>>& x, vector<int>& y);
private:
	int nEstimators = 100;										//the number of decision trees
	string criterion = "gini";									//gini or entropy
	int fit_samples = -1;										//samples that to train
	int max_depth = INT_MAX;									//pre_pruning
	int min_sample_split = 2;									//pre_pruning
	int min_sample_leaf = 1;									//pre_pruning
	double ccp_alpha = 0.0;										//post_pruning
	double min_impurity_decrease = 0.0;							//post_pruning
	vector<Decision_Tree>trees;									//collection of decision trees
};

#endif