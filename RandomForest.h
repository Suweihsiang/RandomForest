#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include"DecisionTree.h"
#include<string>
#include<vector>
#include<thread>
#include<mutex>
#include<condition_variable>

using std::string;
using std::vector;
using std::thread;
using std::mutex;
using std::unique_lock;
using std::condition_variable;

class RandomForest {
public:
	RandomForest();
	RandomForest(int nEstimators, string criterion, int max_depth, int min_sample_split, int min_sample_leaf, double ccp_alpha, double min_impurity_decrease);
	void fit(vector<vector<double>>& x, vector<double>& y, map<int, set<string>>classes, vector<string>features);
	vector<double> predict(vector<vector<double>> x, vector<string>features);
	double score(vector<vector<double>> x, vector<double> y, vector<string>features);
private:
	int nEstimators = 100;
	string criterion = "gini";
	int max_depth = INT_MAX;
	int min_sample_split = 2;
	int min_sample_leaf = 1;
	double ccp_alpha = 0.0;
	double min_impurity_decrease = 0.0;
	vector<Decision_Tree>trees;
};

#endif