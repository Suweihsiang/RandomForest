#include "RandomForest.h"

RandomForest::RandomForest() {}

RandomForest::RandomForest(int nEstimators = 100, string criterion = "gini", int max_depth = INT_MAX, int min_sample_split = 2, int min_sample_leaf = 1, double ccp_alpha = 0.0, double min_impurity_decrease = 0.0) :nEstimators(nEstimators),criterion(criterion), max_depth(max_depth), min_sample_split(min_sample_split), min_sample_leaf(min_sample_leaf), ccp_alpha(ccp_alpha), min_impurity_decrease(min_impurity_decrease) {}

void RandomForest::fit(vector<vector<double>>& x, vector<int>& y) {
	vector<thread> workers;
	mutex queue_mutex;
	condition_variable condition;
	bool ready = false;
	int complete = 0;
	for (int i = 0; i < nEstimators; i++) {
		workers.emplace_back([&,i]() {
			unique_lock<mutex> lock(queue_mutex);
			condition.wait(lock, [&]() {return ready; });
			Decision_Tree dt(criterion, max_depth, min_sample_split, min_sample_leaf, ccp_alpha, min_impurity_decrease);
			dt.fit(x, y);
			cout <<"fit no." << i << " tree. (" << ++complete << "/" << nEstimators << ")" << endl;
			trees.push_back(dt); }
		);
	}
	ready = true;
	condition.notify_all();
	for (int i = 0; i < nEstimators; i++) {
		workers[i].join();
	}
}