#include "RandomForest.h"

RandomForest::RandomForest() {}

RandomForest::RandomForest(int nEstimators = 100, string criterion = "gini", int fit_samples = -1, int max_depth = INT_MAX, int min_sample_split = 2, int min_sample_leaf = 1, double ccp_alpha = 0.0, double min_impurity_decrease = 0.0) :nEstimators(nEstimators), criterion(criterion), fit_samples(fit_samples), max_depth(max_depth), min_sample_split(min_sample_split), min_sample_leaf(min_sample_leaf), ccp_alpha(ccp_alpha), min_impurity_decrease(min_impurity_decrease) {}

RandomForest::~RandomForest() {}

void RandomForest::fit(vector<pair<vector<double>, int>>& datas) {
	vector<thread> workers;//collection of threads
	mutex queue_mutex;
	condition_variable condition;
	bool ready = false;
	int complete = 0;
	for (int i = 0; i < nEstimators; i++) {
		workers.emplace_back([&,i]() {
			unique_lock<mutex> lock(queue_mutex);
			condition.wait(lock, [&]() {return ready; });
			vector<pair<vector<double>, int>>data_fit;
			//randomly choose a number of datas to train
			if (fit_samples != -1) {
				random_device rd;
				mt19937 g(rd());
				shuffle(datas.begin(), datas.end(), g);
				data_fit = vector<pair<vector<double>, int>>(datas.begin(), datas.begin() + fit_samples);
			}
			//
			else { data_fit = datas; }
			vector<vector<double>>x(data_fit.size());
			vector<int>y(data_fit.size());
			for (int i = 0; i < data_fit.size(); i++) {
				x[i] = data_fit[i].first;
				y[i] = data_fit[i].second;
			}
			Decision_Tree dt(criterion, max_depth, min_sample_split, min_sample_leaf, ccp_alpha, min_impurity_decrease);
			dt.fit(x,y);
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

vector<int> RandomForest::predict(vector<vector<double>>& x) {
	vector<vector<int>>vote(x.size(),vector<int>(trees[0].get_classes_size(),0));//vote by each estimator
	for (int i = 0; i < trees.size(); i++) {
		vector<int>result = trees[i].predict(x);
		for (int j = 0; j < x.size(); j++) {
			vote[j][result[j]]++;
		}
	}
	vector<int>y_pred(x.size());
	for (int i = 0; i < x.size(); i++) {
		y_pred[i] = max_element(vote[i].begin(), vote[i].end()) - vote[i].begin();
	}
	return y_pred;
}

double RandomForest::score(vector<vector<double>>& x, vector<int>& y) {
	vector<int> y_pred = predict(x);
	double correct = 0;
	for (int i = 0; i < y_pred.size(); i++) {
		if (y_pred[i] == y[i]) { correct++; }
	}
	return correct / y_pred.size();
}