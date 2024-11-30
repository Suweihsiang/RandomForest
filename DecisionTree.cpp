#include"DecisionTree.h"

Decision_Tree::Decision_Tree() {}

Decision_Tree::~Decision_Tree() {}

void Decision_Tree::set_params(unordered_map<string, double>params) {
	if (params.find("max_depth") != params.end()) {
		max_depth = params["max_depth"];
	}
	if (params.find("min_impurity_decrease") != params.end()) {
		min_impurity_decrease = params["min_impurity_decrease"];
	}
	if (params.find("criterion") != params.end()) {
		criterion = (params["criterion"] == 1) ? "gini" : "entropy";
	}
}

void Decision_Tree::get_params() {
	cout << "criterion = " << criterion << ",max_depth = " << max_depth << ",min_impurity_decrease = " << min_impurity_decrease << endl;
}

void Decision_Tree::fit(MatrixXd& x, VectorXd& y, map<int, set<string>>classes, vector<string>features) {
	MatrixXd data_ = x;
	data_.conservativeResize(data_.rows(), data_.cols() + 1);
	data_.col(data_.cols() - 1) = y;
	Root.data = data_;
	Root.samples = data_.rows();
	map<int, int>label_counts = label_count(y);
	Root.criteria = (criterion == "gini") ? gini_impurity(y, label_counts) : calc_entropy(y, label_counts);
	Root.values = label_counts;
	Node* node = &Root;
	split(node, features);
}

VectorXd Decision_Tree::predict(MatrixXd x, vector<string> features) {
	VectorXd y_pred(x.rows());
	x.conservativeResize(x.rows(), x.cols() + 1);
	x.block(0, 1, x.rows(), x.cols() - 1) = x.block(0, 0, x.rows(), x.cols() - 1).eval();
	x.col(0) = VectorXd::LinSpaced(x.rows(), 0, x.rows() - 1);
	predict_node(&Root, x, y_pred, features);
	return y_pred;
}

double Decision_Tree::score(MatrixXd x, VectorXd y,vector<string>features) {
	VectorXd y_pred = predict(x, features);
	double correct = 0;
	for (int i = 0; i < y_pred.size(); i++) {
		if (y_pred(i) == y(i)) { correct++; }
	}
	return correct / y_pred.size();
}

map<int,int> Decision_Tree::label_count(VectorXd& data) {
	map<int, int>label_counts;
	for (int d : data) {
		label_counts[d]++;
	}
	return label_counts;
}

double Decision_Tree::gini_impurity(VectorXd& data,map<int,int>label_counts) {
	double gini = 1.0;
	double total = data.size();
	for (const auto count : label_counts) {
		gini -= pow(count.second / total, 2);
	}
	return gini;
}

double Decision_Tree::calc_entropy(VectorXd& data, map<int, int>label_counts) {
	double entropy = 0.0;
	double total = data.size();
	for (const auto count : label_counts) {
		entropy -= count.second / total * log10(count.second / total);
	}
	return entropy;
}

double Decision_Tree::threshold(VectorXd& x, VectorXd& y, int col, double crit, vector<string>features, Node* node) {
	double thres;
	sort(x.data(), x.data() + x.size());
	for (int i = 0; i < x.size(); i++) {
		if (i == 0) { thres = x(i) - 1; }
		else {
			double pre_thres = thres;
			thres = (i == x.size() - 1) ? x(i) + 1 : (x(i) + x(i + 1)) / 2;
			if (pre_thres == thres) { continue; }
		}
		MatrixXd data_l, data_r;
		for (int r = 0; r < x.size(); r++) {
			if (node->data(r, col) > thres) {
				data_r.conservativeResize(data_r.rows() + 1, node->data.cols());
				data_r.row(data_r.rows() - 1) = node->data.row(r);
			}
			else {
				data_l.conservativeResize(data_l.rows() + 1, node->data.cols());
				data_l.row(data_l.rows() - 1) = node->data.row(r);
			}
		}
		double crit_sub = 0, crit_l = 0, crit_r = 0;
		map<int, int>label_l, label_r;
		if (data_l.cols() > 0) {
			VectorXd y_l = data_l.col(data_l.cols() - 1);
			label_l = label_count(y_l);
			crit_l = (criterion == "gini") ? gini_impurity(y_l,label_l) : calc_entropy(y_l,label_l);
			crit_sub += (double)y_l.size() / node->data.rows() * crit_l;
		}
		if (data_r.cols() > 0) {
			VectorXd y_r = data_r.col(data_r.cols() - 1);
			label_r = label_count(y_r);
			crit_r = (criterion == "gini") ? gini_impurity(y_r,label_r) : calc_entropy(y_r,label_r);
			crit_sub += (double)y_r.size() / node->data.rows() * crit_r;
		}
		if (crit_sub < crit) {
			crit = crit_sub;
			node->threshold = thres;
			node->feature = features[col];
			node->left = new Node();
			node->right = new Node();
			node->left->data = data_l;
			node->left->criteria = crit_l;
			node->left->samples = data_l.rows();
			node->left->values = label_l;
			node->right->data = data_r;
			node->right->criteria = crit_r;
			node->right->samples = data_r.rows();
			node->right->values = label_r;
		}
	}
	return crit;
}

void Decision_Tree::split(Node* node, vector<string>features) {
	cout << "------------------------------------" << endl;
	static int curr_depth = 1;
	if (node->isLeaf) { depth = max(curr_depth,depth); return; }
	curr_depth++;
	double crit_split = 1;
	VectorXd y = node->data.col(node->data.cols() - 1);
	for (int i = 0; i < node->data.cols() - 1; i++) {
		VectorXd data_i = node->data.col(i);
		crit_split = threshold(data_i, y, i, crit_split, features, node);
	}
	if (node->left->criteria == 0 || curr_depth + 1 > max_depth) { node->left->isLeaf = true; }
	if (node->right->criteria == 0 || curr_depth + 1 > max_depth) { node->right->isLeaf = true; }

	cout << "Split threashold : " << node->feature << "<=" << node->threshold << endl;
	cout << "node criteria = " << node->criteria << endl;
	cout << "node sample size = " << node->samples << endl;
	cout << "node label count : [";
	for (auto p : node->values) { cout << p.first << ":" << p.second << ","; }
	cout << "\b]" << endl;
	cout << "left criteria = " << node->left->criteria << endl;
	cout << "left sample size = " << node->left->samples << endl;
	cout << "left label count : [";
	for (auto p : node->left->values) { cout << p.first << ":" << p.second << ","; }
	cout << "\b]" << endl;
	cout << "right criteria = " << node->right->criteria << endl;
	cout << "right sample size = " << node->right->samples << endl;
	cout << "right label count : [";
	for (auto p : node->right->values) { cout << p.first << ":" << p.second << ","; }
	cout << "\b]" << endl;

	split(node->left, features);
	split(node->right, features);
	curr_depth--;
}

void Decision_Tree::predict_node(Node* node, MatrixXd& x, VectorXd& y_pred, vector<string>features) {
	if (node->isLeaf) {
		auto cls = max_element(node->values.begin(), node->values.end(), [](const auto& left, const auto& right) {return left.second < right.second; });
		for (int i = 0; i < x.rows(); i++) {
			int idx = x(i, 0);
			y_pred(idx) = cls->first;
		}
	}
	else {
		MatrixXd x_left, x_right;
		vector<string>::iterator it = find(features.begin(), features.end(), node->feature);
		int dist = distance(features.begin(), it);
		for (int i = 0; i < x.rows(); i++) {
			if (x(i, dist + 1) > node->threshold) {
				x_right.conservativeResize(x_right.rows() + 1, x.cols());
				x_right.row(x_right.rows() - 1) = x.row(i);
			}
			else {
				x_left.conservativeResize(x_left.rows() + 1, x.cols());
				x_left.row(x_left.rows() - 1) = x.row(i);
			}
		}
		predict_node(node->left, x_left, y_pred, features);
		predict_node(node->right, x_right, y_pred, features);
	}
	return;
}