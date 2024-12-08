#include"DecisionTree.h"

Decision_Tree::Decision_Tree() {}

Decision_Tree::~Decision_Tree() {}

void Decision_Tree::set_params(unordered_map<string, double>params) {
	if (params.find("max_depth") != params.end()) {
		max_depth = params["max_depth"];
	}
	if (params.find("min_sample_split") != params.end()) {
		min_sample_split = params["min_sample_split"];
	}
	if (params.find("min_sample_leaf") != params.end()) {
		min_sample_leaf = params["min_sample_leaf"];
	}
	if (params.find("ccp_alpha") != params.end()) {
		ccp_alpha = params["ccp_alpha"];
	}
	if (params.find("min_impurity_decrease") != params.end()) {
		min_impurity_decrease = params["min_impurity_decrease"];
	}
	if (params.find("criterion") != params.end()) {
		criterion = (params["criterion"] == 1) ? "gini" : "entropy";
	}
}

void Decision_Tree::get_params() {
	cout << "criterion = " << criterion << ",max_depth = " << max_depth << ",min_sample_split = " << min_sample_split << ",min_sample_leaf = " << min_sample_leaf << endl ;
	cout << "ccp_alpha = " << ccp_alpha << ",min_impurity_decrease = " << min_impurity_decrease << endl;
}

Node Decision_Tree::get_root() {
	return Root;
}

void Decision_Tree::fit(MatrixXd& x, VectorXd& y, map<int, set<string>>classes, vector<string>features) {
	MatrixXd data_ = x;
	data_.conservativeResize(data_.rows(), data_.cols() + 1);
	data_.col(data_.cols() - 1) = y;
	Root.number = 0;
	Root.parent_number = -1;
	Root.data = data_;
	Root.samples = data_.rows();
	map<int, int>label_counts = label_count(y);
	Root.criteria = (criterion == "gini") ? gini_impurity(y, label_counts) : calc_entropy(y, label_counts);
	Root.values = label_counts;
	Node* node = &Root;
	split(node, features);
	if (ccp_alpha > 0.0) { cost_complexity_pruning_path(); }
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

vector<pair<double, double>> Decision_Tree::cost_complexity_pruning_path() {
	map<int, vector<double>>tree_map;
	Node r;
	copy_tree(&Root, &r);
	vector<pair<double, double>>path;
	path.push_back({ 0,0 });
	int iters = 0;
	while (!r.isLeaf) {
		construct_treemap(&r, tree_map);
		map<int, vector<double>> best_tree_map;
		path.push_back({ DBL_MAX,DBL_MAX });
		iters++;
		best_tree_map = cost_complexity_pruning(&r, best_tree_map, tree_map, path, iters);
		if (ccp_alpha > 0.0 && path[iters].first > ccp_alpha) { break; }
		construct_pruning_tree(r,best_tree_map);
		if (ccp_alpha == 0.0) { export_tree(&r); }
		tree_map.clear();
	}
	if (ccp_alpha > 0.0) { copy_tree(&r, &Root); }
	return path;
}

void Decision_Tree::export_tree(Node* node) {
	print(node);
	if (node->isLeaf) { return;}
	export_tree(node->left);
	export_tree(node->right);
	return;
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
			node->left->parent = node;
			node->left->number = 2 * node->number + 1;
			node->left->parent_number = node->number;
			node->left->data = data_l;
			node->left->criteria = crit_l;
			node->left->samples = data_l.rows();
			node->left->values = label_l;
			node->right->parent = node;
			node->right->number = 2 * node->number + 2;
			node->right->parent_number = node->number;
			node->right->data = data_r;
			node->right->criteria = crit_r;
			node->right->samples = data_r.rows();
			node->right->values = label_r;
		}
	}
	return crit;
}

void Decision_Tree::split(Node* node, vector<string>features) {
	static int curr_depth = 1;
	node->node_layer = curr_depth;
	if (node->isLeaf) { depth = max(curr_depth, depth); return; }
	curr_depth++;
	double crit_split = 1;
	VectorXd y = node->data.col(node->data.cols() - 1);
	for (int i = 0; i < node->data.cols() - 1; i++) {
		VectorXd data_i = node->data.col(i);
		crit_split = threshold(data_i, y, i, crit_split, features, node);
	}
	if (node->left->samples < min_sample_leaf || node->right->samples < min_sample_leaf) {
		node->left = nullptr;
		node->right = nullptr;
		node->isLeaf = true;
		depth = max(--curr_depth, depth);
		return;
	}
	if (node->left->criteria == 0 || curr_depth + 1 > max_depth || node->left->samples < min_sample_split) { node->left->isLeaf = true; }
	if (node->right->criteria == 0 || curr_depth + 1 > max_depth || node->right->samples < min_sample_split) { node->right->isLeaf = true; }
	
	split(node->left, features);
	split(node->right, features);
	node->node_depth += max(node->left->node_depth, node->right->node_depth) + 1;
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

void Decision_Tree::copy_tree(Node* tree, Node* new_tree) {
	new_tree->data = tree->data;
	new_tree->number = tree->number;
	new_tree->parent_number = tree->parent_number;
	new_tree->feature = tree->feature;
	new_tree->threshold = tree->threshold;
	new_tree->criteria = tree->criteria;
	new_tree->samples = tree->samples;
	new_tree->values = tree->values;
	new_tree->isLeaf = tree->isLeaf;
	new_tree->node_layer = tree->node_layer;
	new_tree->node_depth = tree->node_depth;
	if (tree->isLeaf) { return; }
	new_tree->left = new Node();
	new_tree->right = new Node();
	new_tree->left->parent = new_tree;
	new_tree->right->parent = new_tree;
	copy_tree(tree->left, new_tree->left);
	copy_tree(tree->right, new_tree->right);
	return;
}

void Decision_Tree::construct_treemap(Node *node, map<int, vector<double>>&tree_map) {
	tree_map[node->number] = vector<double>{ static_cast<double>(node->parent_number),static_cast<double>(node->isLeaf),node->criteria * node->samples,static_cast<double>(node->node_depth)};
	if (node->isLeaf) { return; }
	construct_treemap(node->left, tree_map);
	construct_treemap(node->right, tree_map);
}

map<int, vector<double>> Decision_Tree::cost_complexity_pruning(Node *node, map<int, vector<double>> &best_tree_map, map<int, vector<double>> &tree_map, vector<pair<double, double>>&path,int iters) {
	if (tree_map[node->number][1]) { return best_tree_map; }
	map<int, vector<double>>cp_tree_map = tree_map;
	cp_tree_map[node->number][1] = 1;
	for (auto &d : cp_tree_map) {
		if (d.first > node->number) {
			if (findparent(d.first, cp_tree_map, node->number) == node->number) { d.second[1] = -1; }
		}
	}
	vector<pair<int, vector<double>>>result(cp_tree_map.begin(), cp_tree_map.end());
	sort(result.begin(), result.end(), [](pair<int,vector<double>>& a, pair<int, vector<double>>& b) {return a.second[1] > b.second[1]; });
	double impurity = 0.0,ccp_alpha = 0.0;
	for (const auto d : result) {
		if (d.second[1] < 1) { break; }
		impurity += d.second[2];
	}
	double pre_impurity = path[iters - 1].second;
	impurity /= Root.samples;
	ccp_alpha = (impurity - pre_impurity) / cp_tree_map[node->number][3];
	if (ccp_alpha > path[iters - 1].first && ccp_alpha < path[iters].first) { 
		path[iters].first = ccp_alpha; 
		path[iters].second = impurity;
		best_tree_map = cp_tree_map;
	}
	best_tree_map = cost_complexity_pruning(node->left,best_tree_map, tree_map, path,iters);
	best_tree_map = cost_complexity_pruning(node->right,best_tree_map, tree_map, path,iters);
	return best_tree_map;
}

int Decision_Tree::findparent(int number, map<int, vector<double>>& tree_map,int target) {
	int parent = tree_map[number][0];
	if (parent <= target) { return parent; }
	else { return findparent(parent, tree_map, target); }
}

void Decision_Tree::construct_pruning_tree(Node &tree,map<int, vector<double>>& tree_map) {
	bool pruning = false;
	traverse_node(&tree, tree_map,pruning);
	return ;
}

void Decision_Tree::traverse_node(Node* node, map<int, vector<double>>& tree_map,bool& pruning) {
	if (node->isLeaf) { return; }
	if (tree_map[node->number][1] == 1 && !(node->isLeaf)) {
		node->left = nullptr;
		node->right = nullptr;
		node->isLeaf = true;
		node->node_depth = 0;
		pruning = true;
		adjust_parent_depth(node);
		return;
	}
	if (!pruning) { traverse_node(node->left, tree_map, pruning); }
	if (!pruning) { traverse_node(node->right, tree_map, pruning); }
	return;
}

void Decision_Tree::adjust_parent_depth(Node* node) {
	if (node->number == 0) { return; }
	Node* parent_node = node->parent;
	int pre_depth = parent_node->node_depth;
	parent_node->node_depth = max(parent_node->left->node_depth, parent_node->right->node_depth) + 1;
	if (pre_depth == parent_node->node_depth) { return; }
	adjust_parent_depth(parent_node);
	return;
}

void Decision_Tree::print(Node* node) {
	cout << "--------------------------------------------------------------------" << endl;
	cout << "node number = " << node->number << endl;
	cout << "node parent number = " << node->parent_number << endl;
	cout << "node layer = " << node->node_layer << endl;
	cout << "node depth = " << node->node_depth << endl;
	if (!(node->isLeaf)) { cout << "Split threashold : " << node->feature << "<=" << node->threshold << endl; }
	cout << "node criteria = " << node->criteria << endl;
	cout << "node sample size = " << node->samples << endl;
	cout << "node label count : [";
	for (auto p : node->values) { cout << p.first << ":" << p.second << ","; }
	cout << "\b]" << endl;
	cout << "--------------------------------------------------------------------" << endl;
}