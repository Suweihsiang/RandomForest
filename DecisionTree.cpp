#include"DecisionTree.h"

Decision_Tree::Decision_Tree(){}

Decision_Tree::Decision_Tree(string criterion, int max_depth, int min_sample_split, int min_sample_leaf, double ccp_alpha, double min_impurity_decrease):criterion(criterion), max_depth(max_depth), min_sample_split(min_sample_split), min_sample_leaf(min_sample_leaf), ccp_alpha(ccp_alpha), min_impurity_decrease(min_impurity_decrease) {}

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

void Decision_Tree::fit(vector<vector<double>>& x, vector<int>& y) {
	classes = set<int>(y.begin(), y.end());
	vector<pair<vector<double>, int>>data_;//zip x and y
	vector<int>data_idx;//data index
	for (int i = 0; i < x.size(); i++) {
		data_.push_back(make_pair(x[i], y[i]));
		data_[i].first.insert(data_[i].first.begin(), i);
		data_idx.push_back(data_[i].first[0]);
	}
	Root.number = 0;
	Root.parent_number = -1;
	Root.data_index = data_idx;
	Root.samples = data_idx.size();
	vector<int>label_counts = label_count(y);//count labels' amount
	Root.criteria = (criterion == "gini") ? gini_impurity(y, label_counts) : calc_entropy(y, label_counts);
	Root.values = label_counts;
	Node* node = &Root;
	split(node,data_);
	if (ccp_alpha > 0.0) { cost_complexity_pruning_path(); }//post_pruning
	if (min_impurity_decrease > 0.0) { pruning_min_impurity_decrease(node); }//post_pruning
}

vector<int> Decision_Tree::predict(vector<vector<double>> x) {
	vector<int> y_pred(x.size());//save predict result of each data
	for (int i = 0; i < x.size(); i++) { x[i].insert(x[i].begin(), i); }//add index
	predict_node(&Root, x, y_pred);//predict by each node threshold
	return y_pred;
}

double Decision_Tree::score(vector<vector<double>> &x, vector<int> &y) {
	vector<int> y_pred = predict(x);
	double correct = 0;
	for (int i = 0; i < y_pred.size(); i++) {
		if (y_pred[i] == y[i]) { correct++; }
	}
	return correct / y_pred.size();
}

vector<pair<double, double>> Decision_Tree::cost_complexity_pruning_path() {
	map<int, vector<double>>tree_map;
	Node r;
	copy_tree(&Root, &r);//copy a new tree
	vector<pair<double, double>>path;//cost complexity pruning path
	path.push_back({ 0,0 });
	int iters = 0;
	while (!r.isLeaf) {
		construct_treemap(&r, tree_map);//{ node number : parent number , isleaf , criteria * sample , node depth }
		map<int, vector<double>> best_tree_map;
		path.push_back({ DBL_MAX,DBL_MAX });
		iters++;
		best_tree_map = cost_complexity_pruning(&r, best_tree_map, tree_map, path, iters);//get the best pruning method
		if (ccp_alpha > 0.0 && path[iters].first > ccp_alpha) { break; }//post_pruning
		construct_pruning_tree(r,best_tree_map);//follow the best tree map to construct pruning tree
		if (ccp_alpha == 0.0) { export_tree(); }
		tree_map.clear();//prepared for next pruning
	}
	if (ccp_alpha > 0.0) { copy_tree(&r, &Root); }//post_pruning
	return path;
}

void Decision_Tree::export_tree() {
	export_node(&Root);
	return;
}

int Decision_Tree::get_classes_size() const {
	return classes.size();
}

vector<int> Decision_Tree::label_count(vector<int>& data) {
	vector<int>label_counts(classes.size(), 0);
	for (const auto &d : data) {
		label_counts[d]++;
	}
	return label_counts;
}

double Decision_Tree::gini_impurity(vector<int>& data,vector<int>&label_counts) {
	double gini = 1.0;
	double total = data.size();
	for (const auto &count : label_counts) {
		if (count == 0) { continue; }
		gini -= pow(count / total, 2);
	}
	return gini;
}

double Decision_Tree::calc_entropy(vector<int>& data, vector<int>&label_counts) {
	double entropy = 0.0;
	double total = data.size();
	for (const auto &count : label_counts) {
		if (count == 0) { continue; }
		entropy -= count / total * log2(count / total);
	}
	return entropy;
}

void Decision_Tree::split(Node* node, vector<pair<vector<double>, int>>& data) {
	static int curr_depth = 1;
	node->node_layer = curr_depth;
	if (node->isLeaf) { depth = max(curr_depth, depth); return; }
	curr_depth++;//go down to next depth
	double crit_split = 1;
	vector<pair<vector<double>, int>>node_data;//store datas in this node
	for (const auto &idx : node->data_index) {
		node_data.push_back(data[idx]);
	}
	for (int i = 1; i < data[0].first.size(); i++) {
		sortX(node_data, i);//sort datas ascending according to feature i
		crit_split = threshold(node, i, crit_split, node_data);//best criterion using this feature to split
	}
	if (node->left->samples < min_sample_leaf || node->right->samples < min_sample_leaf) {//pre_pruning
		node->left = nullptr;
		node->right = nullptr;
		node->isLeaf = true;
		depth = max(--curr_depth, depth);
		return;
	}
	if (node->left->criteria == 0 || curr_depth  > max_depth || node->left->samples < min_sample_split) { node->left->isLeaf = true; }
	if (node->right->criteria == 0 || curr_depth  > max_depth || node->right->samples < min_sample_split) { node->right->isLeaf = true; }
	
	split(node->left, data);//split left node
	split(node->right,data);//split right node
	node->node_depth += max(node->left->node_depth, node->right->node_depth) + 1;//the depth of this node
	curr_depth--;
}

void Decision_Tree::sortX(vector<pair<vector<double>,int>>& data,int &col) {
	sort(data.begin(), data.end(), [&](pair<vector<double>,int> & a, pair<vector<double>,int> & b) {return a.first[col] < b.first[col]; });
}

double Decision_Tree::threshold(Node* node, int &col, double &crit, vector<pair<vector<double>, int>>& data) {
	double thres;
	int j = 0;
	vector<int>data_l, data_r;//data indices of left node and right node
	vector<int>y_l, y_r;//labels of left node and right node
	for (int i = 0; i < node->samples; i++) {//all datas are placed in right node
		data_r.push_back(data[i].first[0]);
		y_r.push_back(data[i].second);
	}
	int yl_size = 0, yr_size = y_r.size();
	for (int i = 0; i < data.size();) {
		bool move = false;
		//decide split threshold for feature i
		if (i == 0) { thres = data[i].first[col] - 1; }
		else { thres = (data[i].first[col] + data[i - 1].first[col]) / 2; }
		//
		while (j < data.size() && data[j].first[col] <= thres) {
			//move data to the left node
			move = true;
			data_l.push_back(*(data_r.begin()));
			data_r.erase(data_r.begin());
			j++;
		}
		
		if (move) { i = j; }//we can decide split threshold begin at j+1 since before it are all in the left node
		i++;
		double crit_sub = 0, crit_l = 0, crit_r = 0;
		vector<int>label_l, label_r;//labels of left node and right node
		
		if (data_l.size() > 0) {
			while (yl_size < data_l.size()) { y_l.push_back(*(y_r.begin())); y_r.erase(y_r.begin()); yl_size++; }//move labels to left node
			label_l = label_count(y_l);//amount of each labels of left node
			crit_l = (criterion == "gini") ? gini_impurity(y_l, label_l) : calc_entropy(y_l, label_l);
			crit_sub += (double)y_l.size() / node->samples * crit_l;
		}
		
		if (data_r.size() > 0) {
			label_r = label_count(y_r);//amount of each labels of right node
			crit_r = (criterion == "gini") ? gini_impurity(y_r, label_r) : calc_entropy(y_r, label_r);
			crit_sub += (double)y_r.size() / node->samples * crit_r;
		}
		
		if (crit_sub <= crit) {//is this split norm is the best currently ?
			crit = crit_sub;
			node->threshold = thres;
			node->feature_index = col - 1;
			node->left = new Node();
			node->right = new Node();
			node->left->parent = node;
			node->left->number = 2 * node->number + 1;
			node->left->parent_number = node->number;
			node->left->data_index = data_l;
			node->left->criteria = crit_l;
			node->left->samples = data_l.size();
			node->left->values = label_l;
			node->right->parent = node;
			node->right->number = 2 * node->number + 2;
			node->right->parent_number = node->number;
			node->right->data_index = data_r;
			node->right->criteria = crit_r;
			node->right->samples = data_r.size();
			node->right->values = label_r;
		}
		
	}
	return crit;
}

void Decision_Tree::predict_node(Node* node, vector<vector<double>>& x, vector<int>& y_pred) {
	if (node->isLeaf) {
		auto cls = max_element(node->values.begin(), node->values.end()) - node->values.begin();
		for (int i = 0; i < x.size(); i++) {
			int idx = x[i][0];
			y_pred[idx] = cls;//get predict label
		}
	}
	else {
		vector<vector<double>> x_left, x_right;
		int dist = node->feature_index;
		for (int i = 0; i < x.size(); i++) {
			if (x[i][dist + 1] > node->threshold) {
				x_right.push_back(x[i]);
			}
			else {
				x_left.push_back(x[i]);
			}
		}
		predict_node(node->left, x_left, y_pred);//predcit at left node
		predict_node(node->right, x_right, y_pred);//predict at right node
	}
	return;
}

void Decision_Tree::copy_tree(Node* tree, Node* new_tree) {
	new_tree->data_index = tree->data_index;
	new_tree->number = tree->number;
	new_tree->parent_number = tree->parent_number;
	new_tree->feature_index = tree->feature_index;
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

void Decision_Tree::construct_treemap(Node *node, map<int, vector<double>>&tree_map) {//{ node number : parent number , isleaf , criteria * sample , node depth }
	tree_map[node->number] = vector<double>{ static_cast<double>(node->parent_number),static_cast<double>(node->isLeaf),node->criteria * node->samples,static_cast<double>(node->node_depth)};
	if (node->isLeaf) { return; }
	construct_treemap(node->left, tree_map);
	construct_treemap(node->right, tree_map);
}

map<int, vector<double>> Decision_Tree::cost_complexity_pruning(Node *node, map<int, vector<double>> &best_tree_map, map<int, vector<double>> &tree_map, vector<pair<double, double>>&path,int iters) {
	if (tree_map[node->number][1]) { return best_tree_map; }//if node is leaf
	map<int, vector<double>>cp_tree_map = tree_map;//{ node number : parent number , isleaf , criteria * sample , node depth }
	cp_tree_map[node->number][1] = 1;//node is leaf, that is, prune the node
	for (auto &d : cp_tree_map) {
		if (d.first > node->number) {//under the node
			if (findparent(d.first, cp_tree_map, node->number) == node->number) { d.second[1] = -1; }//all the child node are pruning
		}
	}
	vector<pair<int, vector<double>>>result(cp_tree_map.begin(), cp_tree_map.end());
	sort(result.begin(), result.end(), [](pair<int,vector<double>>& a, pair<int, vector<double>>& b) {return a.second[1] > b.second[1]; });//node is leaf is put in front
	double impurity = 0.0,ccp_alpha = 0.0;
	for (const auto d : result) {
		if (d.second[1] < 1) { break; }
		impurity += d.second[2];//calculate impurity
	}
	double pre_impurity = path[iters - 1].second;
	impurity /= Root.samples;//calculate impurity
	ccp_alpha = (impurity - pre_impurity) / cp_tree_map[node->number][3];//calculate ccp_alpha
	if (ccp_alpha > path[iters - 1].first && ccp_alpha < path[iters].first) { //best pruning
		path[iters].first = ccp_alpha; 
		path[iters].second = impurity;
		best_tree_map = cp_tree_map;
	}
	best_tree_map = cost_complexity_pruning(node->left,best_tree_map, tree_map, path,iters);//find left node's best tree map
	best_tree_map = cost_complexity_pruning(node->right,best_tree_map, tree_map, path,iters);//find right node's best tree map
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
	if (tree_map[node->number][1] == 1 && !(node->isLeaf)) {//prune the node
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

void Decision_Tree::export_node(Node* node) {
	print(node);
	if (node->isLeaf) { return; }
	export_node(node->left);
	export_node(node->right);
	return;
}

void Decision_Tree::print(Node* node) {
	cout << "--------------------------------------------------------------------" << endl;
	cout << "node number = " << node->number << endl;
	cout << "node parent number = " << node->parent_number << endl;
	cout << "node layer = " << node->node_layer << endl;
	cout << "node depth = " << node->node_depth << endl;
	if (!(node->isLeaf)) { cout << "Split threashold : x[" << node->feature_index << "] <=" << node->threshold << endl; }
	cout << "node criteria = " << node->criteria << endl;
	cout << "node sample size = " << node->samples << endl;
	cout << "node label count : [";
	for (auto p : node->values) { cout << p << ","; }
	cout << "\b]" << endl;
	cout << "--------------------------------------------------------------------" << endl;
}

void Decision_Tree::pruning_min_impurity_decrease(Node* node) {
	if (node->isLeaf) { return; }
	double N = Root.samples, N_t = node->samples;
	double impurity = node->criteria;
	double N_t_R = 0, N_t_L = 0;
	double impurity_R = 0.0, impurity_L = 0.0;
	if (node->right) {
		N_t_R = node->right->samples;
		impurity_R = node->right->criteria;
	}
	if (node->left) {
		N_t_L = node->left->samples;
		impurity_L = node->left->criteria;
	}
	double impurity_decrease = N_t / N * (impurity - N_t_R / N_t * impurity_R - N_t_L / N_t * impurity_L);
	if (impurity_decrease < min_impurity_decrease) {
		node->left = nullptr;
		node->right = nullptr;
		node->isLeaf = true;
		node->node_depth = 0;
		adjust_parent_depth(node);
		return;
	}
	pruning_min_impurity_decrease(node->left);
	pruning_min_impurity_decrease(node->right);
	return;
}