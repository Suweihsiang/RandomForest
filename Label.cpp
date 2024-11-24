#include"Label.h"

Label::Label(){}

Label::~Label(){}

void Label::fit(vector<vector<string>>&raw_data,vector<int>cols) {
	for (const int col : cols) {
		set<string>label_set;
		for (int i = 1; i < raw_data.size(); i++) {
			label_set.insert(raw_data[i][col]);
		}
		classes[col] = set<string>(label_set.begin(),label_set.end());
		int number = 0;
		for (const string label : label_set) {
			encoder[col].insert({ label,to_string(number++) });
		}
	}
}

void Label::transform(vector<vector<string>>&raw_data, vector<int>cols) {
	for (const int col : cols) {
		for (int i = 1; i < raw_data.size(); i++) {
			raw_data[i][col] = encoder[col][raw_data[i][col]];
		}
	}
	for (int i = 0; i < raw_data[0].size(); i++) {
		if(data.find(raw_data[0][i]) != data.end()){ data[raw_data[0][i]] = vector<string>(); }
		for (int j = 1; j < raw_data.size(); j++) {
			data[raw_data[0][i]].push_back(raw_data[j][i]);
		}
	}
}

void Label::inverse_transform(vector<vector<string>>& raw_data, vector<int>cols) {
	for (int col : cols) {
		for (int i = 1; i < raw_data.size(); i++) {
			raw_data[i][col] = get_key(raw_data[i][col],col);
		}
	}
}

string Label::get_key(string &value,int &col) {
	for (const auto& v : encoder[col]) {
		if (value == v.second) {
			return v.first;
		}
	}
}

unordered_map<string, vector<string>> Label::get_data() const { return data; }

map<int, set<string>> Label::get_classes() const { return classes; }

map<int, map<string, string>> Label::get_encoder() const { return encoder; }