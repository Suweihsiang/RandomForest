#include"Label.h"
#include"DecisionTree.h"
#include<unordered_set>
#include<Python.h>
#include"matplotlibcpp.h"
#include<cuda.h>
#include<cuda_runtime.h>
#include<ctime>
#include "utils.cuh"

using std::unordered_set;

namespace plt = matplotlibcpp;

/*int main(int argc, char** argv) {
    vector<vector<string>>raw_data;
    ifstream ifs(argv[1]);
    string row;
    while (getline(ifs, row)) {
        vector<string>row_data;
        string val;
        istringstream iss(row);
        while (getline(iss, val, ',')) {
            row_data.push_back(val);
        }
        raw_data.push_back(row_data);
    }
    Label lb;
    lb.fit(raw_data, { 5 });
    lb.transform(raw_data, { 5 });
    vector<pair<string, vector<string>>> data = lb.get_data();
    vector<double>y;
    for (int i = 0; i < data[0].second.size(); i++) {
        y.push_back(stod(data[5].second[i]));
    }
    vector<string> features;
    vector<vector<double>> x(data[0].second.size(), vector<double>(data.size()));
    for (int i = 1; i < data.size() - 1; i++) {
        features.push_back(data[i].first);
        for (int j = 0; j < data[0].second.size(); j++) {
            x[j][i - 1] = stod(data[i].second[j]);
        }
    }
    Decision_Tree dt;
    dt.set_params({ {"min_sample_split",7},{"ccp_alpha",0.006}, {"min_impurity_decrease",0.01} });
    dt.get_params();
    dt.fit(x, y,lb.get_classes(), features);
    //vector<pair<double, double>> path = dt.cost_complexity_pruning_path();
    //for (const auto p : path) {
    //    cout << "ccp_alpha : " << p.first;
    //    cout << ", impurity : " << p.second << endl;
    //}
    dt.export_tree();
    //vector<double> y_pred = dt.predict(x, features);
    //double score = dt.score(x, y, features);
    //for(auto pred : y_pred){ cout << pred << " "; }
    //cout << endl;
    //cout << score << endl;

    return 0;
}*/

int main(int argc, char** argv) {
    vector<vector<string>>raw_data;
    ifstream ifs(argv[1]);
    string row;
    while (getline(ifs, row)) {
        vector<string>row_data;
        string val;
        istringstream iss(row);
        while (getline(iss, val, ',')) {
            row_data.push_back(val);
        }
        raw_data.push_back(row_data);
    }
    Label lb;
    lb.fit(raw_data, { 1 });
    lb.transform(raw_data, { 1 });
    vector<pair<string, vector<string>>> data = lb.get_data();
    vector<double>y;
    for (int i = 0; i < data[0].second.size(); i++) {
        y.push_back(stod(data[1].second[i]));
    }
    vector<string> features;
    vector<vector<double>> x(data[0].second.size(), vector<double>(data.size()));
    for (int i = 2; i < data.size(); i++) {
        features.push_back(data[i].first);
        for (int j = 0; j < data[0].second.size(); j++) {
            x[j][i - 2] = stod(data[i].second[j]);
        }
    }
    Decision_Tree dt;
    //dt.set_params({ {"min_sample_split",85},{"ccp_alpha",0.006}, {"min_impurity_decrease",0.01} });
    dt.get_params();
    clock_t a = clock();
    dt.fit(x, y, lb.get_classes(), features);
    clock_t b = clock();
    cout << b - a << "ms" << endl;
    //vector<pair<double, double>> path = dt.cost_complexity_pruning_path();
    //for (const auto p : path) {
    //    cout << "ccp_alpha : " << p.first;
    //    cout << ", impurity : " << p.second << endl;
    //}
    dt.export_tree();
    //vector<double> y_pred = dt.predict(x, features);
    //double score = dt.score(x, y, features);
    //for (auto pred : y_pred) { cout << pred << " "; }
    //cout << endl;
    //cout << score << endl;

    return 0;
}