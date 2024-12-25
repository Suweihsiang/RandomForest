#include"Label.h"
#include"DecisionTree.h"
#include"RandomForest.h"
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
    map<int, set<string>>classes = lb.get_classes();
    vector<int>y(data[0].second.size(), -1);
    vector<vector<double>> x(data[0].second.size(), vector<double>(data.size()));
    for (int i = 0; i < data[0].second.size(); i++) {
        for (int j = 1; j < data.size() - 1; j++) {
            x[i][j - 1] = stod(data[j].second[i]);
        }
        y[i] = stoi(data[5].second[i]);
    }
    vector<pair<vector<double>, int>> datas;
    for (int i = 0; i < data[0].second.size(); i++) {
        vector<double>x_data;
        for (int j = 1; j < data.size() - 1; j++) {
            x_data.push_back(stod(data[j].second[i]));
        }
        datas.push_back(make_pair(x_data, stoi(data[5].second[i])));
    }
    //RandomForest rf(100, "gini", 100, 4, 7, 1, 0.0, 0.0);
    //clock_t a = clock();
    //rf.fit(datas);
    //clock_t b = clock();
    //cout << b - a << "ms" << endl;
    //vector<int> res = rf.predict(x);
    //for (int r : res) { cout << r << ","; }
    //cout << endl;
    //cout << rf.score(x, y) << endl;
    Decision_Tree dt;
    //dt.set_params({ {"min_sample_split",7},{"ccp_alpha",0.006}, {"min_impurity_decrease",0.01} });
    //dt.get_params();
    dt.fit(x, y);
    //vector<pair<double, double>> path = dt.cost_complexity_pruning_path();
    //for (const auto p : path) {
    //    cout << "ccp_alpha : " << p.first;
    //    cout << ", impurity : " << p.second << endl;
    //}
    dt.export_tree();
    //vector<double> y_pred = dt.predict(x);
    //double score = dt.score(x, y);
    //for(auto pred : y_pred){ cout << pred << " "; }
    //cout << endl;
    //cout << score << endl;

    return 0;
}*/

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
    lb.fit(raw_data, { 1 });
    lb.transform(raw_data, { 1 });
    vector<pair<string, vector<string>>> data = lb.get_data();
    vector<int>y(data[0].second.size(), -1);
    vector<vector<double>> x(data[0].second.size(), vector<double>(data.size() - 2));
    for (int i = 0; i < data[0].second.size(); i++) {
        for (int j = 2; j < data.size(); j++) {
            x[i][j - 2] = stod(data[j].second[i]);
        }
        y[i] = stoi(data[1].second[i]);
    }
    vector<pair<vector<double>, int>> datas;
    for (int i = 0; i < data[0].second.size(); i++) {
        vector<double>x;
        for (int j = 2; j < data.size(); j++) {
            x.push_back(stod(data[j].second[i]));
        }
        datas.push_back(make_pair(x, stoi(data[1].second[i])));
    }
    RandomForest rf(100, "gini", 400, INT_MAX, 85, 1, 0.006, 0.01);
    clock_t a = clock();
    rf.fit(datas);
    clock_t b = clock();
    cout << b - a << "ms" << endl;
    //vector<int> res = rf.predict(x);
    //for (int r : res) { cout << r << ","; }
    //cout << endl;
    cout << rf.score(x, y)<<endl;
    //Decision_Tree dt;
    //dt.set_params({ {"min_sample_split",85},{"ccp_alpha",0.006}, {"min_impurity_decrease",0.01} });
    //dt.get_params();
    //clock_t a = clock();
    //dt.fit(x,y);
    //clock_t b = clock();
    //cout << b - a << "ms" << endl;
    //vector<pair<double, double>> path = dt.cost_complexity_pruning_path();
    //for (const auto p : path) {
    //    cout << "ccp_alpha : " << p.first;
    //    cout << ", impurity : " << p.second << endl;
    //}
    //dt.export_tree();
    //vector<int> y_pred = dt.predict(x);
    //double score = dt.score(x, y);
    //for (auto pred : y_pred) { cout << pred << " "; }
    //cout << endl;
    //cout << score << endl;

    return 0;
}*/

int main(int argc, char** argv) {
    vector<vector<string>>raw_data;
    ifstream ifs(argv[1]);
    string row;
    while (getline(ifs, row)) {
        bool skip = false;
        vector<string>row_data;
        string val;
        istringstream iss(row);
        int col_num = 0;
        while (getline(iss, val, ',')) {
            if (val == "") { skip = true; break; }
            row_data.push_back(val);
            col_num++;
        }
        if (skip || col_num < 9) { continue; }
        raw_data.push_back(row_data);
    }
    Label lb;
    lb.fit(raw_data, { 3,8 });
    lb.transform(raw_data, { 3,8 });
    vector<pair<string, vector<string>>> data = lb.get_data();
    //vector<int>y(data[0].second.size(), -1);
    //vector<vector<double>> x(data[0].second.size(), vector<double>(data.size() - 2));
    //for (int i = 0; i < data[0].second.size(); i++) {
    //    for (int j = 2; j < data.size(); j++) {
    //        x[i][j - 2] = stod(data[j].second[i]);
    //    }
    //    y[i] = stoi(data[1].second[i]);
    //}
    vector<pair<vector<double>, int>> datas;
    for (int i = 0; i < data[0].second.size(); i++) {
        vector<double>x;
        for (int j = 2; j < data.size(); j++) {
            x.push_back(stod(data[j].second[i]));
        }
        datas.push_back(make_pair(x, stoi(data[1].second[i])));
    }
    RandomForest rf(100, "entropy", 500, INT_MAX, 2, 1, 0.0, 0.0);
    clock_t a = clock();
    rf.fit(datas);
    clock_t b = clock();
    cout << b - a << "ms" << endl;
    //vector<int> res = rf.predict(x);
    //for (int r : res) { cout << r << ","; }
    //cout << endl;
    random_device rd;
    mt19937 g(rd());
    shuffle(datas.begin(), datas.end(), g);
    vector<pair<vector<double>, int>> test = vector<pair<vector<double>, int>>(datas.begin(), datas.begin() + 212);
    vector<int>y(test.size(), -1);
    vector<vector<double>> x(test.size(), vector<double>(data.size() - 2));
    for (int i = 0; i < test.size(); i++) {
        for (int j = 2; j < data.size(); j++) {
            x[i][j - 2] = stod(data[j].second[i]);
        }
        y[i] = stoi(data[1].second[i]);
    }
    cout << rf.score(x, y) << endl;
    //Decision_Tree dt("entropy", 4, 2, 1, 0.0, 0.0);
    //dt.set_params({ {"min_sample_split",85},{"ccp_alpha",0.006}, {"min_impurity_decrease",0.01} });
    //dt.get_params();
    //clock_t a = clock();
    //dt.fit(x,y);
    //clock_t b = clock();
    //cout << b - a << "ms" << endl;
    //vector<pair<double, double>> path = dt.cost_complexity_pruning_path();
    //for (const auto p : path) {
    //    cout << "ccp_alpha : " << p.first;
    //    cout << ", impurity : " << p.second << endl;
    //}
    //dt.export_tree();
    //vector<int> y_pred = dt.predict(x);
    //double score = dt.score(x, y);
    //for (auto pred : y_pred) { cout << pred << " "; }
    //cout << endl;
    //cout << score << endl;

    return 0;
}