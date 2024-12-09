#include"data.h"
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
    lb.fit(raw_data, { 5 });
    lb.transform(raw_data, { 5 });
    unordered_map<string, vector<string>> data = lb.get_data();
    Data<double> df(data, true, false);
    MatrixXd mat = df.getMatrix();
    MatrixXd x = mat.block(0, 0, mat.rows(), mat.cols() - 1);
    VectorXd y = mat.col(mat.cols() - 1);
    Decision_Tree dt;
    dt.set_params({ {"min_sample_split",7},{"ccp_alpha",0.006}, {"min_impurity_decrease",0.01} });
    dt.get_params();
    dt.fit(x, y,lb.get_classes(), df.getFeatures());
    //vector<pair<double, double>> path = dt.cost_complexity_pruning_path();
    //for (const auto p : path) {
    //    cout << "ccp_alpha : " << p.first;
    //    cout << ", impurity : " << p.second << endl;
    //}
    Node r = dt.get_root();
    dt.export_tree(&r);
    //VectorXd y_pred = dt.predict(x, df.getFeatures());
    //double score = dt.score(x, y, df.getFeatures());
    //cout << y_pred.transpose() << endl;
    //cout << score << endl;

    return 0;
}

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
    unordered_map<string, vector<string>> data = lb.get_data();
    Data<double> df(data, false, false);
    vector<string>features = df.getFeatures();
    df.setIndex(features[3]);
    VectorXd y = df.getMatrix().col(3);
    df.removeColumn(3);
    MatrixXd x = df.getMatrix();
    Decision_Tree dt;
    dt.set_params({ {"min_sample_split",85},{"ccp_alpha",0.006}, {"min_impurity_decrease",0.01} });
    dt.get_params();
    clock_t a = clock();
    dt.fit(x, y, lb.get_classes(), df.getFeatures());
    clock_t b = clock();
    cout << b - a << "ms" << endl;
    //vector<pair<double, double>> path = dt.cost_complexity_pruning_path();
    //for (const auto p : path) {
    //    cout << "ccp_alpha : " << p.first;
    //    cout << ", impurity : " << p.second << endl;
    //}
    Node r = dt.get_root();
    dt.export_tree(&r);
    //VectorXd y_pred = dt.predict(x, df.getFeatures());
    //double score = dt.score(x, y, df.getFeatures());
    //cout << y_pred.transpose() << endl;
    //cout << score << endl;

    return 0;
}*/