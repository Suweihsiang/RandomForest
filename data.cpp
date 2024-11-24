#include"data.h"
//���ŧi���ҪO������
template Data<double>;
template Data<float>;
template Data<int>;

//�}�l��@�禡
template<typename T>
Data<T>::Data() {}

template<typename T>
Data<T>::Data(unordered_map<string, vector<string>>m, bool IndexFirst, bool sortIndex) :rows((m.begin()->second).size()), hasIndexRows(IndexFirst) {
	auto it = m.begin();
	columns = hasIndexRows ? m.size() - 1 : m.size();
	mat.conservativeResize(rows, columns);
	if (hasIndexRows) {
		index_name = it->first;
		indexes = it->second;
		it++;
	}
	setMatrix(it);
	if(IndexFirst && sortIndex){sortbyIndex(true);}
}

template<typename T>
Data<T>::Data(string path,bool IndexFirst,bool isThousand,bool sortIndex) {
	data_name = path.substr(path.rfind('/') + 1, path.rfind('.') - path.rfind('/') - 1);
	ifstream ifs(path);
	string data;
	hasIndexRows = IndexFirst;
	getline(ifs, data);
	if (hasIndexRows) {
		features = split(data, ',');
		index_name = features[0];
		features.erase(features.begin());
	}
	else {
		features = split(data, ',');
	}
	columns = features.size();
	mat.conservativeResize(0, columns);
	int r = 0, c = 0;
	while (getline(ifs, data)) {
		vector<string>cols;
		mat.conservativeResize(mat.rows() + 1, columns);
		if (isThousand) {
			if (hasIndexRows) {
				indexes.push_back(data.substr(0,data.find(',')));
			}
			else {
				istringstream is(data.substr(0, data.find(',')));
				T T_date;
				is >> T_date;
				mat(r, c) = T_date;
				c++;
			}
			cols = camma_remove(data);
		}
		else {
			cols = split(data, ',');
			if (hasIndexRows) {
				indexes.push_back(cols[0]);
				cols.erase(cols.begin());
			}
		}
		setMatrix(r, c, cols);
		r++;
		c = 0;
	}
	rows = mat.rows();
	if(IndexFirst && sortIndex){sortbyIndex(true);}
}

template<typename T>
Data<T>::~Data() {}

template<typename T>
Data<T> Data<T>::operator[](vector<string>fts) {
	Data<T> d_fts;
	d_fts.data_name = data_name;
	d_fts.hasIndexRows = hasIndexRows;
	if (hasIndexRows) {
		d_fts.index_name = index_name;
		d_fts.indexes = indexes;
	}
	int c = 0;
	for (const auto &ft : fts) {
		auto it = find(features.begin(), features.end(), ft);
		if (it != features.end()) {
			d_fts.mat.conservativeResize(rows, c + 1);
			d_fts.features.push_back(ft);
			size_t dist = distance(features.begin(), it);
			d_fts.mat.col(c) = mat.col(dist);
			c++;
		}
	}
	d_fts.rows = rows;
	d_fts, columns = d_fts.mat.cols();
	return d_fts;
}

template<typename T>
void Data<T>::setMatrix(unordered_map<string, vector<string>>::iterator it) {
	for (int c = 0; c < columns; c++) {
		features.push_back(it->first);
		for (int r = 0; r < rows; r++) {
			string s_val = it->second[r];
			if (s_val == "") {
				nullpos.push_back({ r,c });
				c++;
				continue;
			}
			istringstream iss(s_val);
			T T_val;
			iss >> T_val;
			mat(r, c) = T_val;
		}
		it++;
	}
}

template<typename T>
void Data<T>::setMatrix(int r, int c, vector<string>cols) {
	for (const auto &col : cols) {
		if (col == "") {
			nullpos.push_back({ r,c });
			c++;
			continue;
		}
		istringstream iss(col);
		T T_val;
		iss >> T_val;
		mat(r, c) = T_val;
		c++;
	}
	while (c < columns) {
		nullpos.push_back({ r,c });
		c++;
	}
}

template<typename T>
vector<string> Data<T>::split(string s, char dec) {
	istringstream iss(s);
	string subs;
	vector<string>sv;
	while (getline(iss, subs, dec)) {
		sv.push_back(subs);
	}
	return sv;
}

template<typename T>
void Data<T>::setDataname(string _data_name) {
	data_name = _data_name;
}

template<typename T>
string Data<T>::getDataname() const {
	return data_name;
}

template<typename T>
int Data<T>::getRows() const {
	return rows;
}

template<typename T>
int Data<T>::getColumns() const {
	return columns;
}

template<typename T>
Matrix<T, Dynamic, Dynamic> Data<T>::getMatrix() const {
	return mat;
}

template<typename T>
vector<string> Data<T>::getFeatures() const {
	return features;
}

template<typename T>
vector<string> Data<T>::getIndexs() const {
	return indexes;
}

template<typename T>
vector<string> Data<T>::camma_remove(string data) {
	vector<string>numeric_data;
	size_t begin, end;
	do {
		begin = data.find('"');
		end = data.substr(begin + 1).find('"');
		string nd = data.substr(begin+1, end);
		do {
			nd.erase(nd.find(','),1);
		} while (nd.find(',') != nd.npos);
		numeric_data.push_back(nd);
		data = data.substr(begin + end + 2);
	} while (data.find('"') != data.npos);
	return numeric_data;
}

template<typename T>
void Data<T>::setIndex(string index) {
	auto it = find(features.begin(), features.end(), index);
	if (it != features.end()) {
		size_t dist = distance(features.begin(), it);
		if (hasIndexRows) {
			mat.conservativeResize(rows, columns + 1);
			features.push_back(index_name);
			for (int i = 0; i < rows; i++) {
				istringstream iss(indexes[i]);
				T T_val;
				iss >> T_val;
				mat(i, columns) = T_val;
			}
			columns++;
			vector<T> new_indexes(mat.col(dist).data(),mat.col(dist).data() + mat.rows());
			removeColumns({ index });
			index_name = index;
			indexes.clear();
			for (const auto &idx : new_indexes) {
				indexes.push_back(to_string((int)idx));
			}
		}
		else {
			hasIndexRows = true;
			vector<T> new_indexes(mat.col(dist).data(),mat.col(dist).data() + mat.rows());
			removeColumns({ index });
			index_name = index;
			for (const auto &idx : new_indexes) {
				indexes.push_back(to_string((int)idx));
			}
		}
	}
}

template<typename T>
void Data<T>::removeRow(int RowToRemove) {
	indexes.erase(indexes.begin() + RowToRemove);
	mat.block(RowToRemove, 0, rows - RowToRemove - 1, columns) = mat.block(RowToRemove + 1, 0, rows - RowToRemove - 1, columns);
	mat.conservativeResize(rows - 1, columns);
	rows--;
}

template<typename T>
void Data<T>::removeRow(string idx) {
	auto it = find(indexes.begin(), indexes.end(), idx);
	if (it != indexes.end()) {
		auto next = indexes.erase(it);
		int dist = distance(indexes.begin(), next);
		mat.block(dist, 0, mat.rows() - dist - 1, mat.cols()) = mat.block(dist + 1, 0, mat.rows() - dist - 1, mat.cols());
		mat.conservativeResize(mat.rows() - 1, mat.cols());
		rows--;
	}
}

template<typename T>
void Data<T>::removeRows(vector<string>idxs) {
	for (const auto &idx : idxs) {
		removeRow(idx);
	}
}

template<typename T>
void Data<T>::merge(Data df2) {
	vector<string> features2 = df2.getFeatures();
	vector<string> indexes2 = df2.getIndexs();
	for (const auto &feature : features2) {
		if (find(features.begin(),features.end(),feature) == features.end()) {
			features.push_back(feature);
		}
		else {
			features.push_back(feature + "_" + df2.data_name);
		}
	}
	for (const auto &idx : df2.getIndexs()) {
		if (find(indexes.begin(),indexes.end(),idx) == indexes.end()) {
			df2.removeRow(idx);
		}
	}
	for (const auto &idx : getIndexs()) {
		if (find(indexes2.begin(), indexes2.end(),idx) == indexes2.end()) {
			removeRow(idx);
		}
	}
	columns = features.size();
	Matrix<T, Dynamic, Dynamic> mat2 = df2.getMatrix();
	mat.conservativeResize(mat.rows(), mat.cols() + mat2.cols());
	mat.block(0, mat.cols() - mat2.cols(), mat2.rows(), mat2.cols()) = mat2;
}

template<typename T>
void Data<T>::addRows(vector<vector<string>>addrows) {
	size_t r = addrows.size();
	size_t c = addrows[0].size();
	mat.conservativeResize(mat.rows() + r, mat.cols());
	if (hasIndexRows) {
		for (int i = 0; i < r; i++) {
			indexes.push_back(addrows[i][0]);
			for (int j = 1; j < c; j++) {
				istringstream iss(addrows[i][j]);
				T T_val;
				iss >> T_val;
				mat(mat.rows() - r + i, j - 1) = T_val;
			}
		}
	}
	else {
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				istringstream iss(addrows[i][j]);
				T T_val;
				iss >> T_val;
				mat(mat.rows() - r + i, j) = T_val;
			}
		}
	}
	rows += r;
}

template<typename T>
void Data<T>::addColumns(unordered_map<string, vector<string>>ms) {
	auto it = ms.begin();
	for (int c = 0; c < ms.size(); c++) {
		mat.conservativeResize(rows, columns + 1);
		features.push_back(it->first);
		for (int r = 0; r < rows; r++) {
			string s_val = it->second[r];
			istringstream iss(s_val);
			T T_val;
			iss >> T_val;
			mat(r, columns) = T_val;
		}
		it++;
		columns++;
	}
}

template<typename T>
void Data<T>::removeColumn(int ColToRemove) {
	features.erase(features.begin() + ColToRemove);
	mat.block(0, ColToRemove, rows, columns - ColToRemove - 1) = mat.block(0, ColToRemove + 1, rows, columns - ColToRemove - 1);
	mat.conservativeResize(rows, columns - 1);
	columns--;
}

template<typename T>
void Data<T>::removeColumns(vector<string>fts) {
	for (const auto &ft : fts) {
		auto it = find(features.begin(), features.end(), ft);
		if (it != features.end()) {
			auto next = features.erase(it);
			int dist = distance(features.begin(), next);
			mat.block(0, dist, rows, columns - dist - 1) = mat.block(0, dist+1, rows, columns - dist - 1);
			mat.conservativeResize(mat.rows(), mat.cols() - 1);
			columns--;
		}
	}
}

template<typename T>
void Data<T>::renameColumns(unordered_map<string, string>names) {
	for (const auto& name : names) {
		vector<string>::iterator it = find(features.begin(), features.end(), name.first);
		if (it != features.end()) {
			int idx = distance(features.begin(), it);
			features[idx] = name.second;
		}
	}
}

template<typename T>
void Data<T>::sortbyIndex(bool ascending) {
	if (!hasIndexRows) {
		return;
	}
	vector<pair<string, VectorX<T>>>vec;
	for (int i = 0; i < rows; i++) {
		vec.push_back({ indexes[i],mat.row(i) });
	}
	if (ascending) {
		sort(vec.begin(), vec.end(), [](pair<string, VectorX<T>>& v1, pair<string, VectorX<T>>& v2) {return v1.first < v2.first; });
	}
	else {
		sort(vec.begin(), vec.end(), [](pair<string, VectorX<T>>& v1, pair<string, VectorX<T>>& v2) {return v1.first > v2.first; });
	}
	for (int i = 0; i < rows; i++) {
		indexes[i] = vec[i].first;
		mat.row(i) = vec[i].second;
	}
}

template<typename T>
void Data<T>::sortby(string feature,bool ascending) {
	if (feature == index_name) {
		sortbyIndex(ascending);
	}
	else {
		auto it = find(features.begin(), features.end(), feature);
		int dist = distance(features.begin(), it);
		vector<pair<string, VectorX<T>>>vec;
		for (int i = 0; i < rows; i++) {
			hasIndexRows?vec.push_back({ indexes[i],mat.row(i) }): vec.push_back({ to_string(i+1),mat.row(i)});
		}
		if (ascending) {
			sort(vec.begin(), vec.end(), [&dist](pair<string, VectorX<T>>& v1, pair<string, VectorX<T>>& v2) {return v1.second[dist] < v2.second[dist]; });
		}
		else {
			sort(vec.begin(), vec.end(), [&dist](pair<string, VectorX<T>>& v1, pair<string, VectorX<T>>& v2) {return v1.second[dist] > v2.second[dist]; });
		}
		for (int i = 0; i < rows; i++) {
			if (hasIndexRows) { indexes[i] = vec[i].first; }
			mat.row(i) = vec[i].second;
		}
	}
}

template<typename T>
Data<T> Data<T>::groupby(string feature, string operate) {
	setIndex(feature);
	sortbyIndex(true);
	set<string> idx_set(indexes.begin(), indexes.end());
	MatrixX<T> mat_res(0, columns);
	int pos = 0;
	for (const auto &idx : idx_set) {
		RowVectorX<T> sum = RowVectorX<T>::Zero(columns);
		int count = 0;
		for (int i = pos; i < rows; i++) {
			if (indexes[i] == idx) {
				sum += mat.row(i);
				count++;
				pos++;
			}
			else {
				break;
			}
		}
		if (operate == "sum") { ; }
		else if (operate == "mean") { sum /= count; }
		mat_res.conservativeResize(mat_res.rows() + 1, columns);
		mat_res.row(mat_res.rows()-1) = sum;
	}
	vector<string>idx_group(idx_set.begin(), idx_set.end());
	vector<vector<string>>res_group;
	for (int c = 0; c < columns; c++) {
		vector<string>mat_to_str;
		for (int r = 0;r < mat_res.rows(); r++) {
			mat_to_str.push_back(to_string(mat_res(r, c)));
		}
		res_group.push_back(mat_to_str);
	}
	unordered_map<string, vector<string>>mav;
	mav[index_name] = idx_group;
	for (int i = 0; i < columns; i++) {
		mav[features[i]] = res_group[i];
	}
	Data<T> df_group(mav, true);
	return df_group;
}

template<typename T>
void Data<T>::dropna(int axis) {
	sort(nullpos.begin(), nullpos.end(), [&axis](vector<int>& v1, vector<int>& v2) {return v1[axis] < v2[axis]; });
	int removeCount = 0;
	int pre_remove = -1;
	for (const auto &nan : nullpos) {
		if (nan[axis] != pre_remove) {
			(axis == 0) ? removeRow(nan[axis] - removeCount) : removeColumn(nan[axis] - removeCount);
			removeCount++;
			pre_remove = nan[axis];
		}
	}

}

template<typename T>
void Data<T>::print() {
	if (hasIndexRows) {
		cout << index_name << "\t" << " ";
	}
	for (const auto &feature : features) {
		cout << feature << "\t"<<" ";
	}
	cout << endl;
	for (int i = 0; i < rows; i++) {
		if (hasIndexRows) {
			cout << indexes[i] << " ";
		}
		cout << mat.row(i) << endl;
	}
}

template<typename T>
void Data<T>::to_csv(string path) {
	ofstream dataFile;
	dataFile.open(path, ios::out | ios::trunc);
	if (hasIndexRows) {
		dataFile << index_name << " ";
	}
	for (const auto &feature : features) {
		dataFile << feature << " ";
	}
	dataFile << endl;
	for (int r = 0; r < rows; r++) {
		if (hasIndexRows) {
			dataFile << indexes[r] << " ";
		}
		dataFile<< mat.row(r);
		dataFile << endl;
	}
	dataFile.close();
}