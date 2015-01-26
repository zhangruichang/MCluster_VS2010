
#ifndef _SKWIC_H_
#define _SKWIC_H_

#include <fstream>
#include <vector>

class Skwic {
	std::ofstream fout;

	int n_cl;
	double **data;
	int n_d;
	int dim;
	double k_delta;
	double stop_criterion;

	int distType;

	int *cls;
	int *pre_cls;
	double **center;
	double **pre_center;
	double **weight;
	double **pre_weight;
	double *delta;

	int *clusterSize;
	std::vector<int> *clsDataIdx;

	double *wce;

	int abnormal_weight_count;

	void initCenter(void);
	void initWeight(void);
	void initDelta(void);
	void updateWeight(void);
	void partition(void);
	double updateCenter(void);
	void handleEmptyCluster(int clsIdx);
	void updateDelta(void);
	void onlineUpdate(void);
	void tryUpdateWeight(double *weight, int clsIdx, int ignoredDataIdx);
	void getClsDataIdx(int clsIdx);
	void normalize(double *);

	inline double totalWce();
	inline double objFun();
	inline void writeCls();
	inline void writeCenter();
	inline void writeWeight();
	inline void writeDelta();
public:

	enum DistType {CITYBLOCK, EUCLIDEAN, COSINE};

	Skwic();
	~Skwic();

	void setData(double **data, int dim, int num, double k_delta, double stop_criterion);
	void setDistMeasure(int distType);
	int *clustering(int n_cl, double *obj);
	void clear();
};

#endif