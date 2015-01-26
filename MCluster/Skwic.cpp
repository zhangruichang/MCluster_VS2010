// implemented according to H. Frigui, O. Nasraoui; "Simultaneous Clustering and Dynamic Keyword Weighting for Text Documents"
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include "skwic.h"

#define DISPLAY
//#define TRACE_OBJ

using namespace std;

Skwic::Skwic()
{
	data = NULL;
	n_cl = n_d = dim = 0;
	stop_criterion = 0;
	abnormal_weight_count = 0;
	distType = DistType::COSINE;
	wce = NULL;
}

Skwic::~Skwic()
{
	if(data!=NULL)
	{
		delete[] center[0];
		delete[] center;
		delete[] pre_center[0];
		delete[] pre_center;
		delete[] weight[0];
		delete[] weight;
		delete[] pre_weight[0];
		delete[] pre_weight;
		delete[] clusterSize;
		delete[] clsDataIdx;
		delete[] wce;
		fout.close();
	}
}

void Skwic::clear()
{
	if(data!=NULL)
	{
		delete[] center[0];
		delete[] center;
		delete[] pre_center[0];
		delete[] pre_center;
		delete[] weight[0];
		delete[] weight;
		delete[] pre_weight[0];
		delete[] pre_weight;
		delete[] clusterSize;
		delete[] clsDataIdx;
		delete[] wce;
		fout.close();
	}
}

void Skwic::setDistMeasure(int distType)
{
	this->distType = distType;
}

void Skwic::setData(double **data, int dim, int num, double k_delta, double stop_criterion)
{
	int i;

	this->data = data;
	this->dim = dim;
	this->n_d = num;
	this->k_delta = k_delta;
	this->stop_criterion = stop_criterion;

	this->cls = new int[num];
	this->pre_cls = new int[num];
	this->wce = new double[num];

	//if(distType==DistType::COSINE)
	{
		// normalize all vectors
		for(i=0; i<n_d; i++)
		{
			normalize(this->data[i]);
		}
	}
}

void Skwic::initCenter(void)
{
	int i, j, r;
	cout<<"1";
	this->center = new double*[n_cl];
	this->center[0] = new double[dim*n_cl];
	this->pre_center = new double*[n_cl];
	this->pre_center[0] = new double[dim*n_cl];
	this->weight = new double*[n_cl];
	this->weight[0] = new double[dim*n_cl];
	this->pre_weight = new double *[n_cl];
	this->pre_weight[0] = new double[dim*n_cl];
	for(i=1; i<n_cl; i++)
	{
		this->center[i] = this->center[0]+dim*i;
		this->pre_center[i] = this->pre_center[0]+dim*i;
		this->weight[i] = this->weight[0]+dim*i;
		this->pre_weight[i] = this->pre_weight[0]+dim*i;
	}
	this->delta = new double[n_cl];
	this->clusterSize = new int[n_cl];
	this->clsDataIdx = new vector<int>[n_cl];
	cout<<"2";
	// initialize the centers
	// randomly select n_cl centers
	char *flag = new char[n_d];
	memset(flag, 0, n_d*sizeof(char));
	srand((unsigned int)time(NULL));
	for(i=0; i<n_cl; i++)
	{
		r = (int)(n_d*(double)rand()/(double(RAND_MAX)+1));
		if(flag[r]==0)
		{
			//cout<<r;
			for(j=0; j<dim; j++)
			{
				center[i][j] = data[r][j];
				//cout<<' '<<center[i][j];
			}
			//cout<<endl;
			flag[r] = 1;
		}
		else
			i--;
	}
	cout<<"3";
	delete[] flag;
}

void Skwic::initWeight(void)
{
	int i, j;
	// initialize the weights
	for(i=0; i<n_cl; i++)
		for(j=0; j<dim; j++)
			weight[i][j] = 1.0/dim;
}

void Skwic::initDelta(void)
{
	int i, j, k;
	double *ss_v = new double[n_cl];
	for(i=0; i<n_cl; i++)
	{
		ss_v[i] = 0;
		for(k=0; k<dim; k++)
		{
			ss_v[i] += weight[i][k]*weight[i][k];	// the sum of square of the weights (used for updating the delta)
		}
	}
	for(i=0; i<n_cl; i++)
		delta[i] = 0;
	for(j=0; j<n_d; j++)
	{
		for(k=0; k<dim; k++)
		{
			if(distType==DistType::COSINE)
				delta[cls[j]] += weight[cls[j]][k]*(1.0/dim-data[j][k]*center[cls[j]][k]);
			else if(distType==DistType::CITYBLOCK)
				delta[cls[j]] += weight[cls[j]][k]*fabs(data[j][k]-center[cls[j]][k]);
			else
				delta[cls[j]] += weight[cls[j]][k]*(data[j][k]-center[cls[j]][k])*(data[j][k]-center[cls[j]][k]);
		}
	}
	for(i=0; i<n_cl; i++)
	{
		// delta[i] = 1;
		delta[i] = k_delta*delta[i]/ss_v[i];
	}
	delete[] ss_v;
}

// equation (1.8) and (1.9) in the article
void Skwic::updateWeight()
{
	int i, j, k;

	for(i=0; i<n_cl; i++)
	{
		for(k=0; k<dim; k++)
		{
			pre_weight[i][k] = weight[i][k];	// store the weight before updating (used for updating the delta)
			weight[i][k] = 1.0/dim;
		}
	}

	// update weights
	double *d = new double[dim];	// store the Dwc for this data vector
	double sum_d;
	for(j=0; j<n_d; j++)
	{
		// TODO: What if the delta of some cluster equals zero?
		if(delta[cls[j]]==0)
			//continue;					// when using cosine distances, this has better results
			delta[cls[j]] = 0.001;		// when using cityblock distances, this has better results
		sum_d = 0;
		for(k=0; k<dim; k++)
		{
			if(distType==DistType::COSINE)
				d[k] = 1.0/dim-data[j][k]*center[cls[j]][k];
			else if(distType==DistType::CITYBLOCK)
				d[k] = fabs(data[j][k]-center[cls[j]][k]);
			else
				d[k] = (data[j][k]-center[cls[j]][k])*(data[j][k]-center[cls[j]][k]);

			sum_d += d[k];
		}

		for(k=0; k<dim; k++)
		{
			weight[cls[j]][k] += 0.5/delta[cls[j]]*(sum_d/dim-d[k]);
		}
	}
	delete[] d;
	// modify negative weights (1.10)
	double min_v;
	double sum_v;
	for(i=0; i<n_cl; i++)
	{
		min_v = 0;
		for(k=0; k<dim; k++)
		{
			if(weight[i][k]>=0)
				continue;
			if(weight[i][k]<min_v)
				min_v = weight[i][k];
		}
		if(min_v<0)
		{
			abnormal_weight_count++;
			sum_v = 0;
			for(k=0; k<dim; k++)
			{
				if(weight[i][k]<0)
					weight[i][k] -= min_v;
				sum_v += weight[i][k];
			}
			if(sum_v!=0)
				for(k=0; k<dim; k++)
					weight[i][k] /= sum_v;
		}
	}
}

void Skwic::partition()
{
	int i, j, k;
	double dist;
	double minDist = 0;
	memset(clusterSize, 0, sizeof(int)*n_cl);
	for(j=0; j<n_d; j++)
	{
		pre_cls[j] = cls[j];

		for(i=0; i<n_cl; i++)
		{
			dist = 0;
			for(k=0; k<dim; k++)
			{
				if(distType==DistType::COSINE)
					dist += weight[i][k]*(1.0/dim-data[j][k]*center[i][k]);
				else if(distType==DistType::CITYBLOCK)
					dist += weight[i][k]*fabs(data[j][k]-center[i][k]);
				else
					dist += weight[i][k]*(data[j][k]-center[i][k])*(data[j][k]-center[i][k]);
			}
			if(minDist>dist || i==0)
			{
				minDist = dist;
				cls[j] = i;
			}
		}
		clusterSize[cls[j]]++;
	}
}

// equation (1.12) in the article
double Skwic::updateCenter()
{
	int i, j, k;
	for(i=0; i<n_cl; i++)
	{
		for(k=0; k<dim; k++)
		{
			pre_center[i][k] = center[i][k];
			center[i][k] = 0;
		}
	}
	for(j=0; j<n_d; j++)
	{
		for(k=0; k<dim; k++)
		{
			if(weight[cls[j]][k]!=0)
				center[cls[j]][k] += data[j][k];
		}
	}
	for(i=0; i<n_cl; i++)
		if(clusterSize[i]==0)
			handleEmptyCluster(i);
	for(i=0; i<n_cl; i++)
	{
		if(clusterSize[i]==0)
			continue;
		for(k=0; k<dim; k++)
		{
			center[i][k] /= clusterSize[i];
		}
		if(distType==DistType::COSINE)
			normalize(center[i]);
	}

	//
	double dist;
	double minDist;
	double avgDist = 0;
	minDist = 0;
	for(i=0; i<n_cl; i++)
	{
		dist = 0;
		for(k=0; k<dim; k++)
		{
			if(distType==DistType::COSINE)
				dist += /*weight[i][k]**/(1.0/dim-center[i][k]*pre_center[i][k]);
			else if(distType==DistType::CITYBLOCK)
				dist += /*weight[i][k]**/fabs(center[i][k]-pre_center[i][k]);
			else
				dist += /*weight[i][k]**/(center[i][k]-pre_center[i][k])*(center[i][k]-pre_center[i][k]);
		}
		if(minDist>dist || i==0)
			minDist = dist;
		avgDist += dist;
	}
	//return minDist;
	avgDist /= n_cl;
	return avgDist;
}

// take the farthest data point as the new center for the empty cluster
void Skwic::handleEmptyCluster(int clsIdx)
{
	int i, k;
	int farthest;
	double dist;
	double maxDist;
	maxDist = 0;
	farthest = -1;
	/*for(i=0; i<n_d; i++)
	{
		dist = 0;

		for(k=0; k<dim; k++)
		{
			if(distType==DistType::CITYBLOCK)
				dist += weight[cls[i]][k]*fabs(data[i][k]-center[cls[i]][k]);
			else if(distType==DistType::COSINE)
				dist += weight[cls[i]][k]*(1.0/dim-data[i][k]*center[cls[i]][k]);
			else
				dist += weight[cls[i]][k]*(data[i][k]-center[cls[i]][k])*(data[i][k]-center[cls[i]][k]);
		}

		if(dist>maxDist)
		{
			maxDist = dist;
			farthest = i;
		}
	}*/

	if(farthest!=-1)
	{
		for(k=0; k<dim; k++)
		{
			if(weight[cls[farthest]][k]!=0)
				center[cls[farthest]][k] -= data[farthest][k];
			center[clsIdx][k] = data[farthest][k];
		}
		clusterSize[cls[farthest]]--;
		cls[farthest] = clsIdx;
		clusterSize[clsIdx]++;
	}
	else
		for(k=0; k<dim; k++)
			center[clsIdx][k] = pre_center[clsIdx][k];
}

void Skwic::updateDelta(void)
{
	int i, j, k;
	double *ss_v = new double[n_cl];
	for(i=0; i<n_cl; i++)
	{
		ss_v[i] = 0;
		for(k=0; k<dim; k++)
		{
			ss_v[i] += pre_weight[i][k]*pre_weight[i][k];	// the sum of square of the weights (used for updating the delta)
		}
	}

	// update delta
	for(i=0; i<n_cl; i++)
		delta[i] = 0;
	for(j=0; j<n_d; j++)
	{
		for(k=0; k<dim; k++)
		{
			if(distType==DistType::COSINE)
				delta[pre_cls[j]] += pre_weight[pre_cls[j]][k]*(1.0/dim-data[j][k]*pre_center[pre_cls[j]][k]);
			else if(distType==DistType::CITYBLOCK)
				delta[pre_cls[j]] += pre_weight[pre_cls[j]][k]*fabs(data[j][k]-pre_center[pre_cls[j]][k]);
			else
				delta[pre_cls[j]] += pre_weight[pre_cls[j]][k]*(data[j][k]-pre_center[pre_cls[j]][k])*(data[j][k]-pre_center[pre_cls[j]][k]);
		}
	}
	for(i=0; i<n_cl; i++)
	{
		delta[i] = k_delta*delta[i]/ss_v[i];
	}
	delete[] ss_v;
}



void Skwic::getClsDataIdx(int clsIdx)
{
	int i;
	if(clsIdx<0)
	{
		for(i=0; i<n_cl; i++)
		{
			clsDataIdx[i].clear();
		}
		for(i=0; i<n_d; i++)
		{
			clsDataIdx[cls[i]].push_back(i);
		}
	}
	else
	{
		clsDataIdx[clsIdx].clear();
		for(i=0; i<n_d; i++)
		{
			if(cls[i]==clsIdx)
				clsDataIdx[clsIdx].push_back(i);
		}
	}
}

void Skwic::normalize(double *vec)
{
	int i;
	double sum;
	sum = 0;
	for(i=0; i<dim; i++)
	{
		sum += vec[i]*vec[i];
	}
	if(sum==0)
		return;
	for(i=0; i<dim; i++)
	{
		vec[i] /= sqrt(sum);
	}
}

inline double Skwic::totalWce()
{
	int j, k;
	double sum = 0;
	double dist;
	for(j=0; j<n_d; j++)
	{
		dist = 0;
		for(k=0; k<dim; k++)
		{
			if(distType==DistType::CITYBLOCK)
				dist += weight[cls[j]][k]*fabs(data[j][k]-center[cls[j]][k]);
			else if(distType==DistType::COSINE)
				dist += weight[cls[j]][k]*(1.0/dim-data[j][k]*center[cls[j]][k]);
			else
				dist += weight[cls[j]][k]*(data[j][k]-center[cls[j]][k])*(data[j][k]-center[cls[j]][k]);
		}
		wce[j] = dist;
		sum += dist;
	}
	return sum;
}

inline double Skwic::objFun()
{
	double sum = 0;
	int i, k;
	for(i=0; i<n_cl; i++)
		for(k=0; k<dim; k++)
			sum += delta[i]*weight[i][k]*weight[i][k];
	return sum+totalWce();
}

inline void Skwic::writeCls()
{
	int j;
	for(j=0; j<n_d; j++)
		fout<<' '<<cls[j];
	fout<<endl;
}

inline void Skwic::writeCenter()
{
	int i, j;
	double sum;
	for(i=0; i<n_cl; i++)
	{
		fout<<"center "<<i<<" : ";
		sum = 0;
		for(j=0; j<dim; j++)
		{
			sum += center[i][j]*center[i][j];
			fout<<center[i][j]<<' ';
		}
		fout<<endl<<"norm of center: "<<sqrt(sum)<<endl;;
	}
}

inline void Skwic::writeWeight()
{
	int i, j;
	double sum;
	for(i=0; i<n_cl; i++)
	{
		fout<<"weight "<<i<<" : ";
		sum = 0;
		for(j=0; j<dim; j++)
		{
			sum += weight[i][j];
			fout<<weight[i][j]<<' ';
		}
		fout<<endl<<"sum of weights: "<<sum<<endl;
	}
}

inline void Skwic::writeDelta()
{
	int i;
	for(i=0; i<n_cl; i++)
		fout<<"delta "<<i<<" : "<<delta[i]<<endl;
}

int *Skwic::clustering(int n_cl, double *obj)
{
	int i;
	double d;

	if(data==NULL)
	{
		cout<<"no data"<<endl;
		return NULL;
	}

	this->n_cl = n_cl;

	fout.open("skwic_o");
	/*if(!fout)
	{
		cout<<"cannot open the file: skwic_o"<<endl;
		exit(1);
	}*/

#ifdef DISPLAY
	cout<<"initializing..."<<endl;
#endif
	
	initCenter();
	fout<<"//////////initial centers:"<<endl;
	writeCenter();

	initWeight();
	fout<<"//////////initial weights:"<<endl;
	writeWeight();

	partition();
	fout<<"//////////initial partitions:"<<endl;
	writeCls();

	initDelta();
	fout<<"//////////initial delta:"<<endl;
	writeDelta();

#ifdef DISPLAY
#ifdef TRACE_OBJ
	cout<<objFun()<<endl;
#endif
#endif

#ifdef DISPLAY
	cout<<endl;
#endif
	fout<<endl;

	i = 1;
	do{
		fout<<"//////////iteration "<<i<<endl;
#ifdef DISPLAY
		cout<<"//////////iteration "<<i<<endl;
		cout<<"updating the weights..."<<endl;
#endif
		updateWeight();
		fout<<"//////////iteration "<<i<<" weights:"<<endl;
		writeWeight();
#ifdef DISPLAY
#ifdef TRACE_OBJ
		cout<<objFun()<<endl;
#endif
		cout<<"partitioning..."<<endl;
#endif
		partition();
		fout<<"//////////iteration "<<i<<" partitions:"<<endl;
		writeCls();
#ifdef DISPLAY
#ifdef TRACE_OBJ
		cout<<objFun()<<endl;
#endif
		cout<<"updating the centers..."<<endl;
#endif
		d = updateCenter();
		fout<<"//////////iteration "<<i<<" centers:"<<endl;
		writeCenter();

#ifdef DISPLAY
#ifdef TRACE_OBJ
		cout<<objFun()<<endl;
#endif
		cout<<"updating the delta..."<<endl;
#endif
		updateDelta();
		fout<<"//////////iteration "<<i<<" delta:"<<endl;
		writeDelta();

#ifdef DISPLAY
#ifdef TRACE_OBJ
		cout<<objFun()<<endl;
#endif
		cout<<"center changing: "<<d<<endl<<endl;
#endif
		fout<<endl;
		i++;
		if(i>50)
			break;
	}while(d>stop_criterion || i<=20);

	if(obj!=NULL)
		*obj = objFun();

	fout<<abnormal_weight_count<<endl;
#ifdef DISPLAY
	//cout<<i*n_cl<<endl;
	//cout<<abnormal_weight_count<<endl;
	//cout<<*obj<<endl;
#endif
	abnormal_weight_count = 0;

	return cls;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
