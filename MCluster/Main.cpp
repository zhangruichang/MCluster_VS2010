//parameter: 
//1. cluster number: int getClassNum(const char* file)
//2. write obsolute path of inputfile into "filelist_LDA.txt"
//3. DIM modify: SKWIC 256 LDA+SKWIC 20 
//4. input format:
//emptyline
//>r1 |stringone|stringtwo|stringthree	0	0	...
//3times   strtok(NULL, "|");

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include "skwic.h"
#include <cctype>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <direct.h>
using namespace std;
const int DIM = 256;

struct DataItem {
	string id;
	string famIdx;
	double data[DIM];
};

vector<DataItem> dataSet;

void readDataSet(string inputfile)
{
	dataSet.clear();
	//cout<<dataSet.max_size()<<endl;
	ifstream fin(inputfile);
	if (!fin)
	{
		cout << "Cannot open the input file" << endl;
		exit(1);
	}
	DataItem item;
	int count = 1;
	string line;
	getline(fin, line);
	while (getline(fin, line))
	{
		if (count%1000000==0) 
			cout<<"Reading reads number"<<count<<endl;
		istringstream istr(line);
		getline(istr, item.famIdx, '\t');

		string SplitWord;
		int datai=0;
		while(getline(istr, SplitWord, '\t'))
			item.data[datai++]=stof(SplitWord);
		dataSet.push_back(item);
		count++;
	}
	fin.close();
}


inline void writeClusterResult(const int *cls, string outfile)
{
	ofstream fout(outfile);
	for (size_t i = 0; i < dataSet.size(); i++)
		fout <<  dataSet[i].famIdx << '\t' << cls[i] << endl;
	fout << endl;
	fout.close();
}

//TetraFreq
int getIndex(char s)
{
	switch(s){//A+T=G+C=4
		case 'A':
			return 0;
		case 'G':
			return 1;
		case 'C':
			return 2;
		case 'T':
			return 3;
	}
}
string getTetra(int n, int length)
{
	char s[4]={'A','G', 'C', 'T'};
	int *a=new int[length];
	for (int i = 0; i < length; i++) {
		a[i]=n%4;
		n=n/4;
	}
	string result;
	for (int i = 0; i < length; i++) {
		result.push_back(s[a[i]]);
	}
	delete []a;
	return result;
}
float* getFreqs(string seq, int length)
{

	transform(seq.begin(), seq.end(), seq.begin(),::toupper);
	int dimension=(int) pow(float(4), length);
	float* freqs=new float[dimension];
	for(int i=0;i<dimension;i++)
		freqs[i]=0;

	int *a=new int[length];
	int *b=new int[length];
	int index1;
	int index2;
	bool hasN=false;
	string charset="ATCG";
	for (int i = 0; i < seq.length()-length+1; i++) {//for each k-mer
		index1=0;//dna index i
		index2=0;//reverse dna index i
		//a=new int[length];//k-mer index seq
		//b=new int[length];//reverse k-mer index seq

		
		for (int j = 0; j < length; j++) {//for each nucleotide of k-mer
			if (charset.find(seq[i+j])==-1){hasN=true; break;}//if seq has N
			a[j]=getIndex(seq[i+j]);
			index1+=(int)pow(double(4), j)*a[j];//AAAA 0 GAAA 1 CAAA 2 TAAA 3
			// AGCT 0 0123
		}
		if (hasN==true){continue;}//AACT reverse is TTGA
		for (int j = 0; j < length; j++) {
			b[length-1-j]=3-getIndex(seq[i+j]);
			index2+=(int)pow(double(4), length-1-j)*b[length-1-j];//TTTT 0 CTTT 1 GTTT 2 ATTT 3
			//TCGA 01234
		}
		freqs[index1]+=1;// coresponding fregs count++
		freqs[index2]+=1;
	}
	for (int i = 0; i < dimension; i++) {//count to frequency(<=1)
		freqs[i]=(float)freqs[i]/2/(seq.length()-length+1);//freqs /(2*k-mer number)
	}
	delete []b;
	delete []a;
	return freqs;
}
void prepareseq(string inputfile,string outfile,int length)//write k-mer feature to ***.freq files
{
		ifstream fin(inputfile);
		ofstream fout(outfile);
		fout<<"feature\t";
		int FeatureNum=(int)pow(4.0, length);
		for (int i = 0; i < FeatureNum; i++)
			fout<<getTetra(i,length)<<"\t";
		fout<<endl;
		string title, line, seq;
		int dimension=(int)pow(4.0,length);
		float* freqs=new float[dimension];
		int cnt=0;
		while(getline(fin, line))
		{
			if(line.size()>0 && line[0]=='>')
			{
				if(title=="")
				{
					title=line;
					continue;
				}
				freqs=getFreqs(seq,length);
				fout<<title<<'\t';
				for (int i = 0; i < dimension; i++)
					fout<<freqs[i]<<'\t';
				fout<<endl;
				title=line;
				seq="";
			}
			else
				seq+=line;
		}
		if(seq!="")
		{
			freqs=getFreqs(seq,length);
			fout<<title<<'\t';
			for (int i = 0; i < dimension; i++)
				fout<<freqs[i]<<'\t';
			fout<<endl;
		}
		delete []freqs;
		fout.close();
		fin.close();
}

//
int main(int argc, char** argv)
{
	
	if(argc<=1)
	{
		printf("usage : MCluster.exe [input fasta file name] [cluster number]\n");
		return 0;
	}
	else if(argc<=2)
	{
		printf("parameters are too few!\n");
		return 0;
	}
	
	clock_t start, finish;
	double duration;
	start = clock();
	

	int kmerLength=4;
	

	string Dir=".//"; 
	string InputPath=Dir+argv[1];
	
	string OutputFreq=InputPath+".freq";//argv[3];
	cout<<"Getting K-mer frequencies..."<<endl;
	prepareseq(InputPath,OutputFreq,kmerLength);
	cout<<"Getting K-mer frequencies done!"<<endl;

	cout<<"Reading K-mer frequencies..."<<endl;
	readDataSet(OutputFreq);
	cout<<"Reading K-mer frequencies done!"<<endl;
	string outfile = OutputFreq.append(".mahatten.out");
	double **data = new double*[dataSet.size()];
	for (size_t i = 0; i < dataSet.size(); i++)
		data[i] = dataSet[i].data;
	cout<<"Clustering.."<<endl;
	Skwic skwic;
	skwic.setDistMeasure(Skwic::CITYBLOCK);
	skwic.setData(data, DIM, dataSet.size(), 1, 0.01);
	double obj;
	int num; 
	num= atoi(argv[2]);

	int *cls = skwic.clustering(num, &obj);
	cout<<"Clustering Done!"<<endl<<"Saving Clustering Result.."<<endl;
	writeClusterResult(cls, outfile);
	
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;   
	printf("Total time: %f seconds\n", duration);
	return 0;
}
