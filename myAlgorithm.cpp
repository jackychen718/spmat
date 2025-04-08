#include <mpi.h>
#include "element.h"
#include <cmath>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <vector>
#include <random>
#include <string>

using namespace std;


vector<Element> myMatrixGen(double sparsity,uint64_t startRow,uint64_t endRow,uint64_t startCol,uint64_t endCol)
{
	vector<Element> matrix;
	random_device seed;
	mt19937 rng(seed());
	uniform_real_distribution<double> uniform;
	srand(time(NULL));
	double s = 0;
	for(uint64_t i=startRow;i<endRow;i++)
	{
		for(uint64_t j=startCol;j<endCol;j++)
		{	
			double possibility = uniform(rng);
			if(possibility<sparsity)
			{
				matrix.emplace_back(i,j,1+(uint64_t)rand());
			}
		}
	}
	return matrix;
};


vector<Element> myTranspose(vector<Element>& M,uint64_t startRow,uint64_t endRow,uint64_t startCol,uint64_t endCol)
{
	uint64_t N = M.size();
	uint64_t n = endCol - startCol;
	vector<Element> res(N,Element(0,0,0));
	vector<uint64_t> num(n+1,0);
	vector<uint64_t> index(n+1,0);
	for(uint64_t i=0;i<N;i++)
	{
		num[M[i].col-startCol]++;
	}

	for(uint64_t i=1;i<=n;i++)
	{
		index[i] = index[i-1]+num[i-1];
	}

	for(uint64_t i=0;i<N;i++)
	{

		res[index[M[i].col-startCol]].row = M[i].col;
		res[index[M[i].col-startCol]].col = M[i].row;
		res[index[M[i].col-startCol]].val = M[i].val;
		index[M[i].col-startCol]++;
	}
	return res;
};


void myMultiply(vector<vector<uint64_t>>& tempC,vector<Element>& A, vector<Element>& TB,uint64_t startRow,uint64_t endRow,uint64_t startCol,uint64_t endCol)
{

	vector<Element> res;
	uint64_t lenA = A.size();
	uint64_t lenB = TB.size();
	uint64_t iA = 0;
	uint64_t iB = 0;
	unordered_map<uint64_t,vector<Element>> AMap;
	for(const auto& AEntry:A)
	{
		AMap[AEntry.col].push_back(AEntry); 
	}

	for(const auto& BEntry:TB)
	{
		if(AMap.find(BEntry.col)!=AMap.end())
		{
			for(auto const& AEntry:AMap[BEntry.col])
			{
				tempC[AEntry.row-startRow][BEntry.row-startCol]+=AEntry.val*BEntry.val;
			}
		}
	}
};


int main(int argc,char** argv)
{
	uint64_t n = stoull(argv[1]);
	double rate = atof(argv[2]);
	bool flag = atoi(argv[3]);
	string fileName = argv[4];

	int dims[2];
	int periods[2]={1,1};
	int uprank,downrank,leftrank,rightrank;
	int rank2d,rank;
	int coords[2];
	MPI_Comm comm_2d;
	MPI_Status status;
	int nPes;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nPes);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	dims[0]=sqrt(nPes);
	dims[1]=sqrt(nPes);
	uint64_t nPerBlock = n/dims[0];
	MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,1,&comm_2d);
	
	MPI_Comm_rank(comm_2d,&rank2d);
	MPI_Cart_coords(comm_2d,rank2d,2,coords);
	uint64_t startRow = coords[1]*nPerBlock;
	uint64_t endRow = (coords[1]+1)*nPerBlock;
	uint64_t startCol = coords[0]*nPerBlock;
	uint64_t endCol = (coords[0]+1)*nPerBlock;

	//create datatype
	int blends[3] = {1,1,1};
	MPI_Datatype types[3] = {MPI_UINT64_T,MPI_UINT64_T,MPI_UINT64_T};
	MPI_Aint disps[3] = {0,8,16};
	MPI_Datatype mpi_tmp_t,mpi_struct_t;
	MPI_Type_create_struct(3,blends,disps,types,&mpi_tmp_t);
	MPI_Type_create_resized(mpi_tmp_t,0,24,&mpi_struct_t);
	MPI_Type_commit(&mpi_struct_t);

	vector<Element> A = myMatrixGen(rate,startRow,endRow,startCol,endCol);
	vector<Element> B = myMatrixGen(rate,startRow,endRow,startCol,endCol);
	double startTime = 0;
    if (rank == 0) {
        startTime = MPI_Wtime()*1000;
    }
	vector<vector<uint64_t>> tempC(nPerBlock,vector<uint64_t>(nPerBlock,0));
	vector<Element> TB = myTranspose(B,startRow,endRow,startCol,endCol);
	
	
	vector<Element> cpA = A;
	vector<Element> recvA = A;
	vector<Element> cpTB = TB;
	vector<Element> recvTB = TB;
	
	uint64_t sendCountA=cpA.size();
	uint64_t recvCountA=cpA.size();
	uint64_t sendCountB=cpTB.size();
	uint64_t recvCountB=cpTB.size();

	MPI_Cart_shift(comm_2d,0,-coords[1],&rightrank,&leftrank);
	MPI_Cart_shift(comm_2d,1,-coords[0],&downrank,&uprank);

	MPI_Sendrecv(&sendCountA,1,MPI_UINT64_T,leftrank,111,&recvCountA,1,MPI_UINT64_T,rightrank,111,comm_2d,&status);
	recvA.resize(recvCountA);
	MPI_Sendrecv(&cpA[0],sendCountA,mpi_struct_t,leftrank,111,&recvA[0],recvCountA,mpi_struct_t,rightrank,111,comm_2d,&status);
	cpA = recvA;

	MPI_Sendrecv(&sendCountB,1,MPI_UINT64_T,uprank,111,&recvCountB,1,MPI_UINT64_T,downrank,111,comm_2d,&status);
	recvTB.resize(recvCountB);
	MPI_Sendrecv(&cpTB[0],sendCountB,mpi_struct_t,uprank,111,&recvTB[0],recvCountB,mpi_struct_t,downrank,111,comm_2d,&status);
	cpTB = recvTB;
	
	MPI_Cart_shift(comm_2d,0,-1,&rightrank,&leftrank);
	MPI_Cart_shift(comm_2d,1,-1,&downrank,&uprank);

	// block multiplication
	for(int i=0;i<dims[0];i++)
	{
		myMultiply(tempC,cpA,cpTB,startRow,endRow,startCol,endCol);
		MPI_Sendrecv(&sendCountA,1,MPI_UINT64_T,leftrank,111,&recvCountA,1,MPI_UINT64_T,rightrank,111,comm_2d,&status);
		recvA.resize(recvCountA);
		MPI_Sendrecv(&cpA[0],sendCountA,mpi_struct_t,leftrank,111,&recvA[0],recvCountA,mpi_struct_t,rightrank,111,comm_2d,&status);
		cpA = recvA;
		sendCountA = recvCountA;

		MPI_Sendrecv(&sendCountB,1,MPI_UINT64_T,uprank,111,&recvCountB,1,MPI_UINT64_T,downrank,111,comm_2d,&status);
		recvTB.resize(recvCountB);
		MPI_Sendrecv(&cpTB[0],sendCountB,mpi_struct_t,uprank,111,&recvTB[0],recvCountB,mpi_struct_t,downrank,111,comm_2d,&status);
		cpTB = recvTB;
		sendCountB = recvCountB;
	}
	double multplyTime = 0;
    if (rank == 0) {
        multplyTime = MPI_Wtime()*1000;
        printf("%0.6f\n", multplyTime - startTime);
    }

	if(flag)
	{

		vector<Element> C;
		for(uint64_t i=0;i<nPerBlock;i++)
		{
			for(uint64_t j=0;j<nPerBlock;j++)
			{
				C.emplace_back(i+startRow,j+startCol,tempC[i][j]);
			}
		}

		int sizeA = A.size();
		int sizeB = B.size();
		int sizeC = C.size();
		int* arrSizeA = new int[nPes];
		int* arrSizeB = new int[nPes];
		int* arrSizeC = new int[nPes];
		int* dispA = new int[nPes];
		int* dispB = new int[nPes];
		int* dispC = new int[nPes];
		int totalSizeA = 0;
		int totalSizeB = 0;
		int totalSizeC = 0;
		dispA[0] = 0;
		dispB[0] = 0;
		dispC[0] = 0;
		MPI_Allgather(&sizeA,1,MPI_INT,arrSizeA,1,MPI_INT,comm_2d);
		MPI_Allgather(&sizeB,1,MPI_INT,arrSizeB,1,MPI_INT,comm_2d);
		MPI_Allgather(&sizeC,1,MPI_INT,arrSizeC,1,MPI_INT,comm_2d);
		for(int i=1;i<nPes;i++)
		{
			dispA[i] = dispA[i-1] + arrSizeA[i-1]; 
			dispB[i] = dispB[i-1] + arrSizeB[i-1];
			dispC[i] = dispC[i-1] + arrSizeC[i-1];
			totalSizeA += arrSizeA[i-1];
			totalSizeB += arrSizeB[i-1];
			totalSizeC += arrSizeC[i-1];
		}
		totalSizeA += arrSizeA[nPes-1];
		totalSizeB += arrSizeB[nPes-1];
		totalSizeC += arrSizeC[nPes-1];

		vector<Element> totalA(totalSizeA,Element());
		vector<Element> totalB(totalSizeB,Element());
		vector<Element> totalC(totalSizeC,Element());
		MPI_Gatherv(&A[0],sizeA,mpi_struct_t,&totalA[0],arrSizeA,dispA,mpi_struct_t,0,comm_2d);
		MPI_Gatherv(&B[0],sizeB,mpi_struct_t,&totalB[0],arrSizeB,dispB,mpi_struct_t,0,comm_2d);
		MPI_Gatherv(&C[0],sizeC,mpi_struct_t,&totalC[0],arrSizeC,dispC,mpi_struct_t,0,comm_2d);
		
		if(rank2d==0)
		{
			//cout<< "Start to print out\n"<<endl;
			vector<vector<uint64_t> > resA(n,vector<uint64_t>(n,0));
			vector<vector<uint64_t> > resB(n,vector<uint64_t>(n,0));
			vector<vector<uint64_t> > resC(n,vector<uint64_t>(n,0));
			for(auto& ele:totalA)
			{
				resA[ele.row][ele.col] = ele.val;
			}

			for(auto& ele:totalB)
			{
				resB[ele.row][ele.col] = ele.val;
			}

			for(auto& ele:totalC)
			{
				resC[ele.row][ele.col] = ele.val;
				//cout<<ele.row<<" "<<ele.col<<" "<<ele.val<<endl;
			}


			ofstream outfile;
			outfile.open(fileName);
			for(uint64_t i=0;i<n;i++)
			{
				for(uint64_t j=0;j<n;j++)
				{
					outfile<<resA[i][j];
					if(j<n-1)
					{
						outfile<<" ";
					}
				}
				outfile<<"\n";
			}
			outfile<<"\n";

			for(uint64_t i=0;i<n;i++)
			{
				for(uint64_t j=0;j<n;j++)
				{
					outfile<<resB[i][j];
					if(j<n-1)
					{
						outfile<<" ";
					}
				}
				outfile<<"\n";
			}
			outfile<<"\n";

			for(uint64_t i=0;i<n;i++)
			{
				for(uint64_t j=0;j<n;j++)
				{
					outfile<<resC[i][j];
					if(j<n-1)
					{
						outfile<<" ";
					}
				}
				outfile<<"\n";
			}
			outfile.close();
		}
	}	
	MPI_Comm_free(&comm_2d);
	MPI_Finalize();
	return 0;
}