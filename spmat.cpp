#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <vector>
#include <random>
#include <string>
#include <unordered_map>
#include <mpi.h>
#include "element.h"

using namespace std;


vector<Element> transpose(vector<Element>& M,uint64_t m,uint64_t n)
{
	uint64_t N = M.size();
	vector<Element> res(N,Element(0,0,0));
	vector<uint64_t> num(n+1,0);
	vector<uint64_t> index(n+1,0);
	for(uint64_t i=0;i<N;i++)
	{
		num[M[i].col]++;
	}

	for(uint64_t i=1;i<=n;i++)
	{
		index[i] = index[i-1]+num[i-1];
	}

	for(uint64_t i=0;i<N;i++)
	{

		res[index[M[i].col]].row = M[i].col;
		res[index[M[i].col]].col = M[i].row;
		res[index[M[i].col]].val = M[i].val;
		index[M[i].col]++;
	}
	return res;
};

void Multiply(vector<vector<uint64_t>>& tempC,unordered_map<uint64_t,vector<Element>>& AMap, vector<Element>& TB,uint64_t startRow,uint64_t endRow)
{

	for(const auto& BEntry: TB)
	{
		if(AMap.find(BEntry.col)!=AMap.end())
		{
			for(auto const& AEntry:AMap[BEntry.col])
			{
				tempC[AEntry.row-startRow][BEntry.row]+=AEntry.val*BEntry.val; 
			}
		}
	}
};



vector<Element> matrixGen(double sparsity,uint64_t startRow,uint64_t numRows,uint64_t numCols)
{
	vector<Element> matrix;
	random_device seed;
	mt19937 rng(seed());
	uniform_real_distribution<double> uniform;
	srand(time(NULL));
	double s = 0;
	for(uint64_t i=0;i<numRows;i++)
	{
		for(uint64_t j=0;j<numCols;j++)
		{	
			double possibility = uniform(rng);
			if(possibility<sparsity)
			{
				matrix.emplace_back(i+startRow,j,1+(uint64_t)rand());
			}
		}
	}
	return matrix;
}




int main(int argc,char** argv)
{
	uint64_t n = stoull(argv[1]);
	double rate = atof(argv[2]);
	bool flag = atoi(argv[3]);
	string fileName = argv[4];
	
	int rank;
	int size;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	uint64_t rowPerP = n/size;
	uint64_t startRow = rowPerP*rank;
	uint64_t endRow = rowPerP*(rank+1);
	vector<Element> A;
	vector<Element> B;

	A = matrixGen(rate,rank*rowPerP,rowPerP,n);
	B = matrixGen(rate,rank*rowPerP,rowPerP,n);
	vector<vector<uint64_t>> tempC(rowPerP,vector<uint64_t>(n,0));
	double startTime = 0;
    if (rank == 0) {
        startTime = MPI_Wtime()*1000;
    }
	vector<Element> TB = transpose(B,rowPerP,n);
	int* sendcounts = new int[size];
	int* sdispls = new int[size];
	int* recvcounts = new int[size];
	int* rdispls = new int[size];
	for(int i=0;i<size;i++)
	{
		sendcounts[i] = 0;
		sdispls[i] = 0;
		recvcounts[i] = 0;
		rdispls[i] = 0;
	}
	for(auto& ele:TB)
	{
		sendcounts[ele.row/rowPerP]++;
	}

	sdispls[0]=0;
	for(int i=1;i<size;i++)
	{
		//sdispls[i] = sdispls[i-1] + sendcounts[i-1]*sizeof(Element);
		sdispls[i] = sdispls[i-1] + sendcounts[i-1];
	}
	MPI_Alltoall(sendcounts,1,MPI_INT,recvcounts,1,MPI_INT,MPI_COMM_WORLD);
	int totalElement = 0;
	for(int i=1;i<size;i++)
	{
		//rdispls[i] = rdispls[i-1] + recvcounts[i-1]*sizeof(Element);
		rdispls[i] = rdispls[i-1] + recvcounts[i-1];
		totalElement+=recvcounts[i-1];
	}
	totalElement+=recvcounts[size-1];
	vector<Element> partTB(totalElement,Element(0,0,0));

	//create datatype
	int blends[3] = {1,1,1};
	MPI_Datatype types[3] = {MPI_UINT64_T,MPI_UINT64_T,MPI_UINT64_T};
	MPI_Aint disps[3] = {0,8,16};
	MPI_Datatype mpi_tmp_t,mpi_struct_t;
	MPI_Type_create_struct(3,blends,disps,types,&mpi_tmp_t);
	MPI_Type_create_resized(mpi_tmp_t,0,24,&mpi_struct_t);
	MPI_Type_commit(&mpi_struct_t);

	MPI_Alltoallv(&TB[0],sendcounts,sdispls,mpi_struct_t,&partTB[0],recvcounts,rdispls,mpi_struct_t,MPI_COMM_WORLD);

	//vector<Element> sortTB = sortPartTB(partTB,recvcounts,size,rank*rowPerP,(rank+1)*rowPerP);

	unordered_map<uint64_t,vector<Element>> AMap;
	for(const auto& AEntry:A)
	{
		AMap[AEntry.col].push_back(AEntry);
	}


	vector<Element> C; 

	MPI_Comm comm;
	int ndims = 1;
	int dims = size;
	int periods = 1;
	int srank=0;
	int drank=0;

	MPI_Status stat;
	MPI_Cart_create(MPI_COMM_WORLD,ndims,&dims,&periods,0,&comm);
	MPI_Cart_shift(comm,0,1,&srank,&drank);
	for(int i=0;i<size;i++)
	{
		Multiply(tempC,AMap,TB,startRow,endRow);
		int sendSize = TB.size();
		int recvSize = 0;
		MPI_Sendrecv(&sendSize,1,MPI_INT,drank,111,&recvSize,1,MPI_INT,srank,111,comm,&stat);
		vector<Element> tmpTB(recvSize,Element(0,0,0));
		MPI_Sendrecv(&TB[0],sendSize,mpi_struct_t,drank,111,&tmpTB[0],recvSize,mpi_struct_t,srank,111,comm,&stat);
		TB = tmpTB;
	}
	double multplyTime = 0;
    if (rank == 0) {
        multplyTime = MPI_Wtime()*1000;
        printf("%0.6f\n", multplyTime - startTime);
    }

    if(flag)
    {

    	for(uint64_t i=0;i<rowPerP;i++)
		{
			for(uint64_t j=0;j<n;j++)
			{
				C.emplace_back(i+startRow,j,tempC[i][j]);
			}
		}
		int sizeA = A.size();
		int sizeB = B.size();
		int sizeC = C.size();
		int* arrSizeA = new int[size];
		int* arrSizeB = new int[size];
		int* arrSizeC = new int[size];
		int* dispA = new int[size];
		int* dispB = new int[size];
		int* dispC = new int[size];
		int totalSizeA = 0;
		int totalSizeB = 0;
		int totalSizeC = 0;
		dispA[0] = 0;
		dispB[0] = 0;
		dispC[0] = 0;
		MPI_Allgather(&sizeA,1,MPI_INT,arrSizeA,1,MPI_INT,comm);
		MPI_Allgather(&sizeB,1,MPI_INT,arrSizeB,1,MPI_INT,comm);
		MPI_Allgather(&sizeC,1,MPI_INT,arrSizeC,1,MPI_INT,comm);
		for(int i=1;i<size;i++)
		{
			dispA[i] = dispA[i-1] + arrSizeA[i-1]; 
			dispB[i] = dispB[i-1] + arrSizeB[i-1];
			dispC[i] = dispC[i-1] + arrSizeC[i-1];
			totalSizeA += arrSizeA[i-1];
			totalSizeB += arrSizeB[i-1];
			totalSizeC += arrSizeC[i-1];
		}
		totalSizeA += arrSizeA[size-1];
		totalSizeB += arrSizeB[size-1];
		totalSizeC += arrSizeC[size-1];

		vector<Element> totalA(totalSizeA,Element());
		vector<Element> totalB(totalSizeB,Element());
		vector<Element> totalC(totalSizeC,Element());
		MPI_Gatherv(&A[0],sizeA,mpi_struct_t,&totalA[0],arrSizeA,dispA,mpi_struct_t,0,comm);
		MPI_Gatherv(&B[0],sizeB,mpi_struct_t,&totalB[0],arrSizeB,dispB,mpi_struct_t,0,comm);
		MPI_Gatherv(&C[0],sizeC,mpi_struct_t,&totalC[0],arrSizeC,dispC,mpi_struct_t,0,comm);
		
		if(rank==0)
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
	MPI_Comm_free(&comm);
	MPI_Finalize();
	return 0;
}