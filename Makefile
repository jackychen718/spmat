# Define the C++ compiler
CXX=mpicxx

all:
	${CXX} spmat.cpp -o spmat
	${CXX} myAlgorithm.cpp -o myAlgorithm


clean:
	rm -f spmat myAlgorithm

