CFLAGS=-I${MATLAB}/extern/include

all : mex

mex: KCenterClustering.o
	mex $(CFLAGS) mexmain.cpp KCenterClustering.o


KCenterClustering.o : KCenterClustering.cpp
	g++ -c $(CFLAGS) KCenterClustering.cpp

clean : 
	rm -rf KCenterClustering.o
