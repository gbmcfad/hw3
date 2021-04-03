#   Makefile
  
CC = g++

CFLAGS = -std=c++11 -g -O3 -fopenmp

all: omp-scan sor2D-omp

omp-scan: omp-scan.cpp
	$(CC) $(CFLAGS) -o omp-scan omp-scan.cpp

sor2D-omp: sor2D-omp.cpp
	$(CC) $(CFLAGS) -o sor2D-omp sor2D-omp.cpp

clean:
	rm omp-scan sor2D-omp

