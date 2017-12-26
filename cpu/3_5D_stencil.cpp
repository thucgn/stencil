/*************************************************************************
    > File Name: 3_5D_stencil.cpp
    > Author: cgn
    > Func: 
    > Created Time: 二 12/26 10:18:36 2017
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

typedef double FT;
typedef double* PFT;

#define THREADS 12
#define DIMT 2
#define DIMX 492
#define DIMY 492
#define ROWS 41
#define CENTER_WIGHT 2.0
#define ADJACENT_WIGHT 1.1

using namespace std;


void init_griddata(PFT data,size_t size)
{
	srand((unsigned int)time(NULL));	
	for(size_t i = 0;i < size; ++i)
	{
		data[i] = double(rand())/RAND_MAX; //0.0 to 1.0
	}
}

void compute_stencil_3_5D(size_t Nx, size_t Ny, size_t Nz, int NIter)
{
	//initialize
	PFT data = (PFT)_mm_malloc(Nx * Ny * Nz * sizeof(FT), 64);
	init_griddata(data, Nx * Ny * Nz);





	_mm_free(data);
}

int main(int argc, char** argv)
{

	
	return 0;
}
