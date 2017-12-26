/*************************************************************************
    > File Name: 3_5D_stencil.cpp
    > Author: cgn
    > Func: 
    > Created Time: 二 12/26 10:18:36 2017
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

using namespace std;

typedef double FT;
typedef double* PFT;
typedef size_t IT;

#define THREADS 12
#define DIMT 2
#define DIMX 492
#define DIMY 492
#define ROWS 41
#define CENTER_WIGHT 2.0
#define ADJACENT_WIGHT 1.1
#define E(pgd, x, y, z) (pgd->data)[((z)*pgd->ystride + (y)) * pgd->xstride + (x)]

static 

typedef struct grid_data
{
	PFT* data;
	IT Nx;
	IT Ny;
	IT Nz;
	IT xstride;
	IT ystride;
	IT zstride;
	int NIter;

} grid_data;


void init_griddata(grid_data* pgd)
{
	srand((unsigned int)time(NULL));	

	memset(pgd->data, 0, pgd->xstride * pgd->ystride * pgd->zstride * sizeof(FT));

	IT x, y, z;

	for(z = 1; z < pgd->zstride - 1; ++z)
	{
		for(y = 1; y < pgd->ystride - 1; ++y)
			for(x = 1; x < pgd->xstride - 1; ++x)
			{
				E(pgd, x, y, z) = (double)(rand() * 2) / RAND_MAX;
			}

	}
}

void compute_subplane(grid_data* pgd, PFT subplane, int dimx, int dimy)
{
	int Nz = pgd->Nz;
	
}

void compute_stencil_iter(grid_data* pgd)
{
	int xw = (pgd->xstride + DIMX - 1) / DIMX, rx = (pgd->xstride) % DIMX;
	int yw = (pgd->ystride + DIMY - 1) / DIMY, ry = (pgd->ystride) % DIMY;
	int xwi, ywi, dimx, dimy;
	for(xwi = 0; xwi < xw; ++ xwi)
	{
		dimx = (xwi != xw - 1 || !rx) ? DIMX : rx;
		for(ywi = 0; ywi < yw; ++ ywi)
		{
			dimy = (ywi != yw - 1 || !ry) ? DIMY : ry;
			compute_subplane(pgd, &(E(pgd, xwi*DIMX+1, ywi*DIMY+1, 1)), dimx, dimy);
		}
	}
}

void compute_stencil_3_5D(IT Nx, IT Ny, IT Nz, int NIter)
{
	grid_data gd;
	gd.Nx = Nx;
	gd.Ny = Ny;
	gd.Nz = Nz;
	gd.NIter = NIter;
	gd.xstride = Nx + 2;
	gd.ystride = Ny + 2;
	gd.zstride = Nz + 2;
	gd.data = _mm_malloc(gd.xstride * gd.ystride * gd.zstride * sizeof(FT), 64);

	//initialize
	init_griddata(&gd);

	int iter;

	for(iter = 0; iter < NIter; ++ iter)
	{
		compute_stencil_iter(&gd);
	}



	_mm_free(data);
}

int main(int argc, char** argv)
{

	
	return 0;
}
