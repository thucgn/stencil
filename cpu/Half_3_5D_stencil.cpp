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
#define DIMX 404
#define DIMY 404
#define CENTER_WIGHT 2.0
#define ADJACENT_WIGHT 1.1
#define E(pgd, x, y, z) (pgd->data)[((z)*pgd->ystride + (y)) * pgd->xstride + (x)]


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

void read_one_rowpanel(PFT subplane, int dimx, )

void compute_subplane(grid_data* pgd, PFT subplane, int dimx, int dimy, int dimt, grid_data* out)
{
	int Nz = pgd->Nz;

	#pragma omp parallel num_threads(THREADS)
	{
		int threadid = omp_get_thread_num();
		PFT t0buf = &TBuffer[(DIMX+2)*(ROWS+2)*3*sizeof(FT)*threadid*2];
		PFT t1buf = t0buf + (DIMX+2)*(ROWS+2)*3*sizeof(FT);



		for(int z = 0;z < Nz; ++z)
		{
				
		}
	}
}

void compute_stencil_iter(grid_data* pgd, int dimt, grid_data* out)
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
			compute_subplane(pgd, &(E(pgd, xwi*DIMX+1, ywi*DIMY+1, 1)), dimx, dimy, dimt, grid_data* out);
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

	grid_data gd2;
	gd2.Nx = Nx;
	gd2.Ny = Ny;
	gd2.Nz = Nz;
	gd2.NIter = NIter;
	gd2.xstride = Nx + 2;
	gd2.ystride = Ny + 2;
	gd2.zstride = Nz + 2;
	gd2.data = _mm_malloc(gd2.xstride * gd2.ystride * gd2.zstride * sizeof(FT), 64);

	//initialize
	init_griddata(&gd);
	memset(gd2.data, 0, gd2.xstride * gd2.ystride * gd2.zstride * sizeof(FT));

	int itb = (NIter + DIMT - 1) / DIMT, rit = itb % DIMT;
	int dimt;

	for(int i = 0; i < itb; ++i)
	{
		dimt = (i != itb - 1 || !rit) ? DIMT : rit;
		compute_stencil_iter(&gd, dimt, &gd2);
	}



	_mm_free(data);
}

int main(int argc, char** argv)
{

	
	return 0;
}
