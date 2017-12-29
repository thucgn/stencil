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

#define PRINT_DEBUG
#define THREADS 12
#define DIMT 3
#define DIMX 564
#define DIMY 564
#define ROWS 47
#define CENTER_WIGHT 2.0
#define ADJACENT_WIGHT 1.1
#define E(pgd, x, y, z) ((pgd->data)[((z)*pgd->ystride + (y)) * pgd->xstride + (x)])
#define ES(data, x, y, z, xs, ys) (data[((z)*(ys)+(y)) * (xs) + (x)])

static FT buffer[(DIMX+2*DIMT-2)*(DIMY+2*DIMT-2)*3*8] = {
	0
};

static FT buffer2[(DIMX+2*DIMT-4)*(DIMY+2*DIMT-4)*3*8] = {
	0
};


typedef struct grid_data
{
	PFT data;
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

	for(z = 1; z < pgd->Nz+1; ++z)
	{
		for(y = DIMT; y < pgd->Ny+DIMT; ++y)
			for(x = DIMT; x < pgd->Nx+DIMT; ++x)
			{
				//E(pgd, x, y, z) = (double)(rand() * 2) / RAND_MAX;
				E(pgd, x, y, z) = 1.0;
			}

	}
}


void compute_subplane(grid_data* pgd, grid_data* subgrid, int dimx, int dimy, int dimt, grid_data* outsubgrid, bool XLB, bool XRB, bool YLB, bool YRB)
{
	int Nz = pgd->Nz;
	int wr, rr, x,y,z,iter, wi, wc;
	int t1dimx = dimx + 2*DIMT - 2, t1dimy = dimy + 2*DIMT - 2;
	int t2dimx = dimx + 2*DIMT - 4, t2dimy = dimy + 2*DIMT - 4;
	PFT t1buf = &(ES(buffer, DIMT-1,DIMT-1, 0, t1dimx, t1dimy));
	PFT t2buf = &(ES(buffer2, DIMT-2,DIMT-2, 0, t2dimx, t2dimy));

	wr = (dimy + ROWS - 1)/ROWS, rr = dimy % ROWS;

	//#pragma omp parallel for private(x, y, z, iter, wc) num_threads(THREADS)
	for(wi = 0; wi < wr; ++ wi)
	{
		wc = (wi != wr - 1 || !rr) ? ROWS : rr;
		int offset = wi*ROWS;
		int zm0, zm1, zm2;
		for(z = 0; z < Nz; ++z)
		{
			//iter 1
			if(dimt == 1)
			{
				for(y = 1 - dimt; y < wc+dimt-1; ++ y)
				{
					for(x = 1 - dimt; x < dimx+dimt-1; ++x)
					{
						E(outsubgrid, x, y, z) = CENTER_WIGHT*E(subgrid, x,y+offset,z) + ADJACENT_WIGHT * ( E(subgrid,x-1,y+offset,z)
								+ E(subgrid, x+1,y+offset,z)
								+ E(subgrid, x, y-1+offset, z)
								+ E(subgrid, x, y+1+offset, z)
								+ E(subgrid, x, y+offset, z-1)
								+ E(subgrid, x, y+offset, z+1) );
					}
				}

			}
			else
			{
				for(y = 1 - DIMT; y < wc+DIMT-1; ++ y)
				{
					for(x = 1 - DIMT; x < dimx+DIMT-1; ++x)
					{
						if((XLB && x < 0) || (XRB && x >= dimx) || (YLB && y < 0) || (YRB && y >= dimy))
							ES(t1buf, x, y, z%dimt, t1dimx, t1dimy) = 0.0;
						else
							ES(t1buf, x, y, z%dimt, t1dimx, t1dimy) = CENTER_WIGHT*E(subgrid, x,y+offset,z) + ADJACENT_WIGHT * ( E(subgrid,x-1,y+offset,z)
								+ E(subgrid, x+1,y+offset,z)
								+ E(subgrid, x, y-1+offset, z)
								+ E(subgrid, x, y+1+offset, z)
								+ E(subgrid, x, y+offset, z-1)
								+ E(subgrid, x, y+offset, z+1) );
					}
				}

			}
#ifdef PRINT_DEBUG
			printf("iter1 z== : %d\n", z);
			for(y = 1 - DIMT; y < wc + DIMT-1; ++y)
			{
				for(x = 1 - DIMT; x < dimx + DIMT-1; ++x)
				{
					printf("%.1f ", ES(t1buf, x, y, z%dimt, t1dimx, t1dimy));
				}
				printf("\n");
			}
#endif
			//iter 2
			if(z >= 1 && dimt >= 2)
			{
				zm1 = (z-1) % dimt;
				if(z == 1)
				{
					for(y = 2 - DIMT; y < wc + DIMT-2; ++ y)
					{
						for(x = 2 - DIMT; x < dimx + DIMT-2; ++ x)
						{
							if((XLB && x < 0) || (XRB && x >= dimx) || (YLB && y < 0) || (YRB && y >= dimy))
								ES(t2buf, x, y, 0, t2dimx, t2dimy) = 0.0;
							else
								ES(t2buf, x, y, 0, t2dimx, t2dimy) = CENTER_WIGHT*ES(t1buf, x, y, 0, t1dimx, t1dimy) + ADJACENT_WIGHT * ( ES(t1buf, x-1, y, 0, t1dimx, t1dimy)
								+ ES(t1buf, x+1, y, 0, t1dimx, t1dimy)
								+ ES(t1buf, x, y-1, 0, t1dimx, t1dimy)
								+ ES(t1buf, x, y+1, 0, t1dimx, t1dimy)
								+ ES(t1buf, x, y, 1, t1dimx, t1dimy) );
						}
					}

				}
				else
				{
					for(y = 2 - DIMT; y < wc + DIMT-2; ++ y)
					{
						for(x = 2 - DIMT; x < dimx + DIMT-2; ++ x)
						{
							if((XLB && x < 0) || (XRB && x >= dimx) || (YLB && y < 0) || (YRB && y >= dimy))
								ES(t2buf, x, y, zm1, t2dimx, t2dimy) = 0.0;
							else
								ES(t2buf, x, y, zm1, t2dimx, t2dimy) = CENTER_WIGHT*ES(t1buf, x, y, zm1, t1dimx, t1dimy) + ADJACENT_WIGHT * ( ES(t1buf, x-1, y, zm1, t1dimx, t1dimy)
								+ ES(t1buf, x+1, y, zm1, t1dimx, t1dimy)
								+ ES(t1buf, x, y-1, zm1, t1dimx, t1dimy)
								+ ES(t1buf, x, y+1, zm1, t1dimx, t1dimy)
								+ ES(t1buf, x, y, (zm1-1 + dimt)%dimt, t1dimx, t1dimy)
								+ ES(t1buf, x, y, (zm1+1)%dimt, t1dimx, t1dimy) );
						}
					}

				}
			}
#ifdef PRINT_DEBUG
			printf("iter2 z== : %d\n", z);
			for(y = 2 - DIMT; y < wc + DIMT-2; ++y)
			{
				for(x = 2 - DIMT; x < dimx + DIMT-2; ++x)
				{
					printf("%.1f ", ES(t2buf, x, y, (z-1 + dimt)%dimt, t2dimx, t2dimy));
				}
				printf("\n");
			}
#endif
			//iter 3
			if(z >= 2 && dimt == 3)
			{
				zm2 = (z-2) % dimt; 
				if(z == 2)
				{
					for(y = 0; y < wc; ++ y)
					{
						for(x = 0; x < dimx; ++ x)
						{
							E(outsubgrid, x, y, 0) = CENTER_WIGHT*ES(t2buf, x, y, 0, t2dimx, t2dimy) + ADJACENT_WIGHT * ( ES(t2buf, x-1, y, 0, t2dimx, t2dimy)
								+ ES(t2buf, x+1, y, 0, t2dimx, t2dimy)
								+ ES(t2buf, x, y-1, 0, t2dimx, t2dimy)
								+ ES(t2buf, x, y+1, 0, t2dimx, t2dimy)
								+ ES(t2buf, x, y, 1, t2dimx, t2dimy) );
							
						}
					}

				}
				else
				{
					for(y = 0; y < wc; ++ y)
					{
						for(x = 0; x < dimx; ++ x)
						{
							E(outsubgrid, x, y, z-2) = CENTER_WIGHT*ES(t2buf, x, y, zm2, t2dimx, t2dimy) + ADJACENT_WIGHT * ( ES(t2buf, x-1, y, zm2, t2dimx, t2dimy)
								+ ES(t2buf, x+1, y, zm2, t2dimx, t2dimy)
								+ ES(t2buf, x, y-1, zm2, t2dimx, t2dimy)
								+ ES(t2buf, x, y+1, zm2, t2dimx, t2dimy)
								+ ES(t2buf, x, y, (zm2-1+dimt)%dimt, t2dimx, t2dimy) 
								+ ES(t2buf, x, y, (zm2+1)%dimt, t2dimx, t2dimy));
							
						}
					}

				}
			}
#ifdef PRINT_DEBUG
			printf("iter3 z== : %d\n", z);
			for(y = 0; y < wc; ++y)
			{
				for(x = 0; x < dimx; ++x)
				{
					printf("%.1f ", E(outsubgrid, x, y, (z-2+dimt)%dimt));
				}
				printf("\n");
			}
#endif

			

		}
		//z = Nz - 2
		if(dimt == 3)
		{
			int zm = (Nz - 1) % dimt;
			for(y = 2 - DIMT; y < wc + DIMT-2; ++ y)
			{
				for(x = 2 - DIMT; x < dimx + DIMT-2; ++ x)
				{
					if((XLB && x < 0) || (XRB && x >= dimx) || (YLB && y < 0) || (YRB && y >= dimy))
						ES(t2buf, x, y, zm, t2dimx, t2dimy) = 0.0;
					else
						ES(t2buf, x, y, zm, t2dimx, t2dimy) = CENTER_WIGHT*ES(t1buf, x, y, zm, t1dimx, t1dimy) + ADJACENT_WIGHT * ( 
						ES(t1buf, x-1, y, zm, t1dimx, t1dimy)
						+ ES(t1buf, x+1, y, zm, t1dimx, t1dimy)
						+ ES(t1buf, x, y-1, zm, t1dimx, t1dimy)
						+ ES(t1buf, x, y+1, zm, t1dimx, t1dimy)
						+ ES(t1buf, x, y, (zm-1 + dimt)%dimt, t1dimx, t1dimy));
				}
			}
			zm = (Nz - 2) % dimt;
			for(y = 0; y < wc; ++ y)
			{
				for(x = 0; x < dimx; ++ x)
				{
					E(outsubgrid, x, y, Nz-2) = CENTER_WIGHT*ES(t2buf, x, y, zm, t2dimx, t2dimy) + ADJACENT_WIGHT * ( ES(t2buf, x-1, y, zm, t2dimx, t2dimy)
						+ ES(t2buf, x+1, y, zm, t2dimx, t2dimy)
						+ ES(t2buf, x, y-1, zm, t2dimx, t2dimy)
						+ ES(t2buf, x, y+1, zm, t2dimx, t2dimy)
						+ ES(t2buf, x, y, (zm-1+dimt)%dimt, t2dimx, t2dimy) 
						+ ES(t2buf, x, y, (zm+1)%dimt, t2dimx, t2dimy));
					
				}
			}
			zm = (Nz - 1) % dimt;
			for(y = 0; y < wc; ++ y)
			{
				for(x = 0; x < dimx; ++ x)
				{
					E(outsubgrid, x, y, Nz-1) = CENTER_WIGHT*ES(t2buf, x, y, zm, t2dimx, t2dimy) + ADJACENT_WIGHT * ( ES(t2buf, x-1, y, zm, t2dimx, t2dimy)
						+ ES(t2buf, x+1, y, zm, t2dimx, t2dimy)
						+ ES(t2buf, x, y-1, zm, t2dimx, t2dimy)
						+ ES(t2buf, x, y+1, zm, t2dimx, t2dimy)
						+ ES(t2buf, x, y, (zm-1+dimt)%dimt, t2dimx, t2dimy));
					
				}
			}
		}
		
	}

}

void compute_stencil_iter(grid_data* pgd, int dimt, grid_data* out)
{
	int xw = (pgd->Nx + DIMX - 1) / DIMX, rx = (pgd->Nx) % DIMX;
	int yw = (pgd->Ny + DIMY - 1) / DIMY, ry = (pgd->Ny) % DIMY;
	int xwi, ywi, dimx, dimy;
	grid_data subgrid;
	subgrid.Nx = pgd->Nx;
	subgrid.Ny = pgd->Ny;
	subgrid.Nx = pgd->Nz;
	subgrid.xstride = pgd->xstride;
	subgrid.ystride = pgd->ystride;
	subgrid.zstride = pgd->zstride;
	subgrid.NIter = pgd->NIter;

	grid_data outsubgrid;
	outsubgrid.Nx = pgd->Nx;
	outsubgrid.Ny = pgd->Ny;
	outsubgrid.Nx = pgd->Nz;
	outsubgrid.xstride = pgd->xstride;
	outsubgrid.ystride = pgd->ystride;
	outsubgrid.zstride = pgd->zstride;
	outsubgrid.NIter = pgd->NIter;
	
	for(xwi = 0; xwi < xw; ++ xwi)
	{
		dimx = (xwi != xw - 1 || !rx) ? DIMX : rx;
		for(ywi = 0; ywi < yw; ++ ywi)
		{
			dimy = (ywi != yw - 1 || !ry) ? DIMY : ry;
			subgrid.data = &(E(pgd, xwi*DIMX+DIMT, ywi*DIMY+DIMT, 1));
			outsubgrid.data = &(E(out, xwi*DIMX+DIMT, ywi*DIMY+DIMT, 1));
			compute_subplane(pgd, &subgrid, dimx, dimy, dimt, &outsubgrid, xwi==0, xwi==xw-1, ywi==0, ywi==yw-1);
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
	gd.xstride = Nx + 2*DIMT;
	gd.ystride = Ny + 2*DIMT;
	gd.zstride = Nz + 2;
	gd.data = (PFT)_mm_malloc(gd.xstride * gd.ystride * gd.zstride * sizeof(FT), 64);

	grid_data gd2;
	gd2.Nx = Nx;
	gd2.Ny = Ny;
	gd2.Nz = Nz;
	gd2.NIter = NIter;
	gd2.xstride = Nx + 2*DIMT;
	gd2.ystride = Ny + 2*DIMT;
	gd2.zstride = Nz + 2;
	gd2.data = (PFT)_mm_malloc(gd2.xstride * gd2.ystride * gd2.zstride * sizeof(FT), 64);

	//initialize
	init_griddata(&gd);
	memset(gd2.data, 0, gd2.xstride * gd2.ystride * gd2.zstride * sizeof(FT));

	int itb = (NIter + DIMT - 1) / DIMT, rit = NIter  % DIMT;
	if(rit == 2) //iterator once more
	{
		++ itb;
		rit = 0;
	}
	int dimt, i;

	for(i = 0; i < itb; ++i)
	{
		dimt = (i != itb - 1 || !rit) ? DIMT : rit;
		if(i % 2 == 0)
			compute_stencil_iter(&gd, dimt, &gd2);
		else
			compute_stencil_iter(&gd2, dimt, &gd);

	}


#ifdef PRINT_DEBUG
	grid_data* pgd = &gd2;
	if(i % 2 == 0)
	{
		pgd = &gd;
	}
	for(int z = 0; z < Nz; ++z)
	{
		printf("z : %d\n", z);
		for(int y = 0; y < Ny; ++y)
		{
			for(int x = 0; x < Nx; ++x)
			{
				printf("%.1f ", E(pgd, x+DIMT, y+DIMT,z+1));
			}
			printf("\n");
		}
		printf("\n");
	}
#endif


	_mm_free(gd.data);
	_mm_free(gd2.data);
}

int main(int argc, char** argv)
{

	compute_stencil_3_5D(10, 10, 10, 3);

	return 0;
}
