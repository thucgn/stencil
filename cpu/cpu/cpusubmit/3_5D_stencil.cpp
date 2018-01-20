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
#include "immintrin.h"
#include "mkl.h"

using namespace std;

typedef double FT;
typedef double* PFT;
typedef size_t IT;

//#define PRINT_DEBUG
//#define PRINT_FINISH_DEBUG
#define THREADS 12
#define DIMT 3
#define DIMX 480
#define DIMY 480
#define BOUNDARY_OFFSET 2
#define ROWS 40
#define CENTER_WIGHT 1.0
#define ADJACENT_WIGHT 0.1
#define E(pgd, x, y, z) ((pgd->data)[((z)*pgd->ystride + (y)) * pgd->xstride + (x)])
#define ES(data, x, y, z, xs, ys) (data[((z)*(ys)+(y)) * (xs) + (x)])

static FT buffer[(DIMX+2*DIMT-2)*(DIMY+2*DIMT-2)*3*8] = {
	0
};

static FT buffer2[(DIMX+2*DIMT-4)*(DIMY+2*DIMT-4)*3*8] = {
	0
};

FT *bufferB1_1, *bufferB1_2, *bufferB2_1, *bufferB2_2, *bufferB3_1, *bufferB3_2, *bufferB4_1, *bufferB4_2;

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
		for(y = 1; y < pgd->Ny+1; ++y)
			for(x = 1; x < pgd->Nx+1; ++x)
			{
				//E(pgd, x, y, z) = (double)(rand() * 2) / RAND_MAX;
				E(pgd, x, y, z) = 1.0;
			}

	}
}


void compute_subplane(grid_data* pgd, grid_data* subgrid, int dimx, int dimy, int dimt, grid_data* outsubgrid)
{
	int Nz = pgd->Nz;
	int wr, rr, wi;
	memset(buffer, 0, (DIMX+2*DIMT-2)*(DIMY+2*DIMT-2)*3*8);
	memset(buffer2, 0, (DIMX+2*DIMT-4)*(DIMY+2*DIMT-4)*3*8);
	int t1dimx = dimx + 2*DIMT - 2, t1dimy = dimy + 2*DIMT - 2;
	int t2dimx = dimx + 2*DIMT - 4, t2dimy = dimy + 2*DIMT - 4;
	PFT t1buf = &(ES(buffer, DIMT-1,DIMT-1, 0, t1dimx, t1dimy));
	PFT t2buf = &(ES(buffer2, DIMT-2,DIMT-2, 0, t2dimx, t2dimy));

	wr = (dimy + ROWS - 1)/ROWS, rr = dimy % ROWS;

	#pragma omp parallel for schedule(dynamic) num_threads(12)
	for(wi = 0; wi < wr; ++ wi)
	{
		FT *x_stream, *y_m1_stream, *y_a1_stream, *z_m1_stream, *z_a1_stream, *out_center;
		int wc = (wi != wr - 1 || !rr) ? ROWS : rr;
		int x, y, z;
		int offset = wi*ROWS;
		//printf("id: %d, wi: %d, wr: %d, rr:%d, wc:%d, offset:%d\n",omp_get_thread_num(), wi, wr, rr, wc, offset);
		int zm0, zm1, zm2;
		register __m256d vc, vxm1, vxa1, vym1, vya1, vzm1, vza1;
		register __m256d cw = _mm256_set1_pd(CENTER_WIGHT);
		register __m256d aw = _mm256_set1_pd(ADJACENT_WIGHT);
		for(z = 0; z < Nz; ++z)
		{
			//iter 1
			if(dimt == 1)
			{
				for(y = offset; y < wc+offset; ++ y)
				{
					out_center = &(E(outsubgrid, 0, y, z));
					x_stream = &(E(subgrid, 0, y, z));
					y_m1_stream = &(E(subgrid, 0, y-1, z));
					y_a1_stream = &(E(subgrid, 0, y+1, z));
					z_m1_stream = &(E(subgrid, 0, y, z-1));
					z_a1_stream = &(E(subgrid, 0, y, z+1));
					/*for(x = 0; x < dimx; ++x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
								+ *(y_m1_stream+x) + *(y_a1_stream+x)
								+ *(z_m1_stream+x) + *(z_a1_stream+x));

					}*/
					for(x = 0; x+3 < dimx; x += 4)
					{
						vc   = _mm256_load_pd(x_stream+x);				
						vxm1 = _mm256_load_pd(x_stream+x-1);
						vc   = _mm256_mul_pd(vc, cw);
						vxa1 = _mm256_load_pd(x_stream+x+1);
						vc   = _mm256_fmadd_pd(vxm1, aw, vc);
						vym1 = _mm256_load_pd(y_m1_stream+x);
						vc   = _mm256_fmadd_pd(vxa1, aw, vc);
						vya1 = _mm256_load_pd(y_a1_stream+x);
						vc   = _mm256_fmadd_pd(vym1, aw, vc);
						vzm1 = _mm256_load_pd(z_m1_stream+x);
						vc   = _mm256_fmadd_pd(vya1, aw, vc);
						vza1 = _mm256_load_pd(z_a1_stream+x);
						vc   = _mm256_fmadd_pd(vzm1, aw, vc);
						vc   = _mm256_fmadd_pd(vza1, aw, vc);
						_mm256_store_pd(out_center+x, vc);
					}
					//x -= 4;
					for(; x < dimx; ++x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
								+ *(y_m1_stream+x) + *(y_a1_stream+x)
								+ *(z_m1_stream+x) + *(z_a1_stream+x));

					}

				}

			}
			else
			{
				for(y = 1 - DIMT +offset; y < wc+DIMT-1+offset; ++ y)
				{
					out_center = &(ES(t1buf, 0, y, z%dimt, t1dimx, t1dimy));
					x_stream = &(E(subgrid, 0, y, z));
					y_m1_stream = &(E(subgrid, 0, y-1, z));
					y_a1_stream = &(E(subgrid, 0, y+1, z));
					z_m1_stream = &(E(subgrid, 0, y, z-1));
					z_a1_stream = &(E(subgrid, 0, y, z+1));
					/*for(x = 1 - DIMT; x < dimx+DIMT-1; ++x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
								+ *(y_m1_stream+x) + *(y_a1_stream+x)
								+ *(z_m1_stream+x) + *(z_a1_stream+x));
					}*/
					for(x = 1 - DIMT; x+3 < dimx+DIMT-1; x += 4)
					{
						vc   = _mm256_load_pd(x_stream+x);				
						vxm1 = _mm256_load_pd(x_stream+x-1);
						vc   = _mm256_mul_pd(vc, cw);
						vxa1 = _mm256_load_pd(x_stream+x+1);
						vc   = _mm256_fmadd_pd(vxm1, aw, vc);
						vym1 = _mm256_load_pd(y_m1_stream+x);
						vc   = _mm256_fmadd_pd(vxa1, aw, vc);
						vya1 = _mm256_load_pd(y_a1_stream+x);
						vc   = _mm256_fmadd_pd(vym1, aw, vc);
						vzm1 = _mm256_load_pd(z_m1_stream+x);
						vc   = _mm256_fmadd_pd(vya1, aw, vc);
						vza1 = _mm256_load_pd(z_a1_stream+x);
						vc   = _mm256_fmadd_pd(vzm1, aw, vc);
						vc   = _mm256_fmadd_pd(vza1, aw, vc);
						_mm256_store_pd(out_center+x, vc);
					}
					//x -= 4;
					for(; x < dimx+DIMT-1; ++x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
								+ *(y_m1_stream+x) + *(y_a1_stream+x)
								+ *(z_m1_stream+x) + *(z_a1_stream+x));

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
				for(y = 2 - DIMT + offset; y < wc + DIMT-2 + offset; ++ y)
				{
					out_center = &(ES(t2buf, 0, y, zm1, t2dimx, t2dimy));
					x_stream = &(ES(t1buf, 0, y, zm1, t1dimx, t1dimy));
					y_m1_stream = &(ES(t1buf, 0, y-1, zm1, t1dimx, t1dimy));
					y_a1_stream = &(ES(t1buf, 0, y+1, zm1, t1dimx, t1dimy));
					z_m1_stream = &(ES(t1buf, 0, y, (zm1-1+dimt)%dimt, t1dimx, t1dimy));
					z_a1_stream = &(ES(t1buf, 0, y, (zm1+1)%dimt, t1dimx, t1dimy));
					/*for(x = 2 - DIMT; x < dimx + DIMT-2; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x) + *(z_a1_stream+x));
					}*/
					for(x = 2 - DIMT; x+3 < dimx+DIMT-2; x += 4)
					{
						vc   = _mm256_load_pd(x_stream+x);				
						vxm1 = _mm256_load_pd(x_stream+x-1);
						vc   = _mm256_mul_pd(vc, cw);
						vxa1 = _mm256_load_pd(x_stream+x+1);
						vc   = _mm256_fmadd_pd(vxm1, aw, vc);
						vym1 = _mm256_load_pd(y_m1_stream+x);
						vc   = _mm256_fmadd_pd(vxa1, aw, vc);
						vya1 = _mm256_load_pd(y_a1_stream+x);
						vc   = _mm256_fmadd_pd(vym1, aw, vc);
						vzm1 = _mm256_load_pd(z_m1_stream+x);
						vc   = _mm256_fmadd_pd(vya1, aw, vc);
						vza1 = _mm256_load_pd(z_a1_stream+x);
						vc   = _mm256_fmadd_pd(vzm1, aw, vc);
						vc   = _mm256_fmadd_pd(vza1, aw, vc);
						_mm256_store_pd(out_center+x, vc);
					}
					//x -= 4;
					for(; x < dimx+DIMT-2; ++x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
								+ *(y_m1_stream+x) + *(y_a1_stream+x)
								+ *(z_m1_stream+x) + *(z_a1_stream+x));

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
				for(y = offset; y < wc+offset; ++ y)
				{
					out_center = &(E(outsubgrid, 0, y, z-2));
					x_stream = &(ES(t2buf, 0, y, zm2, t2dimx, t2dimy));
					y_m1_stream = &(ES(t2buf, 0, y-1, zm2, t2dimx, t2dimy));
					y_a1_stream = &(ES(t2buf, 0, y+1, zm2, t2dimx, t2dimy));
					z_m1_stream = &(ES(t2buf, 0, y, (zm2-1+dimt)%dimt, t2dimx, t2dimy));
					z_a1_stream = &(ES(t2buf, 0, y, (zm2+1)%dimt, t2dimx, t2dimy));
					/*for(x = 0; x < dimx; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x) + *(z_a1_stream+x));
					}*/
					for(x = 0; x+3 < dimx; x += 4)
					{
						vc   = _mm256_load_pd(x_stream+x);				
						vxm1 = _mm256_load_pd(x_stream+x-1);
						vc   = _mm256_mul_pd(vc, cw);
						vxa1 = _mm256_load_pd(x_stream+x+1);
						vc   = _mm256_fmadd_pd(vxm1, aw, vc);
						vym1 = _mm256_load_pd(y_m1_stream+x);
						vc   = _mm256_fmadd_pd(vxa1, aw, vc);
						vya1 = _mm256_load_pd(y_a1_stream+x);
						vc   = _mm256_fmadd_pd(vym1, aw, vc);
						vzm1 = _mm256_load_pd(z_m1_stream+x);
						vc   = _mm256_fmadd_pd(vya1, aw, vc);
						vza1 = _mm256_load_pd(z_a1_stream+x);
						vc   = _mm256_fmadd_pd(vzm1, aw, vc);
						vc   = _mm256_fmadd_pd(vza1, aw, vc);
						_mm256_store_pd(out_center+x, vc);
					}
					//x -= 4;
					for(; x < dimx; ++x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
								+ *(y_m1_stream+x) + *(y_a1_stream+x)
								+ *(z_m1_stream+x) + *(z_a1_stream+x));

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
			zm0 = (Nz - 1) % dimt;
			for(y = 2 - DIMT + offset; y < wc + DIMT-2 + offset; ++ y)
			{
				out_center = &(ES(t2buf, 0, y, zm0, t2dimx, t2dimy));
				x_stream = &(ES(t1buf, 0, y, zm0, t1dimx, t1dimy));
				y_m1_stream = &(ES(t1buf, 0, y-1, zm0, t1dimx, t1dimy));
				y_a1_stream = &(ES(t1buf, 0, y+1, zm0, t1dimx, t1dimy));
				z_m1_stream = &(ES(t1buf, 0, y, (zm0-1+dimt)%dimt, t1dimx, t1dimy));
				//z_a1_stream = &(ES(t1buf, 0, y, (zm0+1)%dimt, t1dimx, t1dimy));
				/*for(x = 2 - DIMT; x < dimx + DIMT-2; ++ x)
				{
					*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
						+ *(y_m1_stream+x) + *(y_a1_stream+x)
						+ *(z_m1_stream+x) + *(z_a1_stream+x));
				}*/
					for(x = 2 - DIMT; x+3 < dimx+DIMT-2; x += 4)
					{
						vc   = _mm256_load_pd(x_stream+x);				
						vxm1 = _mm256_load_pd(x_stream+x-1);
						vc   = _mm256_mul_pd(vc, cw);
						vxa1 = _mm256_load_pd(x_stream+x+1);
						vc   = _mm256_fmadd_pd(vxm1, aw, vc);
						vym1 = _mm256_load_pd(y_m1_stream+x);
						vc   = _mm256_fmadd_pd(vxa1, aw, vc);
						vya1 = _mm256_load_pd(y_a1_stream+x);
						vc   = _mm256_fmadd_pd(vym1, aw, vc);
						vzm1 = _mm256_load_pd(z_m1_stream+x);
						vc   = _mm256_fmadd_pd(vya1, aw, vc);
						//vza1 = _mm256_load_pd(z_a1_stream+x);
						vc   = _mm256_fmadd_pd(vzm1, aw, vc);
						//vc   = _mm256_fmadd_pd(vza1, aw, vc);
						_mm256_store_pd(out_center+x, vc);
					}
					//x -= 4;
					for(; x < dimx+DIMT-2; ++x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
								+ *(y_m1_stream+x) + *(y_a1_stream+x)
								+ *(z_m1_stream+x));

					}
			}
			zm0 = (Nz - 2) % dimt;
			for(y = offset; y < wc+offset; ++ y)
			{
				out_center = &(E(outsubgrid, 0, y, Nz-2));
				x_stream = &(ES(t2buf, 0, y, zm0, t2dimx, t2dimy));
				y_m1_stream = &(ES(t2buf, 0, y-1, zm0, t2dimx, t2dimy));
				y_a1_stream = &(ES(t2buf, 0, y+1, zm0, t2dimx, t2dimy));
				z_m1_stream = &(ES(t2buf, 0, y, (zm0-1+dimt)%dimt, t2dimx, t2dimy));
				z_a1_stream = &(ES(t2buf, 0, y, (zm0+1)%dimt, t2dimx, t2dimy));
				/*for(x = 0; x < dimx; ++ x)
				{
					*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
						+ *(y_m1_stream+x) + *(y_a1_stream+x)
						+ *(z_m1_stream+x) + *(z_a1_stream+x));
				}*/
					for(x = 0; x+3 < dimx; x += 4)
					{
						vc   = _mm256_load_pd(x_stream+x);				
						vxm1 = _mm256_load_pd(x_stream+x-1);
						vc   = _mm256_mul_pd(vc, cw);
						vxa1 = _mm256_load_pd(x_stream+x+1);
						vc   = _mm256_fmadd_pd(vxm1, aw, vc);
						vym1 = _mm256_load_pd(y_m1_stream+x);
						vc   = _mm256_fmadd_pd(vxa1, aw, vc);
						vya1 = _mm256_load_pd(y_a1_stream+x);
						vc   = _mm256_fmadd_pd(vym1, aw, vc);
						vzm1 = _mm256_load_pd(z_m1_stream+x);
						vc   = _mm256_fmadd_pd(vya1, aw, vc);
						vza1 = _mm256_load_pd(z_a1_stream+x);
						vc   = _mm256_fmadd_pd(vzm1, aw, vc);
						vc   = _mm256_fmadd_pd(vza1, aw, vc);
						_mm256_store_pd(out_center+x, vc);
					}
					//x -= 4;
					for(; x < dimx; ++x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
								+ *(y_m1_stream+x) + *(y_a1_stream+x)
								+ *(z_m1_stream+x) + *(z_a1_stream+x));

					}
			}
			zm0 = (Nz - 1) % dimt;
			for(y = offset; y < wc+offset; ++ y)
			{
				out_center = &(E(outsubgrid, 0, y, Nz-1));
				x_stream = &(ES(t2buf, 0, y, zm0, t2dimx, t2dimy));
				y_m1_stream = &(ES(t2buf, 0, y-1, zm0, t2dimx, t2dimy));
				y_a1_stream = &(ES(t2buf, 0, y+1, zm0, t2dimx, t2dimy));
				z_m1_stream = &(ES(t2buf, 0, y, (zm0-1+dimt)%dimt, t2dimx, t2dimy));
				/*for(x = 0; x < dimx; ++ x)
				{
					*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
						+ *(y_m1_stream+x) + *(y_a1_stream+x)
						+ *(z_m1_stream+x));
				}*/
					for(x = 0; x+3 < dimx; x += 4)
					{
						vc   = _mm256_load_pd(x_stream+x);				
						vxm1 = _mm256_load_pd(x_stream+x-1);
						vc   = _mm256_mul_pd(vc, cw);
						vxa1 = _mm256_load_pd(x_stream+x+1);
						vc   = _mm256_fmadd_pd(vxm1, aw, vc);
						vym1 = _mm256_load_pd(y_m1_stream+x);
						vc   = _mm256_fmadd_pd(vxa1, aw, vc);
						vya1 = _mm256_load_pd(y_a1_stream+x);
						vc   = _mm256_fmadd_pd(vym1, aw, vc);
						vzm1 = _mm256_load_pd(z_m1_stream+x);
						vc   = _mm256_fmadd_pd(vya1, aw, vc);
						//vza1 = _mm256_load_pd(z_a1_stream+x);
						vc   = _mm256_fmadd_pd(vzm1, aw, vc);
						//vc   = _mm256_fmadd_pd(vza1, aw, vc);
						_mm256_store_pd(out_center+x, vc);
					}
					//x -= 4;
					for(; x < dimx; ++x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
								+ *(y_m1_stream+x) + *(y_a1_stream+x)
								+ *(z_m1_stream+x) );

					}
			}
		}
		
	}

}

void compute_boundary(grid_data* pgd, grid_data* subgrid, int dimx, int dimy, int dimt, grid_data* outsubgrid)
{
	//if(dimt == 1)
	//{
		memset(bufferB1_1, 0, (dimx+2)*(BOUNDARY_OFFSET+DIMT)*3*8);
		memset(bufferB1_2, 0, (dimx+2)*(BOUNDARY_OFFSET+DIMT-1)*3*8);
		memset(bufferB2_1, 0, (dimx+2)*(BOUNDARY_OFFSET+DIMT)*3*8);
		memset(bufferB2_2, 0, (dimx+2)*(BOUNDARY_OFFSET+DIMT-1)*3*8);
		memset(bufferB3_1, 0, (BOUNDARY_OFFSET+DIMT)*(dimy+2)*3*8);
		memset(bufferB3_2, 0, (BOUNDARY_OFFSET+DIMT-1)*(dimy+2)*3*8);
		memset(bufferB4_1, 0, (BOUNDARY_OFFSET+DIMT)*(dimy+2)*3*8);
		memset(bufferB4_2, 0, (BOUNDARY_OFFSET+DIMT-1)*(dimy+2)*3*8);

	//}
	int Nx = subgrid->Nx;
	int Ny = subgrid->Ny;
	int Nz = subgrid->Nz;

	//x from 0 to Nx-1, y from 0 to offset-1
	#pragma omp parallel num_threads(4)
	{
	#pragma omp sections
	{
		#pragma omp section // x from 0 to Nz-1, y from 0 to offset-1
		{
			int x, y, z;
			FT *x_stream, *y_m1_stream, *y_a1_stream, *z_m1_stream, *z_a1_stream, *out_center;
			int t1dimx = dimx + 2, t1dimy = BOUNDARY_OFFSET + dimt;
			int t2dimx = dimx + 2, t2dimy = BOUNDARY_OFFSET + dimt - 1;
			int zm0, zm1, zm2;
			FT *buf1 = &(ES(bufferB1_1, 1, 1, 0, t1dimx, t1dimy));
			FT *buf2 = &(ES(bufferB1_2, 1, 1, 0, t2dimx, t2dimy));

			for(z = 0; z < Nz; ++ z)
			{
				if(dimt == 1)
				{
					for(y = 0; y < BOUNDARY_OFFSET; ++ y)
					{
						out_center = &(E(outsubgrid, 0, y, z));
						x_stream = &(E(subgrid, 0, y, z));
						y_m1_stream = &(E(subgrid, 0, y-1, z));
						y_a1_stream = &(E(subgrid, 0, y+1, z));
						z_m1_stream = &(E(subgrid, 0, y, z-1));
						z_a1_stream = &(E(subgrid, 0, y, z+1));
						for(x = 0; x < Nx; ++ x)
						{
							*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
									+ *(y_m1_stream + x) + *(y_a1_stream + x)
									+ *(z_m1_stream + x) + *(z_a1_stream + x));
						}

					}

				}
				else
				{
					for(y = 0; y < BOUNDARY_OFFSET + dimt-1; ++ y)
					{
						out_center = &(ES(buf1, 0, y, z%dimt, t1dimx, t1dimy));
						x_stream = &(E(subgrid, 0, y, z));
						y_m1_stream = &(E(subgrid, 0, y-1, z));
						y_a1_stream = &(E(subgrid, 0, y+1, z));
						z_m1_stream = &(E(subgrid, 0, y, z-1));
						z_a1_stream = &(E(subgrid, 0, y, z+1));
						for(x = 0; x < Nx; ++ x)
						{
							*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
									+ *(y_m1_stream + x) + *(y_a1_stream + x)
									+ *(z_m1_stream + x) + *(z_a1_stream + x));
						}

					}
					if(z >= 1 && dimt >= 2)
					{
						for(y = 0; y < BOUNDARY_OFFSET + dimt-2; ++ y)
						{
							//y == 0 subplane
							zm1 = (z-1)%dimt;
							out_center = &(ES(buf2, 0, y, zm1, t2dimx, t2dimy));
							x_stream = &(ES(buf1, 0, y, zm1, t1dimx, t1dimy));
							y_m1_stream = &(ES(buf1, 0, y-1, zm1, t1dimx, t1dimy));
							y_a1_stream = &(ES(buf1, 0, y+1, zm1, t1dimx, t1dimy));
							z_m1_stream = &(ES(buf1, 0, y, (zm1-1+dimt)%dimt, t1dimx, t1dimy));
							z_a1_stream = &(ES(buf1, 0, y, (zm1+1)%dimt, t1dimx, t1dimy));
							for(x = 0; x < Nx; ++ x)
							{
								*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
										+ *(y_m1_stream + x) + *(y_a1_stream + x)
										+ *(z_m1_stream + x) + *(z_a1_stream + x));
							}
						}

					}
					if(z >= 2 && dimt == 3)
					{
						for(y = 0; y < BOUNDARY_OFFSET; ++ y)
						{
							//y == 0 subplane
							zm2 = (z-2)%dimt;
							out_center = &(E(outsubgrid, 0, y, z-2));
							x_stream = &(ES(buf2, 0, y, zm2, t2dimx, t2dimy));
							y_m1_stream = &(ES(buf2, 0, y-1, zm2, t2dimx, t2dimy));
							y_a1_stream = &(ES(buf2, 0, y+1, zm2, t2dimx, t2dimy));
							z_m1_stream = &(ES(buf2, 0, y, (zm2-1+dimt)%dimt, t2dimx, t2dimy));
							z_a1_stream = &(ES(buf2, 0, y, (zm2+1)%dimt, t2dimx, t2dimy));
							for(x = 0; x < Nx; ++ x)
							{
								*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
										+ *(y_m1_stream + x) + *(y_a1_stream + x)
										+ *(z_m1_stream + x) + *(z_a1_stream + x));
							}
						}

					}

				}
			
			}

			if(dimt == 3)
			{
				zm0 = (Nz - 1) % dimt;
				for(y = 0; y < BOUNDARY_OFFSET + dimt-2; ++ y)
				{
					out_center = &(ES(buf2, 0, y, zm0, t2dimx, t2dimy));
					x_stream = &(ES(buf1, 0, y, zm0, t1dimx, t1dimy));
					y_m1_stream = &(ES(buf1, 0, y-1, zm0, t1dimx, t1dimy));
					y_a1_stream = &(ES(buf1, 0, y+1, zm0, t1dimx, t1dimy));
					z_m1_stream = &(ES(buf1, 0, y, (zm0-1+dimt)%dimt, t1dimx, t1dimy));
					//z_a1_stream = &(ES(buf1, 0, y, (zm0+1)%dimt, t1dimx, t1dimy));
					for(x = 0; x < Nx; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x));
					}
				}
				zm0 = (Nz - 2) % dimt;
				for(y = 0; y < BOUNDARY_OFFSET; ++ y)
				{
					out_center = &(E(outsubgrid, 0, y, Nz-2));
					x_stream = &(ES(buf2, 0, y, zm0, t2dimx, t2dimy));
					y_m1_stream = &(ES(buf2, 0, y-1, zm0, t2dimx, t2dimy));
					y_a1_stream = &(ES(buf2, 0, y+1, zm0, t2dimx, t2dimy));
					z_m1_stream = &(ES(buf2, 0, y, (zm0-1+dimt)%dimt, t2dimx, t2dimy));
					z_a1_stream = &(ES(buf2, 0, y, (zm0+1)%dimt, t2dimx, t2dimy));
					for(x = 0; x < Nx; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x) + *(z_a1_stream+x));
					}
				}
				zm0 = (Nz - 1) % dimt;
				for(y = 0; y < BOUNDARY_OFFSET; ++ y)
				{
					out_center = &(E(outsubgrid, 0, y, Nz-1));
					x_stream = &(ES(buf2, 0, y, zm0, t2dimx, t2dimy));
					y_m1_stream = &(ES(buf2, 0, y-1, zm0, t2dimx, t2dimy));
					y_a1_stream = &(ES(buf2, 0, y+1, zm0, t2dimx, t2dimy));
					z_m1_stream = &(ES(buf2, 0, y, (zm0-1+dimt)%dimt, t2dimx, t2dimy));
					for(x = 0; x < Nx; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x));
					}
				}
			
			}

		}

		#pragma omp section  // x from 0 to Nz-1, y from Ny-offset to Ny - 1
		{
			int x, y, z;
			FT *x_stream, *y_m1_stream, *y_a1_stream, *z_m1_stream, *z_a1_stream, *out_center;
			int t1dimx = dimx + 2, t1dimy = BOUNDARY_OFFSET + dimt;
			int t2dimx = dimx + 2, t2dimy = BOUNDARY_OFFSET + dimt - 1;
			int zm0, zm1, zm2;
			FT *buf1 = &(ES(bufferB2_1, 1, 0, 0, t1dimx, t1dimy));
			FT *buf2 = &(ES(bufferB2_2, 1, 0, 0, t2dimx, t2dimy));

			for(z = 0; z < Nz; ++ z)
			{
				if(dimt == 1)
				{
					for(y = Ny-BOUNDARY_OFFSET; y < Ny; ++ y)
					{
						out_center = &(E(outsubgrid, 0, y, z));
						x_stream = &(E(subgrid, 0, y, z));
						y_m1_stream = &(E(subgrid, 0, y-1, z));
						y_a1_stream = &(E(subgrid, 0, y+1, z));
						z_m1_stream = &(E(subgrid, 0, y, z-1));
						z_a1_stream = &(E(subgrid, 0, y, z+1));
						for(x = 0; x < Nx; ++ x)
						{
							*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
									+ *(y_m1_stream + x) + *(y_a1_stream + x)
									+ *(z_m1_stream + x) + *(z_a1_stream + x));
						}

					}

				}
				else
				{
					int startY = Ny-BOUNDARY_OFFSET-dimt+1;
					//for(y = startY; y < Ny; ++ y)
					for(y = 0; y < BOUNDARY_OFFSET+dimt-1; ++ y)
					{
						out_center = &(ES(buf1, 0, y, z%dimt, t1dimx, t1dimy));
						x_stream = &(E(subgrid, 0, y+startY, z));
						y_m1_stream = &(E(subgrid, 0, y-1+startY, z));
						y_a1_stream = &(E(subgrid, 0, y+1+startY, z));
						z_m1_stream = &(E(subgrid, 0, y+startY, z-1));
						z_a1_stream = &(E(subgrid, 0, y+startY, z+1));
						for(x = 0; x < Nx; ++ x)
						{
							*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
									+ *(y_m1_stream + x) + *(y_a1_stream + x)
									+ *(z_m1_stream + x) + *(z_a1_stream + x));
						}

					}
					if(z >= 1 && dimt >= 2)
					{
						//for(y = Ny-BOUNDARY_OFFSET-dimt+2; y < Ny; ++ y)
						for(y = 0; y < BOUNDARY_OFFSET+dimt-2; ++y)
						{
							//y == 0 subplane
							zm1 = (z-1)%dimt;
							out_center = &(ES(buf2, 0, y, zm1, t2dimx, t2dimy));
							x_stream = &(ES(buf1, 0, y+1, zm1, t1dimx, t1dimy));
							y_m1_stream = &(ES(buf1, 0, y-1+1, zm1, t1dimx, t1dimy));
							y_a1_stream = &(ES(buf1, 0, y+1+1, zm1, t1dimx, t1dimy));
							z_m1_stream = &(ES(buf1, 0, y+1, (zm1-1+dimt)%dimt, t1dimx, t1dimy));
							z_a1_stream = &(ES(buf1, 0, y+1, (zm1+1)%dimt, t1dimx, t1dimy));
							for(x = 0; x < Nx; ++ x)
							{
								*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
										+ *(y_m1_stream + x) + *(y_a1_stream + x)
										+ *(z_m1_stream + x) + *(z_a1_stream + x));
							}
						}

					}
					if(z >= 2 && dimt == 3)
					{
						startY = Ny-BOUNDARY_OFFSET;
						for(y = 0; y < BOUNDARY_OFFSET; ++ y)
						{
							//y == 0 subplane
							zm2 = (z-2)%dimt;
							out_center = &(E(outsubgrid, 0, y+startY, z-2));
							x_stream = &(ES(buf2, 0, y+1, zm2, t2dimx, t2dimy));
							y_m1_stream = &(ES(buf2, 0, y-1+1, zm2, t2dimx, t2dimy));
							y_a1_stream = &(ES(buf2, 0, y+1+1, zm2, t2dimx, t2dimy));
							z_m1_stream = &(ES(buf2, 0, y+1, (zm2-1+dimt)%dimt, t2dimx, t2dimy));
							z_a1_stream = &(ES(buf2, 0, y+1, (zm2+1)%dimt, t2dimx, t2dimy));
							for(x = 0; x < Nx; ++ x)
							{
								*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
										+ *(y_m1_stream + x) + *(y_a1_stream + x)
										+ *(z_m1_stream + x) + *(z_a1_stream + x));
							}
						}

					}
				}
			}
			if(dimt == 3)
			{
				zm0 = (Nz - 1) % dimt;
				//for(y = Ny-BOUNDARY_OFFSET-dimt+2; y < Ny; ++ y)
				for(y = 0; y < BOUNDARY_OFFSET+dimt-2; ++ y)
				{
					out_center = &(ES(buf2, 0, y, zm0, t2dimx, t2dimy));
					x_stream = &(ES(buf1, 0, y+1, zm0, t1dimx, t1dimy));
					y_m1_stream = &(ES(buf1, 0, y-1+1, zm0, t1dimx, t1dimy));
					y_a1_stream = &(ES(buf1, 0, y+1+1, zm0, t1dimx, t1dimy));
					z_m1_stream = &(ES(buf1, 0, y+1, (zm0-1+dimt)%dimt, t1dimx, t1dimy));
					//z_a1_stream = &(ES(buf1, 0, y+1, (zm0+1)%dimt, t1dimx, t1dimy));
					for(x = 0; x < Nx; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x));
					}
				}
				zm0 = (Nz - 2) % dimt;
				int startY = Ny-BOUNDARY_OFFSET;
				for(y = 0; y < BOUNDARY_OFFSET; ++ y)
				{
					out_center = &(E(outsubgrid, 0, y+startY, Nz-2));
					x_stream = &(ES(buf2, 0, y+1, zm0, t2dimx, t2dimy));
					y_m1_stream = &(ES(buf2, 0, y-1+1, zm0, t2dimx, t2dimy));
					y_a1_stream = &(ES(buf2, 0, y+1+1, zm0, t2dimx, t2dimy));
					z_m1_stream = &(ES(buf2, 0, y+1, (zm0-1+dimt)%dimt, t2dimx, t2dimy));
					z_a1_stream = &(ES(buf2, 0, y+1, (zm0+1)%dimt, t2dimx, t2dimy));
					for(x = 0; x < Nx; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x) + *(z_a1_stream+x));
					}
				}
				zm0 = (Nz - 1) % dimt;
				for(y = 0; y < BOUNDARY_OFFSET; ++ y)
				{
					out_center = &(E(outsubgrid, 0, y+startY, Nz-1));
					x_stream = &(ES(buf2, 0, y+1, zm0, t2dimx, t2dimy));
					y_m1_stream = &(ES(buf2, 0, y-1+1, zm0, t2dimx, t2dimy));
					y_a1_stream = &(ES(buf2, 0, y+1+1, zm0, t2dimx, t2dimy));
					z_m1_stream = &(ES(buf2, 0, y+1, (zm0-1+dimt)%dimt, t2dimx, t2dimy));
					for(x = 0; x < Nx; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x));
					}
				}
			
			}
		}
		
		#pragma omp section  // x from 0 to offset-1, y from 0 to Ny-1
		{
			int x, y, z;
			FT *x_stream, *y_m1_stream, *y_a1_stream, *z_m1_stream, *z_a1_stream, *out_center;
			int t1dimx = BOUNDARY_OFFSET+dimt, t1dimy = dimy + 2;
			int t2dimx = BOUNDARY_OFFSET+dimt-1, t2dimy = dimy + 2;
			int zm0, zm1, zm2;
			FT *buf1 = &(ES(bufferB3_1, 1, 1, 0, t1dimx, t1dimy));
			FT *buf2 = &(ES(bufferB3_2, 1, 1, 0, t2dimx, t2dimy));
			for(z = 0; z < Nz; ++z)
			{
				if(dimt == 1)
				{
					for(y = 0; y < Ny; ++ y)
					{
						out_center = &(E(outsubgrid, 0, y, z));
						x_stream = &(E(subgrid, 0, y, z));
						y_m1_stream = &(E(subgrid, 0, y-1, z));
						y_a1_stream = &(E(subgrid, 0, y+1, z));
						z_m1_stream = &(E(subgrid, 0, y, z-1));
						z_a1_stream = &(E(subgrid, 0, y, z+1));
						for(x = 0; x < BOUNDARY_OFFSET; ++ x)
						{
							*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
									+ *(y_m1_stream + x) + *(y_a1_stream + x)
									+ *(z_m1_stream + x) + *(z_a1_stream + x));
						}

					}

				}
				else
				{
					for(y = 0; y < Ny; ++ y)
					{
						out_center = &(ES(buf1, 0, y, z%dimt, t1dimx, t1dimy));
						x_stream = &(E(subgrid, 0, y, z));
						y_m1_stream = &(E(subgrid, 0, y-1, z));
						y_a1_stream = &(E(subgrid, 0, y+1, z));
						z_m1_stream = &(E(subgrid, 0, y, z-1));
						z_a1_stream = &(E(subgrid, 0, y, z+1));
						for(x = 0; x < BOUNDARY_OFFSET+dimt-1; ++ x)
						{
							*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
									+ *(y_m1_stream + x) + *(y_a1_stream + x)
									+ *(z_m1_stream + x) + *(z_a1_stream + x));
						}

					}
					if(z >= 1 && dimt >= 2)
					{
						for(y = 0; y < Ny; ++ y)
						{
							//y == 0 subplane
							zm1 = (z-1)%dimt;
							out_center = &(ES(buf2, 0, y, zm1, t2dimx, t2dimy));
							x_stream = &(ES(buf1, 0, y, zm1, t1dimx, t1dimy));
							y_m1_stream = &(ES(buf1, 0, y-1, zm1, t1dimx, t1dimy));
							y_a1_stream = &(ES(buf1, 0, y+1, zm1, t1dimx, t1dimy));
							z_m1_stream = &(ES(buf1, 0, y, (zm1-1+dimt)%dimt, t1dimx, t1dimy));
							z_a1_stream = &(ES(buf1, 0, y, (zm1+1)%dimt, t1dimx, t1dimy));
							for(x = 0; x < BOUNDARY_OFFSET+dimt-2; ++ x)
							{
								*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
										+ *(y_m1_stream + x) + *(y_a1_stream + x)
										+ *(z_m1_stream + x) + *(z_a1_stream + x));
							}
						}

					}
					if(z >= 2 && dimt == 3)
					{
						for(y = 0; y < Ny; ++ y)
						{
							//y == 0 subplane
							zm2 = (z-2)%dimt;
							out_center = &(E(outsubgrid, 0, y, z-2));
							x_stream = &(ES(buf2, 0, y, zm2, t2dimx, t2dimy));
							y_m1_stream = &(ES(buf2, 0, y-1, zm2, t2dimx, t2dimy));
							y_a1_stream = &(ES(buf2, 0, y+1, zm2, t2dimx, t2dimy));
							z_m1_stream = &(ES(buf2, 0, y, (zm2-1+dimt)%dimt, t2dimx, t2dimy));
							z_a1_stream = &(ES(buf2, 0, y, (zm2+1)%dimt, t2dimx, t2dimy));
							for(x = 0; x < BOUNDARY_OFFSET; ++ x)
							{
								*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
										+ *(y_m1_stream + x) + *(y_a1_stream + x)
										+ *(z_m1_stream + x) + *(z_a1_stream + x));
							}
						}

					}
				}

			}
			if(dimt == 3)
			{
				zm0 = (Nz - 1) % dimt;
				for(y = 0; y < Ny; ++ y)
				{
					out_center = &(ES(buf2, 0, y, zm0, t2dimx, t2dimy));
					x_stream = &(ES(buf1, 0, y, zm0, t1dimx, t1dimy));
					y_m1_stream = &(ES(buf1, 0, y-1, zm0, t1dimx, t1dimy));
					y_a1_stream = &(ES(buf1, 0, y+1, zm0, t1dimx, t1dimy));
					z_m1_stream = &(ES(buf1, 0, y, (zm0-1+dimt)%dimt, t1dimx, t1dimy));
					//z_a1_stream = &(ES(buf1, 0, y, (zm0+1)%dimt, t1dimx, t1dimy));
					for(x = 0; x < BOUNDARY_OFFSET+dimt-2; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x));
					}
				}
				zm0 = (Nz - 2) % dimt;
				for(y = 0; y < Ny; ++ y)
				{
					out_center = &(E(outsubgrid, 0, y, Nz-2));
					x_stream = &(ES(buf2, 0, y, zm0, t2dimx, t2dimy));
					y_m1_stream = &(ES(buf2, 0, y-1, zm0, t2dimx, t2dimy));
					y_a1_stream = &(ES(buf2, 0, y+1, zm0, t2dimx, t2dimy));
					z_m1_stream = &(ES(buf2, 0, y, (zm0-1+dimt)%dimt, t2dimx, t2dimy));
					z_a1_stream = &(ES(buf2, 0, y, (zm0+1)%dimt, t2dimx, t2dimy));
					for(x = 0; x < BOUNDARY_OFFSET; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x) + *(z_a1_stream+x));
					}
				}
				zm0 = (Nz - 1) % dimt;
				for(y = 0; y < Ny; ++ y)
				{
					out_center = &(E(outsubgrid, 0, y, Nz-1));
					x_stream = &(ES(buf2, 0, y, zm0, t2dimx, t2dimy));
					y_m1_stream = &(ES(buf2, 0, y-1, zm0, t2dimx, t2dimy));
					y_a1_stream = &(ES(buf2, 0, y+1, zm0, t2dimx, t2dimy));
					z_m1_stream = &(ES(buf2, 0, y, (zm0-1+dimt)%dimt, t2dimx, t2dimy));
					for(x = 0; x < BOUNDARY_OFFSET; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x));
					}
				}

			}
		}
		#pragma omp section  // x from Nx-offset to Nx-1, y from 0 to Ny-1
		{
			int x, y, z;
			FT *x_stream, *y_m1_stream, *y_a1_stream, *z_m1_stream, *z_a1_stream, *out_center;
			int t1dimx = BOUNDARY_OFFSET+dimt, t1dimy = dimy + 2;
			int t2dimx = BOUNDARY_OFFSET+dimt-1, t2dimy = dimy + 2;
			int zm0, zm1, zm2;
			FT *buf1 = &(ES(bufferB4_1, 0, 1, 0, t1dimx, t1dimy));
			FT *buf2 = &(ES(bufferB4_2, 0, 1, 0, t2dimx, t2dimy));
			for(z = 0; z < Nz; ++z)
			{
				if(dimt == 1)
				{
					for(y = 0; y < Ny; ++ y)
					{
						out_center = &(E(outsubgrid, 0, y, z));
						x_stream = &(E(subgrid, 0, y, z));
						y_m1_stream = &(E(subgrid, 0, y-1, z));
						y_a1_stream = &(E(subgrid, 0, y+1, z));
						z_m1_stream = &(E(subgrid, 0, y, z-1));
						z_a1_stream = &(E(subgrid, 0, y, z+1));
						for(x = Nx-BOUNDARY_OFFSET; x < Nx; ++ x)
						{
							*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
									+ *(y_m1_stream + x) + *(y_a1_stream + x)
									+ *(z_m1_stream + x) + *(z_a1_stream + x));
						}

					}

				}
				else
				{
					int startX = Nx-BOUNDARY_OFFSET-dimt+1;
					for(y = 0; y < Ny; ++ y)
					{
						out_center = &(ES(buf1, 0, y, z%dimt, t1dimx, t1dimy));
						x_stream = &(E(subgrid, startX, y, z));
						y_m1_stream = &(E(subgrid, startX, y-1, z));
						y_a1_stream = &(E(subgrid, startX, y+1, z));
						z_m1_stream = &(E(subgrid, startX, y, z-1));
						z_a1_stream = &(E(subgrid, startX, y, z+1));
						//for(x = Nx-BOUNDARY_OFFSET-dimt+1; x < Nx; ++ x)
						for(x = 0; x < BOUNDARY_OFFSET + dimt - 1; ++ x)
						{
							*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
									+ *(y_m1_stream + x) + *(y_a1_stream + x)
									+ *(z_m1_stream + x) + *(z_a1_stream + x));
						}

					}
					if(z >= 1 && dimt >= 2)
					{
						for(y = 0; y < Ny; ++ y)
						{
							//y == 0 subplane
							zm1 = (z-1)%dimt;
							out_center = &(ES(buf2, 0, y, zm1, t2dimx, t2dimy));
							x_stream = &(ES(buf1, 1, y, zm1, t1dimx, t1dimy));
							y_m1_stream = &(ES(buf1, 1, y-1, zm1, t1dimx, t1dimy));
							y_a1_stream = &(ES(buf1, 1, y+1, zm1, t1dimx, t1dimy));
							z_m1_stream = &(ES(buf1, 1, y, (zm1-1+dimt)%dimt, t1dimx, t2dimy));
							z_a1_stream = &(ES(buf1, 1, y, (zm1+1)%dimt, t1dimx, t2dimy));
							//for(x = Nx-BOUNDARY_OFFSET-dimt+2; x < Nx; ++ x)
							for(x = 0; x < BOUNDARY_OFFSET+dimt-2; ++ x)
							{
								*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
										+ *(y_m1_stream + x) + *(y_a1_stream + x)
										+ *(z_m1_stream + x) + *(z_a1_stream + x));
							}
						}

					}

					if(z >= 2 && dimt == 3)
					{
						startX = Nx-BOUNDARY_OFFSET;
						for(y = 0; y < Ny; ++ y)
						{
							//y == 0 subplane
							zm2 = (z-2)%dimt;
							out_center = &(E(outsubgrid, startX, y, z-2));
							x_stream = &(ES(buf2, 1, y, zm2, t2dimx, t2dimy));
							y_m1_stream = &(ES(buf2, 1, y-1, zm2, t2dimx, t2dimy));
							y_a1_stream = &(ES(buf2, 1, y+1, zm2, t2dimx, t2dimy));
							z_m1_stream = &(ES(buf2, 1, y, (zm2-1+dimt)%dimt, t2dimx, t2dimy));
							z_a1_stream = &(ES(buf2, 1, y, (zm2+1)%dimt, t2dimx, t2dimy));
							for(x = 0; x < BOUNDARY_OFFSET; ++ x)
							{
								*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1) 
										+ *(y_m1_stream + x) + *(y_a1_stream + x)
										+ *(z_m1_stream + x) + *(z_a1_stream + x));
							}
						}

					}
				}

			}
			if(dimt == 3)
			{
				zm0 = (Nz - 1) % dimt;
				for(y = 0; y < Ny; ++ y)
				{
					out_center = &(ES(buf2, 0, y, zm0, t2dimx, t2dimy));
					x_stream = &(ES(buf1, 1, y, zm0, t1dimx, t1dimy));
					y_m1_stream = &(ES(buf1, 1, y-1, zm0, t1dimx, t1dimy));
					y_a1_stream = &(ES(buf1, 1, y+1, zm0, t1dimx, t1dimy));
					z_m1_stream = &(ES(buf1, 1, y, (zm0-1+dimt)%dimt, t1dimx, t1dimy));
					//z_a1_stream = &(ES(buf1, 1, y, (zm0+1)%dimt, t1dimx, t1dimy));
					//for(x = Nx-BOUNDARY_OFFSET-dimt+2; x < Nx; ++ x)
					for(x = 0; x < BOUNDARY_OFFSET + dimt -2 ; ++x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x));
					}
				}
				zm0 = (Nz - 2) % dimt;
				int startX = Nx-BOUNDARY_OFFSET;
				for(y = 0; y < Ny; ++ y)
				{
					out_center = &(E(outsubgrid, startX, y, Nz-2));
					x_stream = &(ES(buf2, 1, y, zm0, t2dimx, t2dimy));
					y_m1_stream = &(ES(buf2, 1, y-1, zm0, t2dimx, t2dimy));
					y_a1_stream = &(ES(buf2, 1, y+1, zm0, t2dimx, t2dimy));
					z_m1_stream = &(ES(buf2, 1, y, (zm0-1+dimt)%dimt, t2dimx, t2dimy));
					z_a1_stream = &(ES(buf2, 1, y, (zm0+1)%dimt, t2dimx, t2dimy));
					//for(x = Nx-BOUNDARY_OFFSET; x < Nx; ++ x)
					for(x = 0; x < BOUNDARY_OFFSET; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x) + *(z_a1_stream+x));
					}
				}
				zm0 = (Nz - 1) % dimt;
				for(y = 0; y < Ny; ++ y)
				{
					out_center = &(E(outsubgrid, startX, y, Nz-1));
					x_stream = &(ES(buf2, 1, y, zm0, t2dimx, t2dimy));
					y_m1_stream = &(ES(buf2, 1, y-1, zm0, t2dimx, t2dimy));
					y_a1_stream = &(ES(buf2, 1, y+1, zm0, t2dimx, t2dimy));
					z_m1_stream = &(ES(buf2, 1, y, (zm0-1+dimt)%dimt, t2dimx, t2dimy));
					for(x = 0; x < BOUNDARY_OFFSET; ++ x)
					{
						*(out_center+x) = CENTER_WIGHT*(*(x_stream+x)) + ADJACENT_WIGHT * ( *(x_stream+x-1) + *(x_stream+x+1)
							+ *(y_m1_stream+x) + *(y_a1_stream+x)
							+ *(z_m1_stream+x));
					}
				}

			}
		}
	}
	}


}

void compute_stencil_iter(grid_data* pgd, int dimt, grid_data* out)
{
	int xw = (pgd->Nx - 2*BOUNDARY_OFFSET + DIMX - 1) / DIMX, rx = (pgd->Nx - 2*BOUNDARY_OFFSET) % DIMX;
	int yw = (pgd->Ny - 2*BOUNDARY_OFFSET + DIMY - 1) / DIMY, ry = (pgd->Ny - 2*BOUNDARY_OFFSET) % DIMY;
	int xwi, ywi, dimx, dimy;
	grid_data subgrid;
	subgrid.Nx = pgd->Nx;
	subgrid.Ny = pgd->Ny;
	subgrid.Nz = pgd->Nz;
	subgrid.xstride = pgd->xstride;
	subgrid.ystride = pgd->ystride;
	subgrid.zstride = pgd->zstride;
	subgrid.NIter = pgd->NIter;

	grid_data outsubgrid;
	outsubgrid.Nx = pgd->Nx;
	outsubgrid.Ny = pgd->Ny;
	outsubgrid.Nz = pgd->Nz;
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
			subgrid.data = &(E(pgd, xwi*DIMX+1+BOUNDARY_OFFSET, ywi*DIMY+1+BOUNDARY_OFFSET, 1));
			outsubgrid.data = &(E(out, xwi*DIMX+1+BOUNDARY_OFFSET, ywi*DIMY+1+BOUNDARY_OFFSET, 1));
			//compute the kernel of subplane
			compute_subplane(pgd, &subgrid, dimx, dimy, dimt, &outsubgrid);
		}
	}

	//compute the boundary
	subgrid.data = &(E(pgd, 1, 1, 1));
	outsubgrid.data = &(E(out, 1, 1, 1));
	compute_boundary(pgd, &subgrid, pgd->Nx, pgd->Ny, dimt, &outsubgrid);
}

double sum_grid(grid_data* pgd)
{
	double res = 0.0;
	size_t size = pgd->xstride * pgd->ystride * pgd->zstride;
	for(size_t i = 0; i < size; ++i)
		res += (pgd->data)[i];
	return res;
}

double checkSum(PFT data, size_t Nx, size_t Ny, size_t Nz, size_t z)
{
	double res = 0.0;
	for(size_t y = 0; y < Ny; ++y)
		for(size_t x = 0; x < Nx; ++x)
			res += ES(data, x, y, z, Nx, Ny);
	return res;
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
	gd.data = (PFT)_mm_malloc(gd.xstride * gd.ystride * gd.zstride * sizeof(FT), 64);

	grid_data gd2;
	gd2.Nx = Nx;
	gd2.Ny = Ny;
	gd2.Nz = Nz;
	gd2.NIter = NIter;
	gd2.xstride = Nx + 2;
	gd2.ystride = Ny + 2;
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
	double tc1 = dsecnd();

	for(i = 0; i < itb; ++i)
	{
		dimt = (i != itb - 1 || !rit) ? DIMT : rit;
		if(i % 2 == 0)
		{
			compute_stencil_iter(&gd, dimt, &gd2);
	//		printf("sumgrid : %.5f \n", sum_grid(&gd2));
		}
		else
		{
			compute_stencil_iter(&gd2, dimt, &gd);
	//		printf("sumgrid : %.5f \n", sum_grid(&gd));
		}

	}
	double tc2 = dsecnd();
	printf("time : %.5f seconds\n", tc2-tc1);


#ifdef PRINT_FINISH_DEBUG
	grid_data* pgd = &gd2;
	if(i % 2 == 0)
	{
		pgd = &gd;
	}
	//for(int j = 0; j < 10 && j < Nz; ++j)
		//printf("checkSum layer %d and %d : %.5f, %.5f\n", j, Nz+1-j, checkSum(pgd->data, pgd->xstride, pgd->ystride, pgd->zstride, j), checkSum(pgd->data, pgd->xstride, pgd->ystride, pgd->zstride, Nz+1-j));
	for(int j = 0; j < Nz+2; j++)
		printf("j %d, %.5f\n", j, checkSum(pgd->data, pgd->xstride, pgd->ystride, pgd->zstride, j));

	/*for(int z = 0; z < Nz; ++z)
	{
		printf("z : %d\n", z);
		for(int y = 0; y < Ny; ++y)
		{
			for(int x = 0; x < Nx; ++x)
			{
				printf("%.1f ", E(pgd, x+1, y+1,z+1));
			}
			printf("\n");
		}
		printf("\n");
	}*/
#endif


	_mm_free(gd.data);
	_mm_free(gd2.data);
}

int main(int argc, char** argv)
{

	int Nx = 512, Ny = 512, Nz = 512, n = 100;
	bufferB1_1 = (PFT)_mm_malloc((Nx+2)*(BOUNDARY_OFFSET+DIMT)*3*8, 64);
	bufferB1_2 = (PFT)_mm_malloc((Nx+2)*(BOUNDARY_OFFSET+DIMT-1)*3*8, 64);
	bufferB2_1 = (PFT)_mm_malloc((Nx+2)*(BOUNDARY_OFFSET+DIMT)*3*8, 64);
	bufferB2_2 = (PFT)_mm_malloc((Nx+2)*(BOUNDARY_OFFSET+DIMT-1)*3*8, 64);
	bufferB3_1 = (PFT)_mm_malloc((BOUNDARY_OFFSET+DIMT)*(Ny+2)*3*8, 64);
	bufferB3_2 = (PFT)_mm_malloc((BOUNDARY_OFFSET+DIMT-1)*(Ny+2)*3*8, 64);
	bufferB4_1 = (PFT)_mm_malloc((BOUNDARY_OFFSET+DIMT)*(Ny+2)*3*8, 64);
	bufferB4_2 = (PFT)_mm_malloc((BOUNDARY_OFFSET+DIMT-1)*(Ny+2)*3*8, 64);
	memset(bufferB1_1, 0, (Nx+2)*(BOUNDARY_OFFSET+DIMT)*3*8);
	memset(bufferB1_2, 0, (Nx+2)*(BOUNDARY_OFFSET+DIMT-1)*3*8);
	memset(bufferB2_1, 0, (Nx+2)*(BOUNDARY_OFFSET+DIMT)*3*8);
	memset(bufferB2_2, 0, (Nx+2)*(BOUNDARY_OFFSET+DIMT-1)*3*8);
	memset(bufferB3_1, 0, (BOUNDARY_OFFSET+DIMT)*(Ny+2)*3*8);
	memset(bufferB3_2, 0, (BOUNDARY_OFFSET+DIMT-1)*(Ny+2)*3*8);
	memset(bufferB4_1, 0, (BOUNDARY_OFFSET+DIMT)*(Ny+2)*3*8);
	memset(bufferB4_2, 0, (BOUNDARY_OFFSET+DIMT-1)*(Ny+2)*3*8);

	compute_stencil_3_5D(Nx, Ny, Nz, n);

	return 0;
}
