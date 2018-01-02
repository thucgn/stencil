/*************************************************************************
    > File Name: matmul_cuda.cpp
    > Author: cgn
    > Func: 
    > Created Time: 2017年12月23日 星期六 06时40分16秒
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

typedef double FT;
typedef double* PFT;

//#define PRINT_DEBUG
#define CENTER_WEIGTH 1.0
#define ADJACENT_WEIGTH 0.1
#define BW 16
#define BH 16
#define E(data, x, y, z, xstride, ystride) data[((z)*(ystride)+y)*(xstride)+x]

inline void checkCudaErrors(cudaError err)
{
	if(cudaSuccess != err)
	{
		printf("error!\n");
	}
}

__global__ void stencil_7_point(int Nx, int Ny, int Nz, PFT in_data, PFT out_data)
{
	register const int ty = threadIdx.y, tx = threadIdx.x;
	register const int rtx = tx+1, rty = ty+1;
	register const int by = blockIdx.y, bx = blockIdx.x, bz = blockIdx.z;
	register const int x = bx*BW+tx, y = by*BW+ty, z = bz*BH;
	register int i, zn, zc, za;

	__shared__ FT buf[3][BW+2][BW+2];

	if(x < Nx && y < Ny)
	{
		//load ghost layer
		buf[0][rty][rtx] = (z != 0) ? E(in_data, x, y, z-1, Nx, Ny) : 0.0;
		//load first layer
		buf[1][rty][rtx] = E(in_data, x, y, z, Nx, Ny);
		//load xyboundry
		if(tx == 0)
			buf[1][rty][0] = (x != 0) ? E(in_data, x-1, y, z, Nx, Ny) : 0.0;
		else if(tx == BW-1)
			buf[1][rty][BW+1] = (x != Nx-1) ? E(in_data, x+1, y, z, Nx, Ny) : 0.0;
		else if(x == Nx-1)
			buf[1][rty][rtx+1] = 0.0;

		if(ty == 0)
			buf[1][0][rtx] = (y != 0) ? E(in_data, x, y-1, z, Nx, Ny) : 0.0;
		else if(ty == BW-1)
			buf[1][BW+1][rtx] = (y != Ny-1) ? E(in_data, x, y+1, z, Nx, Ny) : 0.0;
		else if(ty == Ny-1)
			buf[1][rty+1][rtx] = 0.0;

		__syncthreads();

		for(i = 1; i <= BH && (z+i) <= Nz; ++i)
		{
			zc = i % 3;
			za = (i+1) % 3;
			zn = z + i;
			if(zn < Nz)
			{
				buf[za][rty][rtx] = E(in_data, x, y, zn, Nx, Ny);
				if(tx == 0)
					buf[za][rty][0] = (x != 0) ? E(in_data, x-1, y, zn, Nx, Ny) : 0.0;
				else if(tx == BW-1)
					buf[za][rty][BW+1] = (x != Nx-1) ? E(in_data, x+1, y, zn, Nx, Ny) : 0.0;
				else if(x == Nx-1)
					buf[za][rty][rtx+1] = 0.0;

				if(ty == 0)
					buf[za][0][rtx] = (y != 0) ? E(in_data, x, y-1, zn, Nx, Ny) : 0.0;
				else if(ty == BW-1)
					buf[za][BW+1][rtx] = (y != Ny-1) ? E(in_data, x, y+1, zn, Nx, Ny) : 0.0;
				else if(y == Ny-1)
					buf[za][rty+1][rtx] = 0.0;
			}
			else
			{
				int j, k;
				for(j = 0; j < (BW+2); ++j)
					for(k = 0; k < (BW+2); ++k)
						buf[za][j][k] = 0.0;

			}

			__syncthreads();
			E(out_data, x, y, zn-1, Nx, Ny) = CENTER_WEIGTH * buf[zc][rty][rtx]
				+ ADJACENT_WEIGTH * ( buf[zc][rty][rtx+1] + buf[zc][rty][rtx-1]
				+ buf[zc][rty-1][rtx] + buf[zc][rty+1][rtx]
				+ buf[za][rty][rtx] + buf[(i-1)%3][rty][rtx]);
			__syncthreads();

		}
	}

}

void init_data(PFT data, size_t Nx, size_t Ny, size_t Nz)
{
	memset(data, 0, (Nx)*(Ny)*(Nz)*sizeof(FT));
	srand((unsigned int)time(NULL));
	for(size_t z = 0; z < Nz; ++z)
	{
		for(size_t y = 0; y < Ny; ++y)
			for(size_t x = 0; x < Nx; ++x)
			{
				data[(z*(Nx)+y) * (Nx) + x] = 1.0;
			}
	}
}

void print_data(PFT data, size_t Nx, size_t Ny, size_t Nz)
{
	for(size_t z = 0; z < Nz; ++z)
	{
		printf("\n\nz : %d\n", z);
		for(size_t y = 0; y < Ny; ++y)
		{
			for(size_t x = 0; x < Nx; ++x)
			{
				printf("%.1f ", E(data, x,y,z, Nx, Ny));
			}
			printf("\n");
		}
		
	}
}

double checkSum(PFT data, size_t Nx, size_t Ny, size_t Nz, size_t z)
{
	double res = 0.0;
	for(size_t y = 0; y < Ny; ++y)
		for(size_t x = 0; x < Nx; ++x)
			res += E(data, x, y, z, Nx, Ny);
	return res;
}

void stencil_gpu(size_t Nx, size_t Ny, size_t Nz, size_t NIter)
{
	printf("begin stencil!\n");	
	size_t memsize = Nx * Ny * Nz * sizeof(FT);
	int i;

	PFT h_data = (PFT)malloc(memsize);
	
	init_data(h_data, Nx, Ny, Nz);

#ifdef PRINT_DEBUG
	print_data(h_data, Nx, Ny, Nz);
#endif

	printf("finish data initialization!\n");
	PFT d1_data;
	PFT d2_data;

	checkCudaErrors(cudaMalloc((void**)&d1_data, memsize));
	checkCudaErrors(cudaMalloc((void**)&d2_data, memsize));
	printf("finish cuda malloc!\n");

	
	checkCudaErrors(cudaMemcpy(d1_data, h_data, memsize, cudaMemcpyHostToDevice));
	printf("finish cuda memcpy!\n");

	dim3 threads(BW, BW);
	dim3 grid((Nx+BW-1)/BW, (Ny+BW-1)/BW, (Nz+BH-1)/BH);

	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start)); 
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	for(i = 0; i < NIter; ++i)
	{
		if(i % 2 == 0)
		{
			//printf("begin calculate\n");
			stencil_7_point<<<grid, threads>>>(Nx, Ny, Nz, d1_data, d2_data);
		}
		else
		{
			//printf("begin calculate\n");
			stencil_7_point<<<grid, threads>>>(Nx, Ny, Nz, d2_data, d1_data);
		}
	}
	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


	if(i % 2 == 1)
		checkCudaErrors(cudaMemcpy(h_data, d2_data, memsize, cudaMemcpyDeviceToHost));
	else
		checkCudaErrors(cudaMemcpy(h_data, d1_data, memsize, cudaMemcpyDeviceToHost));
#ifdef PRINT_DEBUG
	print_data(h_data, Nx, Ny, Nz);
#endif
	for(int i = 0;i < 10 && i < Nz; ++i)
	{
		printf("checksum layer %d and %d : %.5f , %.5f\n ", i, Nz-1-i, checkSum(h_data, Nx, Ny, Nz, i), checkSum(h_data, Nx, Ny, Nz, Nz-1-i));
	}
	printf("==============time : %.5f ms\n", msecTotal);

	checkCudaErrors(cudaFree(d1_data));
	checkCudaErrors(cudaFree(d2_data));

	free(h_data);
}

int main()
{
	size_t Nx = 512, Ny = 512, Nz = 512, NIter = 100;

	stencil_gpu(Nx, Ny, Nz, NIter);

	return 0;
}
