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
#define CENTER_WIGHT 2.0
#define ADJACENT_WIGHT 1.1
#define E(data, x, y, z, xstride, ystride) data[((z)*(ystride)+(y))*(xstride)+x]
#define PE(data, x, y, xstride) data[(y)*(xstride) + (x)]
#define PLANE(data, z, xstride, ystride) data[(z)*(xstride)*(ystride)]
#define BLOCK_WIDTH 16

inline void checkCudaErrors(cudaError err)
{
	if(cudaSuccess != err)
	{
		printf("error!\n");
	}
}

__global__ void stencil_7_point(int Nx, int Ny, int Nz, PFT data, PFT out_data)
{

	const int X_EDGE_BLOCKS = Nx/BLOCK_WIDTH;
	const int Y_EDGE_BLOCKS = Ny/BLOCK_WIDTH;
	const int X_STRIDE = Nx+2;
	const int Y_STRIDE = Ny+2;
	register const int ty = threadIdx.y, tx = threadIdx.x;
	register const int rty = ty+1, rtx = tx+1;
	register const int by = blockIdx.y, bx = blockIdx.x;
	register const int gy = by*BLOCK_WIDTH + ty, gx = bx*BLOCK_WIDTH + tx;
	register int i, im, ic, ia;
	register PFT plane;
	register const PFT out_origin = &(E(out_data, 1, 1, 0, X_STRIDE, Y_STRIDE));

	__shared__ FT buf[3][BLOCK_WIDTH+2][BLOCK_WIDTH+2];

	
	if((by<Y_EDGE_BLOCKS-1 && bx<X_EDGE_BLOCKS-1) || (Nx % BLOCK_WIDTH == 0 && Ny % BLOCK_WIDTH == 0))
	{
		//first layer
		plane = &(E(data, 1, 1, 0, X_STRIDE, Y_STRIDE));
		buf[0][rty][rtx] = PE(plane, gx, gy, X_STRIDE);
		if(ty == 0)
			buf[0][0][rtx] = PE(plane, gx, gy-1, X_STRIDE);
		else if(ty == BLOCK_WIDTH-1)
			buf[0][BLOCK_WIDTH+1][rtx] = PE(plane, gx, gy+1, X_STRIDE);

		if(tx == 0)
			buf[0][rty][0] = PE(plane, gx-1, gy, X_STRIDE);
		else if(tx == BLOCK_WIDTH-1)
			buf[0][rty][BLOCK_WIDTH+1] = PE(plane, gx+1, gy, X_STRIDE);

		__syncthreads();

		//second layer
		plane = &(E(data, 1, 1, 1, X_STRIDE, Y_STRIDE));
		buf[1][rty][rtx] = PE(plane, gx, gy, X_STRIDE);
		if(ty == 0)
			buf[1][0][rtx] = PE(plane, gx, gy-1, X_STRIDE);
		else if(ty == BLOCK_WIDTH-1)
			buf[1][BLOCK_WIDTH+1][rtx] = PE(plane, gx, gy+1, X_STRIDE);

		if(tx == 0)
			buf[1][rty][0] = PE(plane, gx-1, gy, X_STRIDE);
		else if(tx == BLOCK_WIDTH-1)
			buf[1][rty][BLOCK_WIDTH+1] = PE(plane, gx+1, gy, X_STRIDE);

		__syncthreads();

		for(i = 2; i < Nz+2; ++i)
		{
			ia = i % 3;
			//ith layer
			plane = &(E(data, 1, 1, i, X_STRIDE, Y_STRIDE));
			buf[ia][rty][rtx] = PE(plane, gx, gy, X_STRIDE);
			if(ty == 0)
				buf[ia][0][rtx] = PE(plane, gx, gy-1, X_STRIDE);
			else if(ty == BLOCK_WIDTH-1)
				buf[ia][BLOCK_WIDTH+1][rtx] = PE(plane, gx, gy+1, X_STRIDE);

			if(tx == 0)
				buf[ia][rty][0] = PE(plane, gx-1, gy, X_STRIDE);
			else if(tx == BLOCK_WIDTH-1)
				buf[ia][rty][BLOCK_WIDTH+1] = PE(plane, gx+1, gy, X_STRIDE);

			__syncthreads();

			//calculate i-1 layer
			ic = (i-1) % 3;
			im = (i-2) % 3;
			E(out_origin, gx, gy, i-1, X_STRIDE, Y_STRIDE) = CENTER_WIGHT * buf[ic][rty][rtx]
				+ ADJACENT_WIGHT*( buf[ic][rty][tx] + buf[ic][rty][rtx+1] 
				+ buf[ic][ty][rtx] + buf[ic][rty+1][rtx]
				+ buf[im][rty][rtx] + buf[ia][rty][rtx]
				);
		}

	}
	else
	{
		//first layer
		if(gx > Nx-1 || gy > Ny-1)
			return;
		plane = &(E(data, 1, 1, 0, X_STRIDE, Y_STRIDE));
		buf[0][rty][rtx] = PE(plane, gx, gy, X_STRIDE);
		if(ty == 0)
			buf[0][0][rtx] = PE(plane, gx, gy-1, X_STRIDE);
		else if(gy == Ny-1)
			buf[0][rty+1][rtx] = 0.0;

		if(tx == 0)
			buf[0][rty][0] = PE(plane, gx-1, gy, X_STRIDE);
		else if(gx == Nx-1)
			buf[0][rty][rtx+1] = 0.0;

		//second layer
		plane = &(E(data, 1, 1, 1, X_STRIDE, Y_STRIDE));
		buf[1][rty][rtx] = PE(plane, gx, gy, X_STRIDE);
		if(ty == 0)
			buf[1][0][rtx] = PE(plane, gx, gy-1, X_STRIDE);
		else if(gy == Ny-1)
			buf[1][rty+1][rtx] = 0.0;

		if(tx == 0)
			buf[1][rty][0] = PE(plane, gx-1, gy, X_STRIDE);
		else if(gx == Nx-1)
			buf[1][rty][rtx+1] = 0.0;


		for(i = 2; i < Nz+2; ++i)
		{
			ia = i % 3;
			//ith layer
			plane = &(E(data, 1, 1, i, X_STRIDE, Y_STRIDE));
			buf[ia][rty][rtx] = PE(plane, gx, gy, X_STRIDE);
			if(ty == 0)
				buf[ia][0][rtx] = PE(plane, gx, gy-1, X_STRIDE);
			else if(gy == Ny-1)
				buf[ia][rty+1][rtx] = 0.0;

			if(tx == 0)
				buf[ia][rty][0] = PE(plane, gx-1, gy, X_STRIDE);
			else if(gx == Nx-1)
				buf[ia][rty][rtx+1] = 0.0;

			__syncthreads();

			//calculate i-1 layer
			ic = (i-1) % 3;
			im = (i-2) % 3;
			E(out_origin, gx, gy, i-1, X_STRIDE, Y_STRIDE) = CENTER_WIGHT * buf[ic][rty][rtx]
				+ ADJACENT_WIGHT*( buf[ic][rty][tx] + buf[ic][rty][rtx+1] 
				+ buf[ic][ty][rtx] + buf[ic][rty+1][rtx]
				+ buf[im][rty][rtx] + buf[ia][rty][rtx]
				);
		}

	}
}

void init_data(PFT data, size_t Nx, size_t Ny, size_t Nz)
{
	memset(data, 0, (Nx+2)*(Ny+2)*(Nz+2)*sizeof(FT));
	srand((unsigned int)time(NULL));
	for(size_t z = 1; z < Nz+1; ++z)
	{
		for(size_t y = 1; y < Ny+1; ++y)
			for(size_t x = 1; x < Nx+1; ++x)
			{
				data[(z*(Ny+2)+y) * (Nx+2) + x] = 1.0;
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

void stencil_gpu(size_t Nx, size_t Ny, size_t Nz, size_t NIter)
{
	printf("begin stencil!\n");	
	size_t memsize = (Nx+2) * (Ny+2) * (Nz+2) * sizeof(FT);
	int i;

	PFT h_data = (PFT)malloc(memsize);
	
	init_data(h_data, Nx, Ny, Nz);

#ifdef PRINT_DEBUG
	//print_data(h_data, Nx+2, Ny+2, Nz+2);
#endif

	printf("finish data initialization!\n");
	PFT d1_data;
	PFT d2_data;

	checkCudaErrors(cudaMalloc((void**)&d1_data, memsize));
	checkCudaErrors(cudaMalloc((void**)&d2_data, memsize));
	printf("finish cuda malloc!\n");

	
	checkCudaErrors(cudaMemcpy(d1_data, h_data, memsize, cudaMemcpyHostToDevice));
	printf("finish cuda memcpy!\n");

	dim3 threads(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 grid((Nx+BLOCK_WIDTH-1)/BLOCK_WIDTH, (Ny+BLOCK_WIDTH-1)/BLOCK_WIDTH);

	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start)); 
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	for(i = 0; i < NIter; ++i)
	{
		if(i % 2 == 0)
		{
			stencil_7_point<<<grid, threads>>>(Nx, Ny, Nz, d1_data, d2_data);
		}
		else
		{
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
	print_data(h_data, Nx+2, Ny+2, Nz+2);
#endif
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
