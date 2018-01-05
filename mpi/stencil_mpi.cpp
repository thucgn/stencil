/*************************************************************************
    > File Name: stencil_mpi.cpp
    > Author: cgn
    > Func: 
    > Created Time: 三  1/ 3 22:40:04 2018
 ************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

typedef double FT;
typedef double* PFT;

#define PRINT_DEBUG
//define INTEL_VAR
#define CENTER_WEIGHT 1.0
#define ADJACENT_WEIGHT 0.1
#define MASTER 0
#define DATA_TAG 1
#define GATHER_TAG 2

#define E(data, x, y, z, xstride, ystride) data[((z)*(ystride)+y) * (xstride) + x]


void init_data(PFT data, size_t Nx, size_t Ny, size_t Nz)
{
	size_t z, y, x;
	for(z = 1; z <= Nz; ++z)
		for(y = 1; y <= Ny; ++y)
			for(x = 1; x <= Nx; ++x)
				E(data, x, y, z, Nx+2, Ny+2) = 1.0;
}

void stencil_7_point(PFT input_data, PFT output_data, int height, int Nx, int Ny)
{
	//in_data and out_data start from x=y=z=0, not the origin of ghost layer
	FT *out_center, *xstream, *ymstream, *yastream, *zmstream, *zastream;
	int x, y, z, xstride = Nx+2, ystride = Ny+2;
	PFT in_data = &(E(input_data, 1,1,1, xstride, ystride));
	PFT out_data = &(E(output_data, 1,1,1, xstride, ystride));
	for(z = 0; z < height; ++z)
	{
		for(y = 0; y < Ny; ++y)
		{

			out_center = &(E(out_data, 0, y, z, xstride, ystride));
			xstream    = &(E(in_data,  0, y, z, xstride, ystride));
			ymstream   = &(E(in_data, 0, y-1, z, xstride, ystride));
			yastream   = &(E(in_data, 0, y+1, z, xstride, ystride));
			zmstream   = &(E(in_data, 0, y, z-1, xstride, ystride));
			zastream   = &(E(in_data, 0, y, z+1, xstride, ystride));
			for(x = 0; x < Nx; ++ x)
			{
				*(out_center+x) = CENTER_WEIGHT * ( *(xstream+x) ) + ADJACENT_WEIGHT * (
						*(xstream+x-1) + *(xstream+x+1) + 
						*(ymstream+x)  + *(yastream+x)  +
						*(zmstream+x)  + *(zastream+x) );
			}
		}
	}
}

void print_data(PFT data, int Nx, int Ny, int Nz, size_t xstride, size_t ystride)
{
	size_t x, y, z;
	for(z = 0; z < (size_t)Nz; ++z)
	{
		printf("\nz : %ld\n", z);
		for(y = 0; y < (size_t)Ny; ++y)
		{
			for(x = 0; x < (size_t)Nx; ++x)
			{
				printf("%.3f ",E(data, x+1, y+1, z+1, xstride, ystride));
			}
			printf("\n");
		}
	}
}

int main(int argc, char* argv[])
{

	int Nx = 10, Ny = 10, Nz = 10, NIter = 1;
	size_t xstride = Nx+2, ystride = Ny+2, zstride = Nz+2;
	int rank, size;
	int i, j, BH, LH;
	size_t memsize = xstride * ystride * zstride * sizeof(FT);
	PFT data = NULL;
	PFT data1 = NULL;
	PFT data2 = NULL;
	PFT pd1 = NULL;
	PFT pd2 = NULL;
	PFT res_data = NULL;
	PFT tmp = NULL;

	MPI_Status status;
	MPI_Request request; 
	MPI_Init(&argc, &argv);	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	BH = Nz/(size-1), LH = Nz - BH * (size-1);
	int height;

	if(rank == MASTER) // master
	{
#ifdef INTEL_VAR
		data = (PFT)_mm_malloc(memsize, 32);
		data2 = (PFT)_mm_malloc(xstride*ystride*(BH+2)*sizeof(FT), 32);
#else
		data = (PFT)malloc(memsize);
		data2 = (PFT)malloc(xstride*ystride*(BH+2)*sizeof(FT));
#endif
		memset(data, 0, memsize);
		memset(data2, 0, memsize);
		init_data(data, (size_t)Nx, (size_t)Ny, (size_t)Nz);
		
		for(i = 1; i < size; ++i)
		{
			height = (i != size-1) ? BH : LH;
			MPI_Send(&(E(data, 0, 0, i*BH, xstride, ystride)), (height+2)*xstride*ystride, MPI_DOUBLE, i, DATA_TAG, MPI_COMM_WORLD);
		}
		data1 = data;
		pd1 = data1;
		pd2 = data2;
		height = BH;
		stencil_7_point(pd1, pd2, height, Nx, Ny);

		for(int iter = 0; iter < NIter-1; ++ iter)
		{
			//send the top layer exactly under the ghost layer
			MPI_Isend(&(E(pd2, 0, 0, height, xstride, ystride)), xstride*ystride, MPI_DOUBLE, MASTER+1, DATA_TAG, MPI_COMM_WORLD, &request);
			//recv the ghost layer
			MPI_Recv(&(E(pd2, 0, 0, height+1, xstride, ystride)), xstride*ystride, MPI_DOUBLE, MASTER+1, DATA_TAG, MPI_COMM_WORLD, &status);
			MPI_Wait(&request, MPI_STATUS_IGNORE);
			
			//swap
			tmp = pd1;
			pd2 = pd1;
			pd1 = tmp;
			stencil_7_point(pd1, pd2, height, Nx, Ny);
		}

		memcpy(&(E(data, 0,0,1, xstride, ystride)), &(E(pd2, 0,0,1, xstride, ystride)), BH*xstride*ystride*sizeof(FT));
		//gather all the results
		for(i = 1; i < size; ++i)
		{
			height = (i != size-1) ? BH : LH;
			MPI_Recv(&(E(data, 0, 0, i*BH+1, xstride, ystride)), height*xstride*ystride, MPI_DOUBLE, i, GATHER_TAG, MPI_COMM_WORLD, &status);
			printf("Finish gather data from process %d, size : %ld\n", i, height*xstride*ystride);
		}

#ifdef PRINT_DEBUG
		print_data(data, Nx, Ny, Nz, xstride, ystride);
#endif

#ifdef INTEL_VAR
		_mm_free(data);
		_mm_free(data2);
#else
		free(data);
		free(data2);
#endif

	}
	else
	{
		height = (rank != size-1) ? BH : LH;
		size_t blocksize = xstride*ystride*(height+2);
#ifdef INTE_VAR
		data1 = (PFT)_mm_malloc(blocksize * sizeof(FT), 32);
		data2 = (PFT)_mm_malloc(blocksize * sizeof(FT), 32);
#else
		data1 = (PFT)malloc(blocksize * sizeof(FT));
		data2 = (PFT)malloc(blocksize * sizeof(FT));
#endif
		memset(data1, 0, blocksize * sizeof(FT));
		memset(data2, 0, blocksize * sizeof(FT));

		pd1 = data1;
		pd2 = data2;
		MPI_Recv(pd1, blocksize, MPI_DOUBLE, MASTER, DATA_TAG, MPI_COMM_WORLD, &status);
		stencil_7_point(pd1, pd2, height, Nx, Ny);

		for(int iter = 0; iter < NIter-1; ++ iter)
		{
			if(rank != size-1) // the top block no need to exchange the top ghost layer
			{
				//send the top layer exactly under the ghost layer
				MPI_Isend(&(E(pd2, 0, 0, height, xstride, ystride)), xstride*ystride, MPI_DOUBLE, rank+1, DATA_TAG, MPI_COMM_WORLD, &request);
				//recv the ghost layer
				MPI_Recv(&(E(pd2, 0, 0, height+1, xstride, ystride)), xstride*ystride, MPI_DOUBLE, rank+1, DATA_TAG, MPI_COMM_WORLD, &status);
				MPI_Wait(&request, MPI_STATUS_IGNORE);
			}

			//send the bottom layer exactly above the ghost layer
			MPI_Isend(&(E(pd2, 0, 0, 1, xstride, ystride)), xstride*ystride, MPI_DOUBLE, rank-1, DATA_TAG, MPI_COMM_WORLD, &request);
			MPI_Recv(&(E(pd2, 0, 0, 0, xstride, ystride)), xstride*ystride, MPI_DOUBLE, rank-1, DATA_TAG, MPI_COMM_WORLD, &status);
			MPI_Wait(&request, MPI_STATUS_IGNORE);

			//swap
			tmp = pd1;
			pd2 = pd1;
			pd1 = tmp;

			stencil_7_point(pd1, pd2, height, Nx, Ny);

		}

		//send to master
		MPI_Send(&(E(pd2, 0, 0, 1, xstride, ystride)), height*xstride*ystride, MPI_DOUBLE, MASTER, GATHER_TAG, MPI_COMM_WORLD);
#ifdef INTEL_VAR
		_mm_free(data1);
		_mm_free(data2);
#else
		free(data1);
		free(data2);
#endif

	}

	MPI_Finalize();


	return 0;
}
