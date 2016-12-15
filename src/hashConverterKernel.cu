#include <cuda_runtime.h>
#include <stdio.h>
#include "hashConverterKernel.h"
#include "curand_kernel.h"

#define BLOCK_SIZE 32

void fillwithrand(Matrix &pMat){

	dim3 block_size(BLOCK_SIZE,BLOCK_SIZE);
	dim3 grid_size(pMat.width/block_size.x, pMat.height/block_size.y); //TODO choose right grid size if pMat.width%block_size != 0
	
	randNumKernel<<<grid_size,block_size>>>(pMat.elements, pMat.pitch);
}

__global__ void randNumkernel(int* elements, const size_t pitch){
	int c_i = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	int r_i = threadIdx.y + BLOCK_SIZE*blockIdx.y;
	
	((int*)((char*)elements + r_i * pitch))[c_i] = 1; //TODO = random 
}
