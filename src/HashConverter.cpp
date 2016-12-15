#include "HashConverter.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "hashConverterKernel.h"

HashConverter::HashConverter(){
	//Allocate Space on device
	d_projMatHamming.width = kDimSiftData;
	d_projMatHamming.height = kDimHashData;
	cudaMallocPitch((void**) &d_projMatHamming.elements, &d_projMatHamming.pitch, d_projMatHamming.width*sizeof(int), d_projMatHamming.height*sizeof(int));
	// TODO: Compute Matrix projMatHamming & projMatBucket
	fillwithrand(d_projMatHamming);
}

HashConverter::~HashConverter(){ 
	cudaFree(d_projMatHamming.elements);
}

void HashConverter::SiftDataToHashData(ImageDataDevice *imgD){

}

