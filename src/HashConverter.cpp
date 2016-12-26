#include "HashConverter.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <iostream>

HashConverter::HashConverter(){
	//Allocate matrix for hashing into 128d-Hamming space
	d_projMatHamming_.width = kDimSiftData;
	d_projMatHamming_.height = kDimHashData;
	cudaMallocPitch(&(d_projMatHamming_.elements),
                  &(d_projMatHamming_.pitch),
                  d_projMatHamming_.width * sizeof(SiftData_t),
                  d_projMatHamming_.height);
  CUDA_CHECK_ERROR;

  ////Allocate for hashing into 8bit bucket groups
  //cudaPitchedPtr matBase;
  //cudaExtent matExt;

  //matExt.width = kDimSiftData * sizeof(SiftData_t);
  //matExt.height = kCntBucketBit;
  //matExt.depth = kCntBucketGroup;

  //cudaMalloc3D(&matBase, matExt);
  //CUDA_CHECK_ERROR;

  //// Map allocated space into each 2D sub-matrix
  //for(int i = 0; i < kCntBucketGroup; i++) {
  //    d_projMatBucket_[i].width = matBase.xsize;
  //    d_projMatBucket_[i].height = matBase.ysize;
  //    d_projMatBucket_[i].pitch = matBase.pitch;
  //    d_projMatBucket_[i].elements = reinterpret_cast<SiftDataPtr>(i * matBase.pitch * matBase.ysize + static_cast<char *>(matBase.ptr));
  //}

  d_projMatBucket_.width = kDimSiftData;
	d_projMatBucket_.height = kDimHashData;
	cudaMallocPitch(&(d_projMatBucket_.elements),
                  &(d_projMatBucket_.pitch),
                  d_projMatBucket_.width * sizeof(SiftData_t),
                  d_projMatBucket_.height);
  CUDA_CHECK_ERROR;

  FillHashingMatrix();
}

HashConverter::~HashConverter(){ 
    cudaFree(d_projMatHamming_.elements);
    cudaFree(d_projMatBucket_.elements);
    //cudaFree(d_projMatBucket_[0].elements);

    CUDA_CHECK_ERROR;
}

void HashConverter::FillHashingMatrix() {
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    //cudaSetRandomGeneratorSeed(gen, 1); // TODO fillin arbitrary number

    SiftDataPtr base = d_projMatHamming_.elements;
    for(int i = 0; i < d_projMatHamming_.height; i++) {
        curandGenerateNormal(gen, base, kDimSiftData, 0, 1);
        base = base + d_projMatHamming_.pitch;
    }

    //for(int i = 0; i < kCntBucketGroup; i++) {
    //    base = d_projMatBucket_[i].elements;
    //    for(int j = 0; j < d_projMatBucket_[i].height; j++) {
    //        curandGenerateNormal(gen, base, kDimSiftData, 0, 1); // mean:0, stddev:1
    //        base = base + d_projMatBucket_[i].pitch;
    //    }
    //}

    base = d_projMatBucket_.elements;
    for(int i = 0; i < d_projMatBucket_.height; i++) {
        curandGenerateNormal(gen, base, kDimSiftData, 0, 1);
        base = base + d_projMatBucket_.pitch;
    }

    CUDA_CHECK_ERROR;
}

