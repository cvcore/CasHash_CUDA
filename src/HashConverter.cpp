#include "HashConverter.h"
#include "Share.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <cmath>

HashConverter::HashConverter() {
	//Allocate matrix for hashing into 128d-Hamming space
	d_projMatHamming_.width = kDimSiftData;
	d_projMatHamming_.height = kDimHashData;
	cudaMallocPitch(&(d_projMatHamming_.elements),
                  &(d_projMatHamming_.pitch),
                  d_projMatHamming_.width * sizeof(SiftData_t),
                  d_projMatHamming_.height);
  CUDA_CHECK_ERROR;

  d_projMatBucket_.width = kDimSiftData;
	d_projMatBucket_.height = kDimHashData;
	cudaMallocPitch(&(d_projMatBucket_.elements),
                  &(d_projMatBucket_.pitch),
                  d_projMatBucket_.width * sizeof(SiftData_t),
                  d_projMatBucket_.height);
  CUDA_CHECK_ERROR;

  FillHashingMatrixCuRand();
}

HashConverter::~HashConverter(){ 
    cudaFree(d_projMatHamming_.elements);
    cudaFree(d_projMatBucket_.elements);
    //cudaFree(d_projMatBucket_[0].elements);

    CUDA_CHECK_ERROR;
}

void HashConverter::FillHashingMatrixCuRand() {
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    
    for(int i = 0; i < d_projMatHamming_.height; i++) {
        curandGenerateNormal(gen, &d_projMatHamming_(i, 0), kDimSiftData, 0, 1);
    }

    for(int i = 0; i < d_projMatBucket_.height; i++) {
        curandGenerateNormal(gen, &d_projMatBucket_(i, 0), kDimSiftData, 0, 1);
    }

    CUDA_CHECK_ERROR;

#ifdef DEBUG_HASH_CONVERTER_RANDOM_MATRIX
    std::cout << "Device random matrix:\n";
    dumpDeviceArray(&d_projMatBucket_(0, 0), 128);
#endif

}

void HashConverter::FillHashingMatrixCMath() {
    SiftDataPtr tempRand = new SiftData_t[kDimSiftData];

    for(int i = 0; i < d_projMatHamming_.height; i++) {
        for(int j = 0; j < kDimSiftData; j++) {
            tempRand[j] = GetNormRand();
        }
        cudaMemcpy(&d_projMatHamming_(i, 0), tempRand, kDimSiftData * sizeof(SiftData_t), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < d_projMatHamming_.height; i++) {
        for(int j = 0; j < kDimSiftData; j++) {
            tempRand[j] = GetNormRand();
        }
        cudaMemcpy(&d_projMatBucket_(i, 0), tempRand, kDimSiftData * sizeof(SiftData_t), cudaMemcpyHostToDevice);
    }

#ifdef DEBUG_HASH_CONVERTER_RANDOM_MATRIX
    std::cout << "Device random matrix:\n";
    dumpDeviceArray(&d_projMatBucket_(0, 0), 128);
#endif

}

void HashConverter::FillHashingMatrixExternal(char const *path) {
    FILE *randomFp = fopen(path, "r");
    if(!randomFp) {
        std::cerr << "Random matrix does not exist!\n";
        exit(-1);
    }

    SiftDataPtr tempRand = new SiftData_t[kDimSiftData];

    for(int i = 0; i < d_projMatHamming_.height; i++) {
        for(int j = 0; j < kDimSiftData; j++) {
            fscanf(randomFp, "%f", &tempRand[j]);
        }
        cudaMemcpy(&d_projMatHamming_(i, 0), tempRand, kDimSiftData * sizeof(SiftData_t), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < kCntBucketGroup * kCntBucketBit; i++) {
        for(int j = 0; j < kDimSiftData; j++) {
            fscanf(randomFp, "%f", &tempRand[j]);
        }
        cudaMemcpy(&d_projMatBucket_(i, 0), tempRand, kDimSiftData * sizeof(SiftData_t), cudaMemcpyHostToDevice);
    }

    delete [] tempRand;
    fclose(randomFp);
}

void HashConverter::CalcHashValues(ImageDevice &d_Image){
	CompHash(d_Image);
	BucketHash(d_Image);
	//cudaFree(d_Image.siftData.elements);
}

float HashConverter::GetNormRand(void) {
    // based on Box-Muller transform; for more details, please refer to the following WIKIPEDIA website:
    // http://en.wikipedia.org/wiki/Box_Muller_transform
    float u1 = (rand() % 1000 + 1) / 1000.0;
    float u2 = (rand() % 1000 + 1) / 1000.0;

    float randVal = sqrt(-2 * log(u1)) * cos(2 * acos(-1.0) * u2);

    return randVal;
}
