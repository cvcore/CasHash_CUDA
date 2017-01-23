#pragma once

#include <cuda_runtime.h>
#include "Share.h"

class HashConverter {
public:
    HashConverter();
    ~HashConverter();
    void CompHash(ImageDevice &d_Image, cudaStream_t stream = 0);
    void BucketHash(ImageDevice &d_Image, cudaStream_t stream = 0);
    void CalcHashValues(ImageDevice &d_Image);
    cudaEvent_t CalcHashValuesAsync(ImageDevice &d_Image, cudaEvent_t sync = NULL);

private:
    void FillHashingMatrixCuRand();
    void FillHashingMatrixCMath();
    void FillHashingMatrixExternal(char const *path);
    float GetNormRand(void);

    Matrix<SiftData_t> d_projMatHamming_; // Matrix for 128-bit hamming vector, width = kDimSiftData
    Matrix<SiftData_t> d_projMatBucket_; // Same structure as d_projMatHamming but we chose to use only 6*8 = 48 bit from it.
    cudaStream_t hashConverterStream_;
};
