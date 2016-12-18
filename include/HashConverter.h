#pragma once

#include <cuda_runtime.h>
#include "Share.h"

class HashConverter {
public:
    HashConverter();
    ~HashConverter();
    void FillHashingMatrix();
    void CompHash(ImageDevice &d_Image);
    void BucketHash(ImageDevice &d_Image);

private:
    Matrix<SiftData_t> d_projMatHamming_; // Matrix for 128-bit hamming vector, width = kDimSiftData
    Matrix<SiftData_t> d_projMatBucket_; // Same structure as d_projMatHamming but we chose to use only 6*8 = 48 bit from it.
};
