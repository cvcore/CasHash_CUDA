#pragma once

#include <cuda_runtime.h>
#include "Share.h"

class HashConverter {
public:
    HashConverter();
    ~HashConverter();
    void SiftDataToHashData(ImageDataDevice *imgD);
    void FillHashingMatrix();

private:
    Matrix d_projMatHamming_; // Matrix for 128-bit hamming vector, width = kDimSiftData
    Matrix d_projMatBucket_[kCntBucketGroup]; // Matrix for bucket making, width = kDimSiftData
};
