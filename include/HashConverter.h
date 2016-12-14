#pragma once

#include <cuda_runtime.h>
#include "Share.h"

class HashConverter {
public:
    HashConverter();
    ~HashConverter();
    void SiftDataToHashData(ImageDataDevice *imgD);

private:
	Matrix d_projMatHamming; // projection matrix of the primary hashing function
   // 	Matrix* d_projMatBucket[kCntBucketGroup][kCntBucketBit][kDimSiftData]; // projection matrix of the secondary hashing function
   //	Matrix d_bucketBitList[kCntBucketGroup][kCntBucketBit]; // selected bits in the result of primary hashing fuction for bucket construction
};
