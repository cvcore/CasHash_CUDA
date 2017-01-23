#include "HashConverter.h"
#include "Share.h"

__global__ void CompHashKernel(Matrix<SiftData_t> g_sift, const Matrix<SiftData_t> g_projMat, Matrix<CompHashData_t> g_compHash) {
    __shared__  float s_siftCur[kDimSiftData]; // shared sift vector
    __shared__ uint32_t s_hashBits[kDimHashData];
    SiftDataPtr g_siftCur = &g_sift(blockIdx.x, 0);
    SiftDataConstPtr g_projMatCur = &g_projMat(threadIdx.x, 0);
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    s_siftCur[tx] = g_siftCur[tx]; // we can do this because kDimSiftData == kBitInCompHash, otherwise we need to setup a if condition
    __syncthreads();

    float element = 0.f;
    for(int i = 0; i < kDimSiftData; i++) {
        element = element + s_siftCur[i] * g_projMatCur[i];
    }
    uint32_t hashVal = static_cast<uint32_t>(element > 0.f);
    hashVal <<= (tx % 32);
    s_hashBits[tx] = hashVal;
    __syncthreads();

    for(int stride = 2; stride <= 32; stride <<= 1) {
        if(tx % stride == 0) {
            s_hashBits[tx] += s_hashBits[tx + stride / 2];
        }
    }

    if(tx % 64 == 0) {
        uint64_t halfCompHash = (static_cast<uint64_t>(s_hashBits[tx + 32] << 32) + s_hashBits[tx]);
        g_compHash(bx, tx / 64) = halfCompHash;
    }
}

void HashConverter::CompHash( ImageDevice &d_Image, cudaStream_t stream ) {
    // d_Image.compHashData.width = 2;
    // d_Image.compHashData.height = d_Image.cntPoint;
    // cudaMallocPitch(&(d_Image.compHashData.elements),
    //                 &(d_Image.compHashData.pitch),
    //                 d_Image.compHashData.width,
    //                 d_Image.compHashData.height);

    d_Image.compHashData.width = 2;
    d_Image.compHashData.pitch = sizeof(CompHashData_t) * 2;
    d_Image.compHashData.height = d_Image.cntPoint;
    cudaMalloc(&(d_Image.compHashData.elements),
               d_Image.compHashData.pitch * d_Image.compHashData.height);
    CUDA_CHECK_ERROR;

    dim3 blockSize(kDimHashData);
    dim3 gridSize(d_Image.cntPoint);

    if(stream == 0)
        CompHashKernel<<<gridSize, blockSize>>>(d_Image.siftData,
                                                d_projMatHamming_,
                                                d_Image.compHashData);
    else {
        CompHashKernel<<<gridSize, blockSize, 0, stream>>>(d_Image.siftData,
                                                           d_projMatHamming_,
                                                           d_Image.compHashData);
    }

    CUDA_CHECK_ERROR;
}

__global__ void BucketHashKernel(Matrix<SiftData_t> g_sift, const Matrix<SiftData_t> g_projMat, Matrix<HashData_t> g_bucketHash, Matrix<BucketEle_t> g_bucketEle) {
    __shared__  float s_siftCur[kDimSiftData]; // shared sift vector
    __shared__ int s_hashBits[kDimHashData];
    SiftDataPtr g_siftCur = &g_sift(blockIdx.x, 0);
    SiftDataConstPtr g_projMatCur = &g_projMat(threadIdx.x, 0);
    int tx = threadIdx.x; // hash group
    int bx = blockIdx.x; // sift vector index
    int idx = tx + bx * blockDim.x;

    s_siftCur[tx] = g_siftCur[tx]; // we can do this because kDimSiftData == kBitInCompHash, otherwise we need to setup a if condition
    if(idx < g_bucketEle.height)
        g_bucketEle(idx, 0) = 0;

    __syncthreads();

    float element = 0.f;
    for(int i = 0; i < kDimSiftData; i++) {
        element = element + s_siftCur[i] * g_projMatCur[i];
    }

    int hashVal = static_cast<int>(element > 0.f);

    hashVal <<= tx % 8;
    s_hashBits[tx] = hashVal;
    __syncthreads();

    for(int stride = 2; stride <= 8; stride <<= 1) {
        if(tx % stride == 0) {
            s_hashBits[tx] += s_hashBits[tx + stride / 2];
        }
    }

    if(tx % 8 == 0 && tx / 8 < kCntBucketGroup) {
        hashVal = s_hashBits[tx];
        g_bucketHash(bx, tx / 8) = hashVal;
        BucketElePtr baseAddr = &(g_bucketEle(kCntBucketPerGroup * tx / 8 + hashVal, 0));
        int currIdx = atomicInc(baseAddr, kMaxMemberPerGroup) + 1;
 
#ifdef DEBUG_HASH_CONVERTER
        printf("%d %d %d\n", tx / 8, hashVal, currIdx);
        if(currIdx == kMaxMemberPerGroup) {
            printf("Warning: bucket full! Consider increasing bucket #%d in group %d!\n", hashVal, tx / 8);
        }
#endif

        g_bucketEle(kCntBucketPerGroup * tx / 8 + hashVal, currIdx) = bx;
    }
}

void HashConverter::BucketHash( ImageDevice &d_Image, cudaStream_t stream ) {
    d_Image.bucketIDList.width = kCntBucketGroup;
    d_Image.bucketIDList.height = d_Image.cntPoint;
    cudaMallocPitch(&(d_Image.bucketIDList.elements),
                    &(d_Image.bucketIDList.pitch),
                    d_Image.bucketIDList.width * sizeof(HashData_t),
                    d_Image.bucketIDList.height);

    d_Image.bucketList.width = kMaxMemberPerGroup;
    d_Image.bucketList.height = kCntBucketGroup * kCntBucketPerGroup;
    cudaMallocPitch(&(d_Image.bucketList.elements),
                    &(d_Image.bucketList.pitch),
                    d_Image.bucketList.width * sizeof(BucketEle_t),
                    d_Image.bucketList.height);


    //for(int i = 0; i < d_Image.bucketList.height; i++) {
    //    cudaMemset(&(d_Image.bucketList(i, 0)),
    //               0,
    //               sizeof(BucketEle_t));
    //    CUDA_CHECK_ERROR;
    //}

    //CUDA_CHECK_ERROR;

    // TODO bucketEle
    dim3 blockSize(kDimHashData);
    dim3 gridSize(d_Image.cntPoint);
    
    if(stream == 0)
        BucketHashKernel<<<gridSize, blockSize>>>(d_Image.siftData,
                                                  d_projMatBucket_,
                                                  d_Image.bucketIDList,
                                                  d_Image.bucketList);
    else {
        BucketHashKernel<<<gridSize, blockSize, 0, stream>>>(d_Image.siftData, d_projMatBucket_, d_Image.bucketIDList, d_Image.bucketList);
    }

#ifdef DEBUG_HASH_CONVERTER2
    for(int m = 0; m < kCntBucketGroup; m++) {
        for(int bucket = 0; bucket < kCntBucketPerGroup; bucket++) {
            BucketEle_t bucketSize;
            cudaMemcpy(&bucketSize, &(d_Image.bucketList(m * kCntBucketPerGroup + bucket, 0)), sizeof(BucketEle_t), cudaMemcpyDeviceToHost);
            std::cout << "Group: " << m << " Bucket: " << bucket << " Size: " << bucketSize << "\n";
        }
    }
    CUDA_CHECK_ERROR;
#endif

}
 
