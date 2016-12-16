#include "HashConverter.h"
#include "Share.h"
#include <iostream>

// blockIdx.y: bit
// threadIdx.x + blockIdx.x * BLOCK_SIZE = Nr. sift vec

// threadIdx.x: final bit.
// blockIdx.x: Nr. sift vec

__global__ void CompHashKernel(Matrix<SiftData_t> g_sift, const Matrix<SiftData_t> g_projMat, Matrix<CompHashData_t> g_compHash) {
    __shared__  float s_siftCur[kDimSiftData]; // shared sift vector
    __shared__ uint32_t s_hashBits[kBitInCompHash];
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
    uint32_t elementBit = static_cast<int>(element > 0.f);
    elementBit <<= tx % 32;
    __syncthreads();

    for(int stride = 2; stride <= 32; stride <<= 1) {
        if(tx % stride == 0) {
            s_hashBits[tx] += s_hashBits[tx + stride / 2];
        }
        __syncthreads();
    }
    if(tx % 64 == 0) {
        uint64_t halfCompHash = static_cast<uint64_t>(s_hashBits[tx + 32] << 32 + s_hashBits[tx]);
        g_compHash(bx, tx/64) = halfCompHash;
    }
}

void HashConverter::CompHash( ImageDevice &d_Image ) {
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

    CompHashKernel<<<gridSize, blockSize>>>(d_Image.siftData,
                                            d_projMatHamming_,
                                            d_Image.compHashData);
    CUDA_CHECK_ERROR;
}
