#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <string>

#ifdef __CUDACC__
#define CUDA_UNIVERSAL_QUALIFIER __host__ __device__
#else
#define CUDA_UNIVERSAL_QUALIFIER
#endif

const int kDimSiftData = 128; // the number of dimensions of SIFT feature
const int kDimHashData = 128; // the number of dimensions of Hash code
const int kBitInCompHash = 64; // the number of Hash code bits to be compressed; in this case, use a <uint64_t> variable to represent 64 bits
const int kDimCompHashData = kDimHashData / kBitInCompHash; // the number of dimensions of CompHash code
const int kMinMatchListLen = 16; // the minimal list length for outputing SIFT matching result between two images
const int kMaxCntPoint = 1000000; // the maximal number of possible SIFT points; ensure this value is not exceeded in your application

const int kCntBucketBit = 8; // the number of bucket bits
const int kCntBucketGroup = 6; // the number of bucket groups
const int kCntBucketPerGroup = 1 << kCntBucketBit; // the number of buckets in each group

const int kCntCandidateTopMin = 6; // the minimal number of top-ranked candidates
const int kCntCandidateTopMax = 10; // the maximal number of top-ranked candidates

typedef float SiftData_t; // CUDA GPUs are optimized for float arithmetics, we use float instead of int
typedef float* SiftDataPtr;
typedef const float* SiftDataConstPtr;
typedef uint8_t HashData_t;
typedef uint8_t* HashDataPtr; // Hash code is represented with <uint8_t> type; only the lowest bit is used
typedef uint64_t CompHashData_t;
typedef uint64_t* CompHashDataPtr; // CompHash code is represented with <uint64_t> type
typedef int* BucketElePtr; // index list of points in a specific bucket

template <typename T>
struct Matrix {
    int width;
    int height;
    size_t pitch; // row size in bytes
    T* elements;

    CUDA_UNIVERSAL_QUALIFIER inline T& operator() (int i, int j) {
        return *(reinterpret_cast<T *>(reinterpret_cast<char *>(elements) + i * pitch) + j);
    } // no more ugly pointer calcs

    CUDA_UNIVERSAL_QUALIFIER inline const T& operator() (int i, int j) const {
         return *(reinterpret_cast<T *>(reinterpret_cast<char *>(elements) + i * pitch) + j);
    }

    Matrix(int H, int W) : height(H), width(W){
        pitch = sizeof(T) * width; // init pitch, will be adjusted later if use cudaMallocPitch
    }

    Matrix() : width(0), height(0), pitch(0), elements(NULL) {
    }
};

struct ImageHost {
    int cntPoint; // the number of SIFT points
    std::string keyFilePath;
    Matrix<SiftData_t> siftData; // [cntPoint x 128] Matrix, storing all sift vectors one-off

};

struct ImageDevice {
    int cntPoint;
    Matrix<SiftData_t> siftData;
    Matrix<CompHashData_t> compHashData; // [cntPoint x 2 Matrix]

    /* not currently used: */
    uint16_t* bucketIDList[kCntBucketGroup];
    int cntEleInBucket[kCntBucketGroup][kCntBucketPerGroup];
    BucketElePtr bucketList[kCntBucketGroup][kCntBucketPerGroup];

};

#define CUDA_CHECK_ERROR                                                         \
    do {                                                                         \
        const cudaError_t err = cudaGetLastError();                              \
        if (err != cudaSuccess) {                                                \
            const char *const err_str = cudaGetErrorString(err);                 \
            std::cerr << "Cuda error in " << __FILE__ << ":" << __LINE__ - 1     \
                      << ": " << err_str << " (" << err << ")" << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while(0)

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CATCH_ERROR(val) check ( (val), #val, __FILE__, __LINE__)
