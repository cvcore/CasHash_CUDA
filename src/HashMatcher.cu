#include "HashMatcher.h"

#include "cub/cub.cuh"


struct DistIndexPair {
    SiftData_t targetSiftDist;
    BucketEle_t index;
};

struct MinDistOp {
CUDA_UNIVERSAL_QUALIFIER DistIndexPair operator() (const DistIndexPair a, const DistIndexPair b) {
        return (a.targetSiftDist <= b.targetSiftDist) ? a : b;
    }
};

__global__ void GeneratePairKernel(Matrix<HashData_t> g_queryImageBucketID,
                                   Matrix<CompHashData_t> g_queryImageCompHashData,
                                   Matrix<SiftData_t> g_queryImageSiftData,
                                   int queryImageCntPoint,
                                   Matrix<BucketEle_t> g_targetImageBucket,
                                   Matrix<CompHashData_t> g_targetImageCompHashData,
                                   Matrix<SiftData_t> g_targetImageSiftData,
                                   BucketElePtr g_pairResult) {

    int candidate[kDimHashData + 1][kMaxCandidatePerDist]; // 6 * 1K, local memory
    int candidateLen[kDimHashData + 1];
    bool candidateUsed[kMaxCntPoint];
    int candidateTop[kCntCandidateTopMax];
    int candidateTopLen = 0;

    int querySiftIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(querySiftIndex >= queryImageCntPoint)
        return;

    memset(candidateLen, 0, sizeof(candidateLen));
    
    CompHashData_t currentCompHash[2];
    currentCompHash[0] = g_queryImageCompHashData(querySiftIndex, 0);
    currentCompHash[1] = g_queryImageCompHashData(querySiftIndex, 1);

#ifdef DEBUG_HASH_MATCHER1
    printf("current comphash: %lld %lld\n", currentCompHash[0], currentCompHash[1]);
#endif

    for(int m = 0; m < kCntBucketGroup; m++) {

        HashData_t currentBucket = g_queryImageBucketID(querySiftIndex, m);
        BucketEle_t *targetBucket = &g_targetImageBucket(m * kCntBucketPerGroup + currentBucket, 0);
        int targetBucketElements = targetBucket[0];

        for(int bucketIndex = 1; bucketIndex <= targetBucketElements; bucketIndex++) {

            int targetIndex = targetBucket[bucketIndex];

            int targetSiftDist = __popcll(currentCompHash[0] ^ g_targetImageCompHashData(targetIndex, 0)) +
                __popcll(currentCompHash[1] ^ g_targetImageCompHashData(targetIndex, 1));
            candidate[targetSiftDist][candidateLen[targetSiftDist]++] = targetIndex;
            candidateUsed[targetIndex] = false;

#ifdef DEBUG_HASH_MATCHER2
            printf("(%d %d) ", targetIndex, targetSiftDist);
#endif
        }
    }

    for(int targetSiftDist = 0; targetSiftDist <= kDimHashData; targetSiftDist++) {
        for(int i = 0; i < candidateLen[targetSiftDist]; i++) {
            int targetIndex = candidate[targetSiftDist][i];

            //if(blockIdx.x == 0 && threadIdx.x == 0) {
            //    printf("%d ", targetIndex);
            //}

            if(!candidateUsed[targetIndex]) {
                candidateUsed[targetIndex] = true;
                candidateTop[candidateTopLen++] = targetIndex;
                if(candidateTopLen == kCntCandidateTopMax)
                    break;
            }
        }

        if(candidateTopLen >= kCntCandidateTopMin)
            break;
    }

    SiftDataConstPtr queryImageSift = &g_queryImageSiftData(querySiftIndex, 0);

    double minVal1 = 0.0;
    int minValInd1 = -1;
    double minVal2 = 0.0;
    int minValInd2 = -1;

    for(int candidateListIndex = 0; candidateListIndex < candidateTopLen; candidateListIndex++) {

        int candidateIndex = candidateTop[candidateListIndex];
        SiftDataConstPtr candidateSift = &g_targetImageSiftData(candidateIndex, 0);

        float candidateDist = 0.f;
        for(int i = 0; i < kDimSiftData; i++) {
            float diff = queryImageSift[i] - candidateSift[i];
            candidateDist = candidateDist + diff * diff;
        }

        if (minValInd2 == -1 || minVal2 > candidateDist) {
            minVal2 = candidateDist;
            minValInd2 = candidateIndex;
        }

        if (minValInd1 == -1 || minVal1 > minVal2) {
            float minValTemp = minVal2;
            minVal2 = minVal1;
            minVal1 = minValTemp;
            int minValIndTemp = minValInd2;
            minValInd2 = minValInd1;
            minValInd1 = minValIndTemp;
        }
    }


    if (minVal1 < minVal2 * 0.32f) {
        g_pairResult[querySiftIndex] = minValInd1 + 1;
    } else {
        g_pairResult[querySiftIndex] = 0;
    }

}

template <int BLOCK_SIZE = HASH_MATCHER_BLOCK_SIZE, int ITEMS_PER_THREAD = kCntBucketGroup>
__global__ void GeneratePairKernelFast(Matrix<HashData_t> g_queryImageBucketID,
                                       Matrix<CompHashData_t> g_queryImageCompHashData,
                                       Matrix<SiftData_t> g_queryImageSiftData,
                                       const int queryImageCntPoint,
                                       Matrix<BucketEle_t> g_targetImageBucket,
                                       Matrix<CompHashData_t> g_targetImageCompHashData,
                                       Matrix<SiftData_t> g_targetImageSiftData,
                                       BucketElePtr g_pairResult) {

    typedef cub::BlockLoad<BucketEle_t, BLOCK_SIZE, 1, cub::BLOCK_LOAD_DIRECT> LoadBucketVectorsT;
    typedef cub::BlockRadixSort<int, BLOCK_SIZE, ITEMS_PER_THREAD, BucketEle_t> SortVectorDistT;
    typedef cub::BlockRadixSort<float, BLOCK_SIZE, 1, BucketEle_t> SortVectorSiftDistT;
    typedef cub::BlockReduce<DistIndexPair, BLOCK_SIZE> BlockReduceT;
    typedef cub::BlockExchange<BucketEle_t, BLOCK_SIZE, ITEMS_PER_THREAD> BlockExchangeT;

    __shared__ union {
        typename LoadBucketVectorsT::TempStorage load;
        typename SortVectorDistT::TempStorage sort;
        typename BlockReduceT::TempStorage reduce;
        typename BlockExchangeT::TempStorage exchange;
        typename SortVectorSiftDistT::TempStorage sortSift;
    } tempStorage;

    ///* in case of same candidate numbers being thrown from different bucket */
    //const int lastTopThreadsCnt = POSSIBLE_CANDIDATES / ITEMS_PER_THREAD; // FIXME: deal with demainders != 0
    //__shared__ BucketEle_t s_lastTopVectors[POSSIBLE_CANDIDATES];

    /* Initialize lastTops as INVALID */
    //if(threadIdx.x < POSSIBLE_CANDIDATES) {
    //    s_lastTopVectors[threadIdx.x] = INVALID_CANDIDATE;
    //}
    //__syncthreads();

    BucketEle_t targetVectors[ITEMS_PER_THREAD];
    int targetDists[ITEMS_PER_THREAD];

    int queryIndex = blockIdx.x;
    CompHashData_t queryCompHash[2];
    queryCompHash[0] = g_queryImageCompHashData(queryIndex, 0);
    queryCompHash[1] = g_queryImageCompHashData(queryIndex, 1);

    for(int group = 0; group < ITEMS_PER_THREAD; group++) {
        BucketElePtr currentBucketPtr = &g_targetImageBucket(g_queryImageBucketID(queryIndex, group), 0);
        int currentBucketSize = *currentBucketPtr;

        LoadBucketVectorsT(tempStorage.load).Load(currentBucketPtr + 1, *reinterpret_cast<BucketEle_t (*)[1]>(&targetVectors[group]), currentBucketSize, INVALID_CANDIDATE);
        __syncthreads();
    }

    #pragma unroll
    for(int i = 0; i < ITEMS_PER_THREAD; i++) {
        BucketEle_t targetIndex = targetVectors[i];

        if(targetIndex != INVALID_CANDIDATE) {
            targetDists[i] = __popcll(queryCompHash[0] ^ g_targetImageCompHashData(targetIndex, 0)) +
                __popcll(queryCompHash[1] ^ g_targetImageCompHashData(targetIndex, 1));
        } else {
            targetDists[i] = MAX_COMPHASH_DISTANCE;
        }
    }

    SortVectorDistT(tempStorage.sort).SortBlockedToStriped(targetDists, targetVectors, 0, 8); // maximum targetSiftDist possible = 128 -> end_bit = 8
    __syncthreads();

    //for(int group = 0; group < kCntBucketGroup; group++) {
    //    BucketElePtr currentBucketPtr = &g_targetImageBucket(g_queryImageBucketID(queryIndex, group), 0);
    //    int currentBucketSize = *currentBucketPtr;

    //    LoadBucketVectorsT(tempStorage.load).Load(currentBucketPtr + 1, targetVectors, currentBucketSize, INVALID_CANDIDATE);
    //    __syncthreads();

    //    if(threadIdx.x >= blockDim.x - lastTopThreadsCnt) {
    //        int offset = (threadIdx.x - (blockDim.x - lastTopThreadsCnt)) * ITEMS_PER_THREAD;

    //        #pragma unroll
    //        for(int i = 0; i < ITEMS_PER_THREAD; i++) {
    //            targetVectors[i] = s_lastTopVectors[i + offset];
    //        }
    //    }

    //    #pragma unroll
    //    for(int i = 0; i < ITEMS_PER_THREAD; i++) {
    //        BucketEle_t targetIndex = targetVectors[i];

    //        if(targetIndex != INVALID_CANDIDATE) {
    //            targetDists[i] = __popcll(queryCompHash[0] ^ g_targetImageCompHashData(targetIndex, 0)) +
    //                __popcll(queryCompHash[1] ^ g_targetImageCompHashData(targetIndex, 1));
    //        } else {
    //            targetDists[i] = MAX_COMPHASH_DISTANCE;
    //        }
    //    }

    //    SortVectorDistT(tempStorage.sort).Sort(targetDists, targetVectors, 0, 8); // end_bit = 8, maximum possible targetSiftDist = 128

    //    if(threadIdx.x < lastTopThreadsCnt) {
    //        int offset = threadIdx.x * ITEMS_PER_THREAD;

    //        #pragma unroll
    //        for(int i = 0; i < ITEMS_PER_THREAD; i++) {
    //            s_lastTopVectors[i + offset] = targetVectors[i];
    //        }
    //    }

    //    __syncthreads();
    //}

    float targetSiftDist = MAX_SIFT_DISTANCE;

    if(targetVectors[0] != INVALID_CANDIDATE) {
        targetSiftDist = 0;
        SiftDataPtr querySiftVector = &g_queryImageSiftData(queryIndex, 0),
            targetSiftVector = &g_targetImageSiftData(targetVectors[0], 0);

        for(int i = 0; i < kDimSiftData; i++) {
            float diff = querySiftVector[i] - targetSiftVector[i];
            targetSiftDist += diff * diff;
        }
    }

    //SortVectorSiftDistT(tempStorage.sortSift).Sort(targetSiftDist, *reinterpret_cast<int (*)[1]>(targetVectors));

    //if(threadIdx.x == 0) {
    //    if(min1.targetSiftDist < min2.targetSiftDist * 0.32f) {
    //        g_pairResult[queryIndex] = min1.index;
    //    } else {
    //        g_pairResult[queryIndex] = INVALID_CANDIDATE;
    //    }
    //}
}

cudaEvent_t HashMatcher::GeneratePair(int queryImageIndex, int targetImageIndex) {
    ImageDevice &queryImage = d_imageList_[queryImageIndex];
    const ImageDevice &targetImage = d_imageList_[targetImageIndex];

    BucketElePtr candidateArray;
    cudaMalloc(&candidateArray, sizeof(BucketEle_t) * queryImage.cntPoint);
    CUDA_CHECK_ERROR;

    queryImage.targetCandidates[targetImageIndex] = candidateArray;

    dim3 gridSize(queryImage.cntPoint);
    dim3 blockSize(HASH_MATCHER_BLOCK_SIZE);
    
    GeneratePairKernelFast<<<gridSize, blockSize, 0, hashMatcherStream_>>>(
        queryImage.bucketIDList,
        queryImage.compHashData,
        queryImage.siftData,
        queryImage.cntPoint,
        targetImage.bucketList,
        targetImage.compHashData,
        targetImage.siftData,
        candidateArray);

    cudaEvent_t finish;
    cudaEventCreate(&finish);
    cudaEventRecord(finish, hashMatcherStream_);

    return finish;
}
