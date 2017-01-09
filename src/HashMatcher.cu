#include "HashMatcher.h"
#include "Share.h"

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

            int dist = __popcll(currentCompHash[0] ^ g_targetImageCompHashData(targetIndex, 0)) +
                __popcll(currentCompHash[1] ^ g_targetImageCompHashData(targetIndex, 1));
            candidate[dist][candidateLen[dist]++] = targetIndex;
            candidateUsed[targetIndex] = false;

#ifdef DEBUG_HASH_MATCHER2
            printf("(%d %d) ", targetIndex, dist);
#endif
        }
    }

    for(int dist = 0; dist <= kDimHashData; dist++) {
        for(int i = 0; i < candidateLen[dist]; i++) {
            int targetIndex = candidate[dist][i];

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


    if (minVal1 < minVal2 * 0.32) {
        g_pairResult[querySiftIndex] = minValInd1 + 1;
    } else {
        g_pairResult[querySiftIndex] = 0;
    }

}

MatchPairListPtr HashMatcher::GeneratePair(int queryImageIndex, int targetImageIndex) {
    const ImageDevice &queryImage = d_imageList_[queryImageIndex];
    const ImageDevice &targetImage = d_imageList_[targetImageIndex];

    BucketElePtr d_pairResult, h_pairResult;
    cudaMalloc(&d_pairResult, sizeof(BucketEle_t) * queryImage.cntPoint);
    CUDA_CHECK_ERROR;

    cudaMemset(d_pairResult, 0, sizeof(BucketEle_t) * queryImage.cntPoint);
    CUDA_CHECK_ERROR;

    dim3 blockSize(HASH_MATCHER_BLOCK_SIZE);
    dim3 gridSize(queryImage.cntPoint / HASH_MATCHER_BLOCK_SIZE);

    GeneratePairKernel<<<gridSize, blockSize>>>(queryImage.bucketIDList,
                                                queryImage.compHashData,
                                                queryImage.siftData,
                                                queryImage.cntPoint,
                                                targetImage.bucketList,
                                                targetImage.compHashData,
                                                targetImage.siftData,
                                                d_pairResult);

    h_pairResult = new BucketEle_t[queryImage.cntPoint];
    cudaMemcpy(h_pairResult, d_pairResult, sizeof(BucketEle_t) * queryImage.cntPoint, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR;
    cudaFree(d_pairResult);
    CUDA_CHECK_ERROR;

    MatchPairListPtr matchPairList(new MatchPairList_t);
    
    for(int resultIndex = 0; resultIndex < queryImage.cntPoint; resultIndex++) {
        if(h_pairResult[resultIndex] != 0) {
            matchPairList->push_back(std::make_pair(resultIndex, h_pairResult[resultIndex] - 1));
        }
    }

    return matchPairList;
}
