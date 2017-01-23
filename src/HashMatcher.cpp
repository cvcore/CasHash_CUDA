#include "HashMatcher.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

HashMatcher::HashMatcher() {
    hashMatcherStream_ = 0;
}

HashMatcher::~HashMatcher() { 

}

int PairListIndex(int imageIndex1, int imageIndex2) {
    if(imageIndex2 > imageIndex1)
        std::swap(imageIndex1, imageIndex2);

    return imageIndex1 * (imageIndex1 - 1) / 2 + imageIndex2;
}

void HashMatcher::AddImage(const ImageDevice &d_Image) {
    d_imageList_.push_back(d_Image);

    int currentImages = d_imageList_.size();

    for(int imageIndex = 0; imageIndex < currentImages - 1; imageIndex++) {
        GeneratePair(currentImages - 1, imageIndex); // pair with all previous images
        // TODO pair with user-specified list
    }
}

cudaEvent_t HashMatcher::AddImageAsync(const ImageDevice &d_Image, cudaEvent_t sync) {
    if(hashMatcherStream_ == 0) {
        cudaStreamCreate(&hashMatcherStream_);
    }

    if(sync) {
        cudaStreamWaitEvent(hashMatcherStream_, sync, 0);
    }

    d_imageList_.push_back(d_Image);

    int currentImages = d_imageList_.size();

    for(int imageIndex = 0; imageIndex < currentImages - 1; imageIndex++) {
        GeneratePair(currentImages - 1, imageIndex); // pair with all previous images
        // TODO pair with user-specified list
    }

    cudaEvent_t finish;
    cudaEventCreate(&finish);
    cudaEventRecord(finish, hashMatcherStream_);

    return finish;
}

int HashMatcher::NumberOfMatch(int queryImageIndex, int targetImageIndex) {
    return MatchPairList(queryImageIndex, targetImageIndex)->size();
}

MatchPairListPtr HashMatcher::MatchPairList( int queryImageIndex, int targetImageIndex ) {
    auto queryTargetPair = std::make_pair(queryImageIndex, targetImageIndex);

    if(!matchDataBase_.count(queryTargetPair)) {
        ImageDevice &queryImage = d_imageList_[queryImageIndex];
        BucketElePtr d_candidateArray = queryImage.targetCandidates[targetImageIndex];
        BucketElePtr h_candidateArray = new BucketEle_t[queryImage.cntPoint];

        cudaMemcpy(h_candidateArray, d_candidateArray, queryImage.cntPoint * sizeof(BucketEle_t), cudaMemcpyDeviceToHost);
        cudaFree(d_candidateArray);
        CUDA_CHECK_ERROR;

        MatchPairListPtr newMatchPairList(new MatchPairList_t);

        for(int point = 0; point < queryImage.cntPoint; point++) {
            if(h_candidateArray[point] != INVALID_CANDIDATE) {
                newMatchPairList->push_back(std::make_pair(point, h_candidateArray[point]));
            }
        }

        matchDataBase_[queryTargetPair] = newMatchPairList;
    }

    return matchDataBase_[queryTargetPair];
}
