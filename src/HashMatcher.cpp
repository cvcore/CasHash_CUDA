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
        h_matchDatabase.push_back(GeneratePair(currentImages - 1, imageIndex)); // pair with all previous images
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
        h_matchDatabase.push_back(GeneratePair(currentImages - 1, imageIndex)); // pair with all previous images
        // TODO pair with user-specified list
    }

    cudaEvent_t finish;
    cudaEventCreate(&finish);
    cudaEventRecord(finish, hashMatcherStream_);

    return finish;
}

int HashMatcher::NumberOfMatch(int imageIndex1, int imageIndex2) {
    if(imageIndex1 == imageIndex2) {
        return -1;
        std::cerr << "Error: NumberOfMatch should be used between different images!\n";
    }

    return h_matchDatabase[PairListIndex(imageIndex1, imageIndex2)]->size();
}

MatchPairListPtr HashMatcher::MatchPairList( int imageIndex1, int imageIndex2 ) {
    return h_matchDatabase[PairListIndex(imageIndex1, imageIndex2)];
}
