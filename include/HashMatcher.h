#pragma once

#include <cuda_runtime.h>
#include "Share.h"
#include <vector>

class HashMatcher {
public:
    HashMatcher();
    ~HashMatcher();
    int NumberOfMatch(int imageIndex1, int imageIndex2);
    MatchPairListPtr MatchPairList(int imageIndex1, int imageIndex2);
    void AddImage(const ImageDevice &d_Image); /* return value: image index */
    cudaEvent_t AddImageAsync(const ImageDevice &d_Image, cudaEvent_t sync = NULL);
    
private:
    std::vector<ImageDevice> d_imageList_;
    std::vector<MatchPairListPtr> h_matchDatabase;
    MatchPairListPtr GeneratePair(int imageIndex1, int imageIndex2);
    cudaStream_t hashMatcherStream_;
};
