#pragma once

#include <cuda_runtime.h>
#include "Share.h"
#include <vector>

const BucketEle_t INVALID_CANDIDATE = ~0;
const int MAX_COMPHASH_DISTANCE = ~(1 << (sizeof(int) * 8 - 1));
const float MAX_SIFT_DISTANCE = 1.0e38f;
const int POSSIBLE_CANDIDATES = 8;
const int HASH_MATCHER_BLOCK_SIZE = 32;
//const int HASH_MATCHER_ITEMS_PER_THREAD = 2;

class HashMatcher {
public:
    HashMatcher();
    ~HashMatcher();
    int NumberOfMatch(int queryImageIndex, int targetImageIndex);
    MatchPairListPtr MatchPairList(int queryImageIndex, int targetImageIndex);
    void AddImage(const ImageDevice &d_Image); /* return value: image index */
    cudaEvent_t AddImageAsync(const ImageDevice &d_Image, cudaEvent_t sync = NULL);
    
private:
    std::vector<ImageDevice> d_imageList_;
    std::map< std::pair< int, int >, MatchPairListPtr > matchDataBase_;
    cudaStream_t hashMatcherStream_;

    cudaEvent_t GeneratePair(int queryImageIndex, int targetImageIndex);
};
