#pragma once

#include <cuda_runtime.h>
#include "Share.h"
#include <vector>

const int HASH_MATCHER_BLOCK_SIZE = 128;

class HashMatcher {
public:
    HashMatcher();
    ~HashMatcher();
    int NumberOfMatch(int imageIndex1, int imageIndex2);
    MatchPairPtr MatchPairList(int imageIndex1, int imageIndex2);
    int AddImage(ImageDevice d_Image); /* return value: image index */
    
private:
    std::vector<ImageDevice> d_imageList_;
    std::vector<MatchPairListPtr> h_matchDatabase;
    MatchPairPtr GeneratePair(int imageIndex1, int imageIndex2);
};
