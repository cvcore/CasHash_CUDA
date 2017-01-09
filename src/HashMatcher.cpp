#include "HashMatcher.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

HashMatcher::HashMatcher() {

}

HashMatcher::~HashMatcher() { 

}

int PairListIndex(int imageIndex1, int imageIndex2) {
    if(imageIndex2 > imageIndex1)
        std::swap(imageIndex1, imageIndex2);

    return imageIndex1 * (imageIndex1 - 1) / 2 + imageIndex2;
}

int HashMatcher::AddImage(const ImageDevice &d_image) {
    d_imageList_.push_back(d_image);

    int currentImages = d_imageList_.size();

    for(int imageIndex = 0; imageIndex < currentImages - 1; imageIndex++) {
        h_matchDatabase.push_back(GeneratePair(currentImages - 1, imageIndex)); // pair with all previous images
        // TODO pair with user-specified list
    }

    return d_imageList_.size();
}

int HashMatcher::NumberOfMatch(int imageIndex1, int imageIndex2) {
    if(imageIndex1 == imageIndex2) {
        return -1;
        std::cerr << "Error: NumberOfMatch should be used between different images!\n";
    }

    return h_matchDatabase[PairListIndex(imageIndex1, imageIndex2)]->size();
}
