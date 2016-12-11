#pragma once

#include <cuda_runtime.h>
#include "Share.h"

class KeyFileReader {
public:
    KeyFileReader();
    ~KeyFileReader();
    void Read(ImageDataDevice *imgD, const char *keyPath);

private:
    SiftDataPtr siftArray_;
    size_t siftArraySize_;
    void TempArrayAdjust(size_t newSize);
};
