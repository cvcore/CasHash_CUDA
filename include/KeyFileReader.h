#pragma once


#include "Share.h"

class KeyFileReader {
public:
    void Read(ImageDataDevice *imgD, const char *keyPath);

private:
    SiftDataPtr siftArray_;
    size_t siftArraySize_;
}
