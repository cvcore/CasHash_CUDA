#include "KeyFileReader.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

KeyFileReader::KeyFileReader() {
}

KeyFileReader::~KeyFileReader() {
    std::vector<ImageHost>::iterator it;
    for(it = h_imageList_.begin(); it != h_imageList_.end(); ++it) {
        delete[] it->siftData.elements;
    }
}

void KeyFileReader::AddKeyFile( const char *path ) {
    FILE *keyFile = fopen(path, "r");
    if(keyFile == NULL) {
        fprintf(stderr, "Key file %s does not exist!\n", path);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Reading SIFT vector from %s\n", path);
    int cntPoint, cntDim;
    fscanf(keyFile, "%d%d", &cntPoint, &cntDim);
    if(cntDim != kDimSiftData) {
        fprintf(stderr, "Unsupported SIFT vector dimension %d, should be %d!\n", cntDim, kDimSiftData);
        exit(EXIT_FAILURE);
    }

    ImageHost newImage;
    newImage.cntPoint = cntPoint;
    newImage.keyFilePath = path;

    size_t requiredSize = cntPoint * cntDim;
    newImage.siftData.elements = new SiftData_t[requiredSize];
    newImage.siftData.width = cntDim;
    newImage.siftData.height = cntPoint;
    newImage.siftData.pitch = cntDim * sizeof(SiftData_t);
    if( newImage.siftData.elements == NULL) {
        fprintf(stderr, "Can't allocate memory for host image!\n");
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < cntPoint; i++) {
        SiftDataPtr rowVector = newImage.siftData.elements + kDimSiftData * i;
        for(int j = 0; j < kDimSiftData; j++) {
            fscanf(keyFile, "%f", &rowVector[j]);
            siftAccumulator_[j] = siftAccumulator_[j] + static_cast<float>(rowVector[j]);
            cntTotalVector_++;
        }
    }
    fclose(keyFile);
    h_imageList_.push_back(newImage);
    cntImage = h_imageList_.size();
}

void KeyFileReader::OpenKeyList( const char *path ) {
    FILE *keyList = fopen(path, "r");
    char keyFilePath[256];
    if(keyList == NULL) {
        fprintf(stderr, "Keylist file %s does not exist!\n", path);
        exit(EXIT_FAILURE);
    }
    while(fscanf(keyList, "%s", keyFilePath) > 0) {
        AddKeyFile(keyFilePath);
    }
    fclose(keyList);
}

void KeyFileReader::ZeroMeanProc() {
    std::vector<ImageHost>::iterator it;
    SiftData_t mean[kDimSiftData];
    for(int i = 0; i < kDimSiftData; i++) {
        mean[i] = siftAccumulator_[i] / cntTotalVector_;
    }
    for(it = h_imageList_.begin(); it != h_imageList_.end(); ++it) {
        for(int i = 0; i < it->cntPoint; i++) {
            SiftDataPtr rowVector = it->siftData.elements + i * kDimSiftData;
            for(int j = 0; j < kDimSiftData; j++) {
                rowVector[j] -= mean[j];
            }
        }
    }
}

void KeyFileReader::UploadImage( ImageDevice &d_Image, const int index ) {
    d_Image.cntPoint = h_imageList_[index].cntPoint;
    d_Image.siftData.width = kDimSiftData;
    d_Image.siftData.height = h_imageList_[index].cntPoint;

    cudaMallocPitch(&(d_Image.siftData.elements),
                    &(d_Image.siftData.pitch),
                    d_Image.siftData.width * sizeof(SiftData_t),
                    d_Image.siftData.height);

    cudaMemcpy2D(d_Image.siftData.elements,
                 d_Image.siftData.pitch,
                 h_imageList_[index].siftData.elements,
                 h_imageList_[index].siftData.pitch,
                 h_imageList_[index].siftData.width * sizeof(SiftData_t),
                 h_imageList_[index].siftData.height, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR;
}
