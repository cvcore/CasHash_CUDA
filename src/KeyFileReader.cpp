#include "KeyFileReader.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

KeyFileReader::KeyFileReader() {
}

KeyFileReader::~KeyFileReader() {
    std::vector<ImageDataHost>::iterator it;
    for(it = h_imageList_.begin(); it != h_imageList_.end(); ++it) {
        delete[] it->siftDataMatrix;
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
    ImageDataHost newImage;
    newImage.cntPoint = cntPoint;
    newImage.keyFilePath = path;
    size_t requiredSize = cntPoint * cntDim;
    newImage.siftDataMatrix = new SiftData_t[requiredSize];
    if( newImage.siftDataMatrix == NULL) {
        fprintf(stderr, "Can't allocate memory for host image!\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < cntPoint; i++) {
        SiftDataPtr rowVector = newImage.siftDataMatrix + kDimSiftData * i;
        for(int j = 0; j < kDimSiftData; j++) {
            fscanf(keyFile, "%d", &rowVector[j]);
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
    std::vector<ImageDataHost>::iterator it;
    int mean[kDimSiftData];
    for(int i = 0; i < kDimSiftData; i++) {
        mean[i] = static_cast<int>(siftAccumulator_[i] / cntTotalVector_);
    }
    for(it = h_imageList_.begin(); it != h_imageList_.end(); ++it) {
        for(int i = 0; i < it->cntPoint; i++) {
            SiftDataPtr rowVector = it->siftDataMatrix + i * kDimSiftData;
            for(int j = 0; j < kDimSiftData; j++) {
                rowVector[j] -= mean[j];
            }
        }
    }
}

void KeyFileReader::UploadImage( ImageDataDevice &imgDev, const int index ) {
    SiftDataPtr& d_siftMat = imgDev.siftDataMatrix;
    size_t& pitch = imgDev.siftDataMatrixPitch;
    SiftDataPtr h_siftMat = h_imageList_[index].siftDataMatrix;
    size_t height = h_imageList_[index].cntPoint;

    cudaMallocPitch(&d_siftMat, &pitch, kDimSiftData * sizeof(SiftData_t), height);
    cudaMemcpy2D(d_siftMat, pitch, h_siftMat, sizeof(SiftData_t) * kDimSiftData, sizeof(SiftData_t) * kDimSiftData, height, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR;
}
