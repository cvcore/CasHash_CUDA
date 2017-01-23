#include "KeyFileReader.h"
#include "Share.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cstring>

KeyFileReader::KeyFileReader() {
    std::memset(siftAccumulator_, 0, sizeof(siftAccumulator_));
    keyFileReaderStream_ = 0;
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
        fscanf(keyFile, "%*f%*f%*f%*f"); //ignoring sift headers
        SiftDataPtr rowVector = newImage.siftData.elements + kDimSiftData * i;
        for(int j = 0; j < kDimSiftData; j++) {
            fscanf(keyFile, "%f", &rowVector[j]);
            siftAccumulator_[j] = siftAccumulator_[j] + rowVector[j];
        }
        cntTotalVector_++;
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
    SiftData_t mean[kDimSiftData];

    for(int i = 0; i < kDimSiftData; i++) {
        mean[i] = siftAccumulator_[i] / cntTotalVector_;
    }

    std::vector<ImageHost>::iterator it;

    for(it = h_imageList_.begin(); it != h_imageList_.end(); ++it) {
        for(int i = 0; i < it->cntPoint; i++) {
            SiftDataPtr rowVector = &it->siftData(i, 0);
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
                 h_imageList_[index].siftData.height,
                 cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR;
}

cudaEvent_t KeyFileReader::UploadImageAsync( ImageDevice &d_Image, const int index, cudaEvent_t sync ) {
    if(keyFileReaderStream_ == 0) {
        cudaStreamCreate(&keyFileReaderStream_);
    }

    if(sync) {
        cudaStreamWaitEvent(keyFileReaderStream_, sync, 0);
    }

    d_Image.cntPoint = h_imageList_[index].cntPoint;
    d_Image.siftData.width = kDimSiftData;
    d_Image.siftData.height = h_imageList_[index].cntPoint;

    cudaMallocPitch(&(d_Image.siftData.elements),
                    &(d_Image.siftData.pitch),
                    d_Image.siftData.width * sizeof(SiftData_t),
                    d_Image.siftData.height);

    cudaMemcpy2DAsync(d_Image.siftData.elements,
                      d_Image.siftData.pitch,
                      h_imageList_[index].siftData.elements,
                      h_imageList_[index].siftData.pitch,
                      h_imageList_[index].siftData.width * sizeof(SiftData_t),
                      h_imageList_[index].siftData.height,
                      cudaMemcpyHostToDevice,
                      keyFileReaderStream_);

    cudaEvent_t finish;
    cudaEventCreate(&finish);
    cudaEventRecord(finish, keyFileReaderStream_);

    CUDA_CHECK_ERROR;

    return finish;
}
