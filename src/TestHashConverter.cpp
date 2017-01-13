#include <iostream>
#include <cuda_runtime.h>

#include "HashConverter.h"
#include "KeyFileReader.h"
#include "Share.h"

int main(int argc, char *argv[]) {
    ImageDevice d_Img;
    KeyFileReader kf;
    std::cout << "reading keylist\n";
    kf.OpenKeyList(argv[1]);

    std::cout << "Vector #0 in host image #0:\n";
    dumpHostArray(&kf.h_imageList_[0].siftData(0, 0), kDimSiftData);

    std::cout << "vector #1 in image #0:\n";
    dumpHostArray(&kf.h_imageList_[0].siftData(1, 0), kDimSiftData);

    std::cout << "Removing DC components..\n";
    kf.ZeroMeanProc();
    std::cout << "uploading image\n";
    kf.UploadImage(d_Img, 0);

    std::cout << "Vector #0 in host image #0:\n";
    dumpHostArray(&kf.h_imageList_[0].siftData(0, 0), kDimSiftData);

    std::cout << "Vector #0 in device image #0:\n";
    dumpDeviceArray(&d_Img.siftData(0, 0), kDimSiftData);

    std::cout << "HashConverter instantiated\n";
    HashConverter hc;

    std::cout << "Calculating comphash...\n";
    hc.CompHash(d_Img);
    CUDA_CHECK_ERROR;
    std::cout << "Constructing buckets...\n";
    hc.BucketHash(d_Img);

    std::cout << "Bucket information for image #0:\n";
    dumpDeviceArray(&d_Img.bucketIDList(0, 0), kCntBucketGroup);

    cudaFree(d_Img.compHashData.elements);
    CUDA_CHECK_ERROR;
    cudaFree(d_Img.siftData.elements);
    CUDA_CHECK_ERROR;

    return 0;
}
