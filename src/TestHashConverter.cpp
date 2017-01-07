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
    std::cout << "uploading image\n";
    kf.UploadImage(d_Img, 0);

    // std::cout << d_Img.siftData.elements << std::endl;

    std::cout << "hc\n";
    HashConverter hc;
    // hc.CompHash(d_Img);
    // CUDA_CHECK_ERROR;
    hc.BucketHash(d_Img);
    // std::cout << d_Img.siftData.elements << std::endl;
    // cudaFree(d_Img.compHashData.elements);
    CUDA_CHECK_ERROR;
    // cudaFree(d_Img.siftData.elements);
    // CUDA_CHECK_ERROR;


    return 0;
}
