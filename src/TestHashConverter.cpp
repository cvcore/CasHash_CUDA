#include <iostream>
#include <cuda_runtime.h>

#include "HashConverter.h"
#include "KeyFileReader.h"
#include "Share.h"

int main(int argc, char *argv[]) {
    ImageHost h_Img;
    ImageDevice d_Img;
    KeyFileReader kf;
    kf.OpenKeyList(argv[1]);
    kf.UploadImage(d_Img, 0);

    std::cout << d_Img.siftData.elements << std::endl;

    HashConverter hc;
    hc.CompHash(d_Img);
    CUDA_CHECK_ERROR;
    CUDA_CHECK_ERROR;
    std::cout << d_Img.siftData.elements << std::endl;
    cudaFree(d_Img.compHashData.elements);
    CUDA_CHECK_ERROR;
    cudaFree(d_Img.siftData.elements);
    CUDA_CHECK_ERROR;

    return 0;
}
