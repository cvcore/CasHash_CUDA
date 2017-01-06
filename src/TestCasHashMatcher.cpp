#include <iostream>
#include <cuda_runtime.h>

#include "CasHashMatcher.h"
#include "HashConverter.h"
#include "KeyFileReader.h"
#include "Share.h"

int main(int argc, char *argv[]) {
    
    KeyFileReader kf;
    std::cout << "reading keylist\n";
    kf.OpenKeyList(argv[1]);

    std::cout << "hc\n";
    HashConverter hc;
    std::cout << "filling hash matrix" << '\n';

    std::vector<ImageDevice> d_imageList;

    for (int i = 0; i < kf.cntImage; i++){
	ImageDevice curImg;
	std::cout << "uploading image " << i << "\n";
	kf.Upload(curImg, i);
	hc.CalcHashValues(curImg);
    }

    // std::cout << d_Img.siftData.elements << std::endl;


    // 
    // CUDA_CHECK_ERROR;
    //
    // std::cout << d_Img.siftData.elements << std::endl;
    // cudaFree(d_Img.compHashData.elements);
    // CUDA_CHECK_ERROR;
    // cudaFree(d_Img.siftData.elements);
    // CUDA_CHECK_ERROR;


    return 0;
}
