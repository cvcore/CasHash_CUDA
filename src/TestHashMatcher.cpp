#include <iostream>
#include <cuda_runtime.h>

#include "HashMatcher.h"
#include "HashConverter.h"
#include "KeyFileReader.h"
#include "Share.h"

int main(int argc, char *argv[]) {
    
    KeyFileReader kf;
    std::cerr << "reading keylist\n";
    kf.OpenKeyList(argv[1]);
    std::cerr << "preprocessing to zero-mean vectors\n";
    kf.ZeroMeanProc();

    std::cerr << "initialing cuda device\n";
    cudaSetDevice(0);

    std::cerr << "filling hash matrix" << '\n';
    HashConverter hc;

    HashMatcher hm;

    for (int i = 0; i < kf.cntImage; i++) {
        ImageDevice curImg;

        std::cerr << "------------------\nuploading image " << i << "\n";
        kf.UploadImage(curImg, i);

        std::cerr << "Converting hash values\n";
        hc.CalcHashValues(curImg);
        //dumpDeviceArray(&curImg.compHashData(0, 0), 2);
        //dumpDeviceArray(&curImg.bucketIDList(0, 0), 6);

        std::cerr << "Adding image to hashmatcher\n";
        hm.AddImage(curImg);

        for(int j = 0; j < i; j++) {
            std::cerr << hm.NumberOfMatch(i, j) << " match(es) found between image " << i << " and " << j << "\n";
#ifdef DEBUG_HASH_MATCHER
            MatchPairListPtr mpList = hm.MatchPairList(i, j);
            for(MatchPairList_t::iterator it = mpList->begin(); it != mpList->end(); it++) {
                std::cerr << "(" << it->first << ", " << it->second << ") ";
            }
            std::cerr << std::endl;
#endif
        }

    }

    return 0;
}
