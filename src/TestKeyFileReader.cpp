#include <Share.h>
#include <KeyFileReader.h>

int main(int argc, char *argv[]) {
    KeyFileReader kfr;

    if(argc != 3) {
        fprintf(stderr, "Usage: %s <list.txt> outfile\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    kfr.OpenKeyList(argv[1]);
    kfr.ZeroMeanProc();

    ImageDevice d_Img;
    for(int i = 0; i < kfr.cntImage; i++) {
        kfr.UploadImage(d_Img, i);
    }

    return 0;
}
 
