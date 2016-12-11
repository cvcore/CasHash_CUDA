#include "KeyFileReader.h"

int main() {
    if(argc != 3) {
        fprintf(stderr, "Usage: %s <list.txt> outfile", argv[0]);
    }
    KeyFileReader kfr1;
    kfr1.Read(argv[1]);
    return 0;
}
