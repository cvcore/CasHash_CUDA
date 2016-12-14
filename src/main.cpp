#include <stdio.h>
#include <stdlib.h>
#include "KeyFileReader.h"
#include "HashConverter.h"

int main(int argc, char **argv) {
    char path1[256];

    if(argc != 3) {
        fprintf(stderr, "Usage: %s <list.txt> outfile\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    FILE *fp = fopen(argv[1], "r");
    if(fp == NULL) {
        fprintf(stderr, "Index file %s does not exist!\n", argv[1]);
        exit(EXIT_FAILURE);
    }

    KeyFileReader kfr1;
    ImageDataDevice imgDev;
    while(fscanf(fp, "%s", path1) > 0) {
        kfr1.Read(&imgDev, path1);
    }
    fclose(fp);

    return 0;
}
