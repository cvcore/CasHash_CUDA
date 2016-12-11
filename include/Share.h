#pragma once

const int kDimSiftData = 128; // the number of dimensions of SIFT feature
const int kDimHashData = 128; // the number of dimensions of Hash code
const int kBitInCompHash = 64; // the number of Hash code bits to be compressed; in this case, use a <uint64_t> variable to represent 64 bits
const int kDimCompHashData = kDimHashData / kBitInCompHash; // the number of dimensions of CompHash code
const int kMinMatchListLen = 16; // the minimal list length for outputing SIFT matching result between two images
const int kMaxCntPoint = 1000000; // the maximal number of possible SIFT points; ensure this value is not exceeded in your application

const int kCntBucketBit = 8; // the number of bucket bits
const int kCntBucketGroup = 6; // the number of bucket groups
const int kCntBucketPerGroup = 1 << kCntBucketBit; // the number of buckets in each group

const int kCntCandidateTopMin = 6; // the minimal number of top-ranked candidates
const int kCntCandidateTopMax = 10; // the maximal number of top-ranked candidates

typedef int SiftData_t;
typedef int* SiftDataPtr; // SIFT feature is represented with <int> type
typedef uint8_t* HashDataPtr; // Hash code is represented with <uint8_t> type; only the lowest bit is used
typedef uint64_t* CompHashDataPtr; // CompHash code is represented with <uint64_t> type
typedef int* BucketElePtr; // index list of points in a specific bucket

typedef struct {
    int cntPoint; // the number of SIFT points
    char keyFilePath[100]; // the path to SIFT feature file
    SiftDataPtr* siftDataPtrList; // SIFT feature for each SIFT point
    HashDataPtr* hashDataPtrList; // Hash code for each SIFT point
    CompHashDataPtr* compHashDataPtrList; // CompHash code for each SIFT point
    uint16_t* bucketIDList[kCntBucketGroup]; // bucket entries for each SIFT point
    int cntEleInBucket[kCntBucketGroup][kCntBucketPerGroup]; // the number of SIFT points in each bucket
    BucketElePtr bucketList[kCntBucketGroup][kCntBucketPerGroup]; // SIFT point index list for all buckets
}   ImageData; // all information needed for an image to perform CasHash-Matching

typedef struct {
    int cntPoint;
    SiftDataPtr siftArray;
    int siftArrayPitch;
    HashDataPtr* hashDataPtrList;
    CompHashDataPtr* compHashDataPtrList;
    uint16_t* bucketIDList[kCntBucketGroup];
    int cntEleInBucket[kCntBucketGroup][kCntBucketPerGroup];
    BucketElePtr bucketList[kCntBucketGroup][kCntBucketPerGroup];
} ImageDataDevice;

typedef std::vector<std::pair<int, int> > MatchList; // SIFT point match list between two images

