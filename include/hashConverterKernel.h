#pragma once

#include "Share.h"

void fillwithrand(Matrix &pMat);

__global__ void randNumKernel(int* elements, const size_t pitch);
