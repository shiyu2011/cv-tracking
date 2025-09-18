#include <stdio.h>
__global__ void SimplePrintKernel(float a, float b, float *c) {
    *c = a + b;
    printf("The sum of %f and %f is %f\n", a, b, *c);
}