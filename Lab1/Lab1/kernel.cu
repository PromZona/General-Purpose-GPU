#ifdef WIN32
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif // WIN32

#include <stdlib.h>
#include <stdio.h>

#define CSC(call) 						\
do {									\
	cudaError_t	status = call;			\
	if (status != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Massage: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));			\
		exit(0);						\
	}									\
} while(0)


__global__ void powerKernel(double* arr, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (idx < n) {
        arr[idx] = arr[idx] * arr[idx];
        idx += offset;
    }
}

int main()
{
    int n;
    scanf("%d", &n);

    double* arr = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++)
    {
        double buff;
        scanf("%lf", &buff);
        arr[i] = buff;
    }

    double* cudaArr;
    CSC(cudaMalloc(&cudaArr, sizeof(double) * n));
    CSC(cudaMemcpy(cudaArr, arr, sizeof(double) * n, cudaMemcpyHostToDevice));

    powerKernel<<<256, 256>>>(cudaArr, n);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(arr, cudaArr, sizeof(double) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(cudaArr));
    for (int i = 0; i < n; i++)
        printf("%.10e ", arr[i]);
    printf("\n");
    free(arr);
    return 0;
}
