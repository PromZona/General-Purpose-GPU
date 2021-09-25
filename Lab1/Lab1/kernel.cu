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

void DefaultInput(double* arr,int n)
{
    for (int i = 0; i < n; i++)
    {
        double buff;
        scanf("%lf", &buff);
        arr[i] = buff;
    }
}

void GenerateData(double* arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        arr[i] = i % 1000;
    }
}

void process(int n)
{
    double* arr = (double*)malloc(sizeof(double) * n);
    // DefaultInput(arr, n);
    GenerateData(arr, n);

    double* cudaArr;
    CSC(cudaMalloc(&cudaArr, sizeof(double) * n));
    CSC(cudaMemcpy(cudaArr, arr, sizeof(double) * n, cudaMemcpyHostToDevice));

    cudaEvent_t start, end;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&end));
    CSC(cudaEventRecord(start));

    powerKernel << <1024, 1024 >> > (cudaArr, n);
    CSC(cudaGetLastError());

    CSC(cudaEventRecord(end));
    CSC(cudaEventSynchronize(end));
    float t;
    CSC(cudaEventElapsedTime(&t, start, end));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(end));

    printf("time = %f ms\n", t);

    CSC(cudaMemcpy(arr, cudaArr, sizeof(double) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(cudaArr));

    /*
    for (int i = 0; i < n; i++)
        printf("%.10e ", arr[i]);
    printf("\n");
    */
    free(arr);
}

int main()
{
    process(1000);
    process(10000);
    process(100000);
    process(1000000);
    process(10000000);
    process(100000000);
    return 0;
}
