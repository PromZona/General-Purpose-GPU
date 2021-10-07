#ifdef WIN32
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif // WIN32

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void ssaaKernel(uchar4 * out, int wOut, int hOut, int widthSize, int heightSize)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for (int x = idx; x < wOut; x += offsetx) {
		for (int y = idy; y < hOut; y += offsety) {
			int3 result = { 0, 0, 0 };
			for (int i = 0; i < widthSize; i++) {
				for (int j = 0; j < heightSize; j++) {
					uchar4 point = tex2D(tex, x * widthSize + i, y * heightSize + j);
					result.x += point.x;
					result.y += point.y;
					result.z += point.z;
				}
			}
			int count = widthSize * heightSize;
			result.x /= count;
			result.y /= count;
			result.z /= count;
			out[y * wOut + x] = make_uchar4(result.x, result.y, result.z, 255);
		}
	}
}

void Process(std::string inputFile, std::string outputFile, int widthOut, int heightOut)
{
	int widthInput, heightInput;

	FILE* fp = fopen(inputFile.c_str(), "rb");

	fread(&widthInput, sizeof(int), 1, fp);
	fread(&heightInput, sizeof(int), 1, fp);
	uchar4* data = (uchar4*)malloc(sizeof(uchar4) * widthInput * heightInput);
	fread(data, sizeof(uchar4), widthInput * heightInput, fp);
	fclose(fp);

	// Подготовка данных для текстуры
	cudaArray* arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, widthInput, heightInput));

	CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * widthInput * heightInput, cudaMemcpyHostToDevice));

	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;

	CSC(cudaBindTextureToArray(tex, arr, ch));

	uchar4* dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * widthOut * heightOut));

	int widthSize = widthInput / widthOut;
	int heightSize = heightInput / heightOut;


	cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));

	ssaaKernel << <dim3(16, 16), dim3(16, 16) >> > (dev_out, widthOut, heightOut, widthSize, heightSize);
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));

	printf("time = %f ms\n", t);


	CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * widthOut * heightOut, cudaMemcpyDeviceToHost));

	// Отвязываем данные от текстурной ссылки
	CSC(cudaUnbindTexture(tex));

	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

	fp = fopen(outputFile.c_str(), "wb");
	fwrite(&widthOut, sizeof(int), 1, fp);
	fwrite(&heightOut, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), widthOut * heightOut, fp);
	fclose(fp);

	free(data);
}

int main()
{
	Process("sample.data", "sampleOut.data", 250, 250);
    return 0;
}
