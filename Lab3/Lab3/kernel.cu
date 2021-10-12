#ifdef WIN32
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

#define CSC(call) 						\
do {									\
	cudaError_t	status = call;			\
	if (status != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Massage: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));			\
		exit(0);						\
	}									\
} while(0)

__constant__ double NORM_AVG[32][3];

__device__ int get_class(uchar4 pixel, int nc)
{
	double res[32]{};
	for (int i = 0; i < nc; i++)
	{
		res[i] = pixel.x * NORM_AVG[i][0] + pixel.y * NORM_AVG[i][1] + pixel.z * NORM_AVG[i][2];
	}
	double maxEl = res[0];
	int class_index = 0;
	for (int i = 0; i < nc; i++)
	{
		if (res[i] > maxEl)
		{
			maxEl = res[i];
			class_index = i;
		}
	}
	return class_index;
}

__global__ void sam_kernel(uchar4 * data, int w, int h, int nc)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for (int x = idx; x < w; x += offsetx)
	{
		for (int y = idy; y < h; y += offsety)
		{
			data[x + y * w].w = get_class(data[x + y * w], nc);
			// data[x + y * w].x = 255;
			// data[x + y * w].y = 200;
			// data[x + y * w].z = 100;
			// data[x + y * w].w = 2;
		}
	}
}

int main()
{
	std::string inputFile;
	std::string outputFile;
	int widthInput, heightInput;
	int nc;

	std::cin >> inputFile >> outputFile >> nc;
	std::vector<std::vector<int2>> classes(nc);
	for (int i = 0; i < nc; i++)
	{
		int np;
		std::cin >> np;
		classes[i].resize(np);
		for (int j = 0; j < np; j++) {
			std::cin >> classes[i][j].x >> classes[i][j].y;
		}
	}

	// Image reading
	FILE* fp = fopen(inputFile.c_str(), "rb");
	fread(&widthInput, sizeof(int), 1, fp);
	fread(&heightInput, sizeof(int), 1, fp);
	size_t image_size = sizeof(uchar4) * widthInput * heightInput;
	uchar4* data = (uchar4*)malloc(image_size);
	fread(data, sizeof(uchar4), widthInput * heightInput, fp);
	fclose(fp);

	double avg[32][3];
	for (int i = 0; i < 32; i++)
		for (int j = 0; j < 3; j++)
			avg[i][j] = 0.0;

	for (int i = 0; i < nc; i++)
	{
		int points_count = classes[i].size();
		for (int j = 0; j < points_count; j++)
		{
			int2 point = classes[i][j];
			uchar4 pixel = data[point.x + point.y * widthInput];
			avg[i][0] += pixel.x;
			avg[i][1] += pixel.y;
			avg[i][2] += pixel.z;
		}
		avg[i][0] /= points_count;
		avg[i][1] /= points_count;
		avg[i][2] /= points_count;
	}

	for (int i = 0; i < nc; i++)
	{
		double r = avg[i][0];
		double g = avg[i][1];
		double b = avg[i][2];

		double norm = std::sqrt(r*r + g*g + b*b);

		avg[i][0] = r / norm;
		avg[i][1] = g / norm;
		avg[i][2] = b / norm;
	}

	CSC(cudaMemcpyToSymbol(NORM_AVG, avg, sizeof(double) * 32 * 3));

	uchar4* dev_out;
	CSC(cudaMalloc(&dev_out, image_size));
	CSC(cudaMemcpy(dev_out, data, image_size, cudaMemcpyHostToDevice));
	sam_kernel<<<dim3(16, 16), dim3(16, 16)>>> (dev_out, widthInput, heightInput, nc);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, dev_out, image_size, cudaMemcpyDeviceToHost));

	fp = fopen(outputFile.c_str(), "wb");
	fwrite(&widthInput, sizeof(int), 1, fp);
	fwrite(&heightInput, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), widthInput * heightInput, fp);
	fclose(fp);

	CSC(cudaFree(dev_out));

    return 0;
}