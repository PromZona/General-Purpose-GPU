#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

struct uchar4
{
	unsigned char x;
	unsigned char y;
	unsigned char z;
	unsigned char w;
};

struct int2
{
	int x;
	int y;
};

int get_class(uchar4 pixel, int nc, double norm_avg[32][3])
{
	double res[32]{};
	for (int i = 0; i < nc; i++)
	{
		res[i] = pixel.x * norm_avg[i][0] + pixel.y * norm_avg[i][1] + pixel.z * norm_avg[i][2];
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

void sam_kernel(uchar4* data, int w, int h, int nc, double norm_avg[32][3])
{
	for (int x = 0; x < w; x++)
	{
		for (int y = 0; y < h; y++)
		{
			data[x + y * w].w = get_class(data[x + y * w], nc, norm_avg);
		}
	}
}

void Process(std::string inputFile, std::string outputFile, int nc, std::vector<std::vector<int2>> classes)
{
	int widthInput, heightInput;

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

		double norm = std::sqrt(r * r + g * g + b * b);

		avg[i][0] = r / norm;
		avg[i][1] = g / norm;
		avg[i][2] = b / norm;
	}

	auto start = std::chrono::high_resolution_clock::now();
	sam_kernel(data, widthInput, heightInput, nc, avg);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed = (finish - start);
	printf("%f ms\n", elapsed.count());

	fp = fopen(outputFile.c_str(), "wb");
	fwrite(&widthInput, sizeof(int), 1, fp);
	fwrite(&heightInput, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), widthInput * heightInput, fp);
	fclose(fp);

	free(data);
}

int main()
{
	int nc;
	std::cin >> nc;
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

	Process("10x10.data", "10x10out.data", nc, classes);
	Process("50x50.data", "50x50out.data", nc, classes);
	Process("100x100.data", "100x100out.data", nc, classes);
	Process("250x250.data", "250x250out.data", nc, classes);
	Process("500x500.data", "500x500out.data", nc, classes);
	Process("1000x1000.data", "1000x1000out.data", nc, classes);
    return 0;
}