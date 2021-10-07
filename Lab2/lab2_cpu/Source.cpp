#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <iostream>
#include <string>
#include <chrono>



struct uchar4
{
	unsigned char x;
	unsigned char y;
	unsigned char z;
	unsigned char w;
};

void ssaaKernel(uchar4* in, uchar4* out, int wIn, int wOut, int hOut, int widthSize, int heightSize)
{
	for (int x = 0; x < wOut; x++) {
		for (int y = 0; y < hOut; y++) {
			int xRes = 0;
			int yRes = 0;
			int zRes = 0;

			for (int i = 0; i < widthSize; i++) {
				for (int j = 0; j < heightSize; j++) {
					int xx = x * widthSize + i;
					int yy = y * heightSize + j;
					
					uchar4 point = in[yy * wIn + xx];
					xRes += point.x;
					yRes += point.y;
					zRes += point.z;
				}
			}
			int count = widthSize * heightSize;
			xRes /= count;
			yRes /= count;
			zRes /= count;
			out[y * wOut + x] = uchar4{ (unsigned char)xRes,(unsigned char)yRes, (unsigned char)zRes, 0 };
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

	uchar4* dev_out = (uchar4*)malloc(sizeof(uchar4) * widthOut * heightOut);

	int widthSize = widthInput / widthOut;
	int heightSize = heightInput / heightOut;

	auto start = std::chrono::high_resolution_clock::now();
	ssaaKernel(data, dev_out, widthInput, widthOut, heightOut, widthSize, heightSize);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed = (finish - start);
	printf("%f ms\n", elapsed.count());

	free(data);
	free(dev_out);
}

int main()
{
	Process("10x10.data", "5x5out.data", 5, 5);
	Process("50x50.data", "25x25out.data", 25, 25);
	Process("100x100.data", "50x50out.data", 50, 50);
	Process("250x250.data", "125x125out.data", 125, 125);
	Process("500x500.data", "250x250out.data", 250, 250);
	Process("1000x1000.data", "500x500out.data", 500, 500);
    return 0;
}