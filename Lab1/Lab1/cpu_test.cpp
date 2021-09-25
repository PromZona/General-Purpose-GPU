#include <stdlib.h>
#include <stdio.h>
#include <chrono>

void processCPU(int n)
{
    double* arr = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++)
    {
        arr[i] = i % 1000;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++)
    {
        arr[i] = arr[i] * arr[i];
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = (finish - start);

    printf("%f ms\n", elapsed.count());
    free(arr);
}

int main()
{
    processCPU(1000);
    processCPU(10000);
    processCPU(100000);
    processCPU(1000000);
    processCPU(10000000);
    processCPU(100000000);
    return 0;
}