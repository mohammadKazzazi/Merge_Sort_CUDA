//Do NOT MODIFY THIS FILE

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include "gpuerrors.h"
#include "merge_sort.h"

// ===========================> Functions Prototype <===============================
double calc_mse (int* data1, int* data2, int size);
void gpuKernel(const int *f, int *result, int n, double* gpu_kernel_time);
void merge(int* arr, int* temp, int left, int mid, int right);
void mergeSortCPU(int* arr, int* temp, int left, int right);
// =================================================================================


//MAIN PROGRAM
int main(int argc, char** argv)
{   

    int n, size;
    n = atoi(argv[1]);
    size = (1 << n);
    
    //Create CPU based Arrays
    int *arr = (int*)malloc(size * sizeof(int));
    int *result_gpu = (int*)malloc(size * sizeof(int));
    int *result_cpu = (int*)malloc(size * sizeof(int));

    // Initialize the array with random values
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < size; ++i){
        arr[i] = rand() % 10000;
        result_cpu[i] = arr[i];
    }

    // GPU calculations
    double gpu_kernel_time = 0.0;
    clock_t t1 = clock();
    gpuKernel (arr, result_gpu, size, &gpu_kernel_time);
    clock_t t2 = clock();
    
    free(arr);

    int *temp = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i)
        temp[i] = 0;
    mergeSortCPU(result_cpu, temp, 0, size-1);

    // check correctness of GPU calculations against CPU
    double mse = 0.0;
    mse += calc_mse(result_cpu, result_gpu, size);

    printf("m=%d n=%d GPU=%g ms GPU-Kernel=%g ms mse=%g\n",
        n, size, (t2-t1)/1000.0, gpu_kernel_time, mse);

    //Destroy all variables
    free(temp);
    free(result_gpu);
    free(result_cpu);

    return 0;
}

double calc_mse (int* data1, int* data2, int size) {
	double mse = 0.0;
	int i; for (i=0; i<size; i++) {
		double e = data1[i]-data2[i];
		e = e * e;
		mse += e;
	}
	return mse;
}

//CPU Merge Recursive Call function
void merge(int* arr, int* temp, int left, int mid, int right) 
{
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) 
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i <= mid)
        temp[k++] = arr[i++];

    while (j <= right)
        temp[k++] = arr[j++];

    for (int idx = left; idx <= right; ++idx)
        arr[idx] = temp[idx];
}

//CPU Implementation of Merge Sort
void mergeSortCPU(int* arr, int* temp, int left, int right) 
{
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    mergeSortCPU(arr, temp, left, mid);
    mergeSortCPU(arr, temp, mid + 1, right);

    merge(arr, temp, left, mid, right);
}
