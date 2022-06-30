#include <cmath>
#include <iostream>
#include "./gpu-new-forward.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16

#define KernelSize 7
#define InputfeatureMax 4
#define OutputfeatureMax 16

__constant__ float ConstK[OutputfeatureMax * InputfeatureMax * KernelSize * KernelSize];


__global__ void conv_forward_kernel(float* y, const float* x, const float* k,
	const int B, const int M, const int C, const int H, const int W, const int K)
{
	/*
	Modify this function to implement the forward pass described in Chapter 16.
	We have added an additional dimension to the tensors to support an entire mini-batch
	The goal here is to be correct AND fast.

	Function paramter definitions:
	y - output
	x - input
	k - kernel
	B - batch_size (number of images in x)
	M - number of output feature maps
	C - number of input feature maps
	H - input height dimension
	W - input width dimension
	K - kernel height and width (K x K)
	*/
	const int H_out = H - K + 1;
	const int W_out = W - K + 1;

	// We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define constk4d(i3, i2, i1, i0) ConstK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int W_grid = ceil(W_out / (1.0 * blockDim.x));
	//int H_grid = ceil(1.0 * W_out / blockDim.y);
	// Insert your GPU convolution kernel code here
	// the n simple
	int n = blockIdx.x;
	// the m_th output feature
	int m = blockIdx.y;
	// the absolute index
	int h = (blockIdx.z / W_grid) * BLOCK_SIZE + threadIdx.y;
	int w = (blockIdx.z % W_grid) * BLOCK_SIZE + threadIdx.x;
	if (w < W_out && h < H_out)
	{
		float acc = 0;
		for (int c = 0; c < C; c++)
		{
			for (int p = 0; p < K; p++)
				for (int q = 0; q < K; q++)
					acc += x4d(n, c, h + p, w + q) * constk4d(m, c, p, q);
		}
		y4d(n, m, h, w) = acc;
	}

#undef y4d
#undef x4d
	//#undef k4d
#undef constk4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(float* host_y, float* host_x, float* host_k, float** device_y_ptr, float** device_x_ptr, float** device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
	// Allocate memory and copy over the relevant data structures to the GPU
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int Numphoto = B / NumStream;
	int X_segSize = Numphoto * C * H * W;
	int Y_segSize = Numphoto * M * W_out * H_out;
	host_x_str = host_x;
	host_y_str = host_y;
	for (int i = 0; i < NumStream; i++)
	{
		cudaMalloc(&d_X[i], sizeof(float) * X_segSize);
		cudaMalloc(&d_Y[i], sizeof(float) * Y_segSize);
	}
	/*cudaMalloc(&d_X0, sizeof(float) * X_segSize);
	cudaMalloc(&d_X1, sizeof(float) * X_segSize);
	cudaMalloc(&d_X2, sizeof(float) * X_segSize);
	cudaMalloc(&d_X3, sizeof(float) * X_segSize);


	cudaMalloc(&d_Y0, sizeof(float) * Y_segSize);
	cudaMalloc(&d_Y1, sizeof(float) * Y_segSize);
	cudaMalloc(&d_Y2, sizeof(float) * Y_segSize);
	cudaMalloc(&d_Y3, sizeof(float) * Y_segSize);*/


	cudaMemcpyToSymbol(ConstK, host_k, sizeof(float) * M * C * K * K);
	//cudaMalloc(device_x_ptr, sizeof(float) * B * C * H * W);
	//cudaMalloc(device_y_ptr, sizeof(float) * B * M * W_out * H_out);
	//cudaMalloc(device_k_ptr, sizeof(float) * M * C * K * K);

	/*cudaMemcpy(device_x_str, host_x, sizeof(float) * B * C * H * W, cudaMemcpyHostToDevice);*/
	//cudaMemcpy(*device_k_ptr, host_k, sizeof(float) * M * C * K * K, cudaMemcpyHostToDevice);
	// We pass double pointers for you to initialize the relevant device pointers,
	//  which are passed to the other two functions.

	// Useful snippet for error checking
	// cudaError_t error = cudaGetLastError();
	// if(error != cudaSuccess)
	// {
	//     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
	//     exit(-1);
	// }
}


__host__ void GPUInterface::conv_forward_gpu(float* device_y, const float* device_x, const float* device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
	cudaStream_t  stream_array[NumStream];
	for (int i = 0; i < NumStream; i++)
	{
		cudaStreamCreate(&stream_array[i]);
	}
	//cudaStreamCreate(&stream0);
	//cudaStreamCreate(&stream1);
	//cudaStreamCreate(&stream2);
	//cudaStreamCreate(&stream3);
	// Set the kernel dimensions and call the kernel
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int Numphoto = B / NumStream;

	int X_segSize = Numphoto * C * H * W;
	int Y_segSize = Numphoto * M * W_out * H_out;

	int W_grid = ceil(W_out / (1.0 * BLOCK_SIZE));
	int H_grid = ceil(H_out / (1.0 * BLOCK_SIZE));
	int Z = W_grid * H_grid;
	dim3 block_Dim(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 grid_Dim(Numphoto, M, Z);

	for (int i = 0; i < NumStream; i++)
	{
		cudaMemcpyAsync(d_X[i], host_x_str + X_segSize * i, sizeof(float) * X_segSize, cudaMemcpyHostToDevice, stream_array[i]);
		conv_forward_kernel << < grid_Dim, block_Dim, 0, stream_array[i] >> > (d_Y[i], d_X[i], device_k, B, M, C, H, W, K);

	}
	/*cudaMemcpyAsync(d_X0, host_x_str, sizeof(float) * X_segSize, cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(d_X1, host_x_str + X_segSize, sizeof(float) * X_segSize, cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_X2, host_x_str + X_segSize * 2, sizeof(float) * X_segSize, cudaMemcpyHostToDevice, stream2);
	cudaMemcpyAsync(d_X3, host_x_str + X_segSize * 3, sizeof(float) * X_segSize, cudaMemcpyHostToDevice, stream3);*/

	/*conv_forward_kernel << < grid_Dim, block_Dim, 0, stream0 >> > (d_Y0, d_X0, device_k, B, M, C, H, W, K);
	conv_forward_kernel << < grid_Dim, block_Dim, 0, stream1 >> > (d_Y1, d_X1, device_k, B, M, C, H, W, K);
	conv_forward_kernel << < grid_Dim, block_Dim, 0, stream2 >> > (d_Y2, d_X2, device_k, B, M, C, H, W, K);
	conv_forward_kernel << < grid_Dim, block_Dim, 0, stream3 >> > (d_Y3, d_X3, device_k, B, M, C, H, W, K);*/

	for (int i = 0; i < NumStream; i++)
	{
		cudaMemcpyAsync(host_y_str + i * Y_segSize, d_Y[i], sizeof(float) * Y_segSize, cudaMemcpyDeviceToHost, stream_array[i]);
	}

	/*cudaMemcpyAsync(host_y_str, d_Y0, sizeof(float) * Y_segSize, cudaMemcpyDeviceToHost, stream0);
	cudaMemcpyAsync(host_y_str + Y_segSize, d_Y1, sizeof(float) * Y_segSize, cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(host_y_str + Y_segSize * 2, d_Y2, sizeof(float) * Y_segSize, cudaMemcpyDeviceToHost, stream2);
	cudaMemcpyAsync(host_y_str + Y_segSize * 3, d_Y3, sizeof(float) * Y_segSize, cudaMemcpyDeviceToHost, stream3);*/

	//cudaDeviceSynchronize();

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float* host_y, float* device_y, float* device_x, float* device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
	// Copy the output back to host
	// int H_out = H - K + 1;
	// int W_out = W - K + 1;
	// int Numphoto = B / NumStream;
	// int X_segSize = Numphoto * C * H * W;
	// int Y_segSize = Numphoto * M * W_out * H_out;
	// cudaMemcpy(host_y, device_y_str, sizeof(float) * B * M * W_out * H_out, cudaMemcpyDeviceToHost);
	// Free device memory
	for (int i = 0; i < NumStream; i++)
	{
		cudaFree(d_X[i]);
		cudaFree(d_Y[i]);
	}
	//cudaFree(d_X0);
	//cudaFree(d_X1);
	//cudaFree(d_X2);
	//cudaFree(d_X3);

	//cudaFree(d_Y0);
	//cudaFree(d_Y1);
	//cudaFree(d_Y2);
	//cudaFree(d_Y3);
}


__host__ void GPUInterface::get_device_properties()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
		std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
		std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
		std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
		std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
		std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
		std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
		std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
		std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
	}
}
