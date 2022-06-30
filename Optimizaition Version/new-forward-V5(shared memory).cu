#include <cmath>
#include <iostream>
#include "./gpu-new-forward.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16
__global__ void conv_forward_kernel(float* y, const float* x, const float* k, const int B, const int M, const int C, const int H, const int W, const int K)
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

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int W_grid = ceil(W_out / (1.0 * blockDim.x));
	int Tile_width = BLOCK_SIZE + K - 1;
	extern __shared__ float shmem[];
	float* X_shared = &shmem[0];
	float* W_shared = &shmem[Tile_width * Tile_width];

	//int H_grid = ceil(1.0 * W_out / blockDim.y);
	// Insert your GPU convolution kernel code here
	// the n simple
	int n = blockIdx.x;
	// the m_th output feature
	int m = blockIdx.y;
	// the absolute index
	int h_base = (blockIdx.z / W_grid) * BLOCK_SIZE;
	int w_base = (blockIdx.z % W_grid) * BLOCK_SIZE;
	int h = (blockIdx.z / W_grid) * BLOCK_SIZE + threadIdx.y;
	int w = (blockIdx.z % W_grid) * BLOCK_SIZE + threadIdx.x;
	float acc = 0;
	for (int c = 0; c < C; c++)
	{
		if ((threadIdx.x < K) && (threadIdx.y < K))
			W_shared[threadIdx.y * K + threadIdx.x] = k4d(m, c, threadIdx.y, threadIdx.x);
		__syncthreads();

		for (int p = h; p < h_base + Tile_width; p += BLOCK_SIZE)
		{
			for (int q = w; q < w_base + Tile_width; q += BLOCK_SIZE)
			{
				X_shared[(p - h_base) * Tile_width + q - w_base] = x4d(n, c, p, q);
			}
		}

		__syncthreads();
		for (int p = 0; p < K; p++)
		{
			for (int q = 0; q < K; q++)
				acc += X_shared[(threadIdx.y+p) * Tile_width + (threadIdx.x + q)] * W_shared[p * K + q];
		}
		__syncthreads();
	}
	if (w < W_out && h < H_out)
	{
		y4d(n, m, h, w) = acc;
	}

#undef y4d
#undef x4d
#undef k4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(float* host_y, const float* host_x, const float* host_k, float** device_y_ptr, float** device_x_ptr, float** device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
	// Allocate memory and copy over the relevant data structures to the GPU
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	cudaMalloc(device_x_ptr, sizeof(float) * B * C * H * W);
	cudaMalloc(device_y_ptr, sizeof(float) * B * M * W_out * H_out);
	cudaMalloc(device_k_ptr, sizeof(float) * M * C * K * K);
	cudaMemcpy(*device_x_ptr, host_x, sizeof(float) * B * C * H * W, cudaMemcpyHostToDevice);
	cudaMemcpy(*device_k_ptr, host_k, sizeof(float) * M * C * K * K, cudaMemcpyHostToDevice);
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
	// Set the kernel dimensions and call the kernel
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int W_grid = ceil(W_out / (1.0 * BLOCK_SIZE));
	int H_grid = ceil(H_out / (1.0 * BLOCK_SIZE));
	int Z = W_grid * H_grid;
	dim3 block_Dim(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 grid_Dim(B, M, Z);
	conv_forward_kernel << < grid_Dim, block_Dim >> > (device_y, device_x, device_k, B, M, C, H, W, K);
	cudaDeviceSynchronize();

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float* host_y, float* device_y, float* device_x, float* device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
	// Copy the output back to host
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	cudaMemcpy(host_y, device_y, sizeof(float) * B * M * W_out * H_out, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(device_k);
	cudaFree(device_x);
	cudaFree(device_y);
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
