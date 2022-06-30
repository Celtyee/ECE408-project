#include <cmath>
#include <iostream>
#include "./gpu-new-forward.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"


#define BLOCK_SIZE 16
#define ImageNum 20


//__constant__ float ConstK[OutputfeatureMax * InputfeatureMax * KernelSize * KernelSize];
//__constant__ float Half_ConstK[OutputfeatureMax * InputfeatureMax * KernelSize * KernelSize];


__global__ void conv_forward_kernel(float* device_y, float* unroll_x, const float* device_k,
	const int B, const int M, const int C, const int H, const int W, const int K, const int sampleID)
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
	K - kernel height and width (K x K) = 7
	*/
	const int H_out = H - K + 1;
	const int W_out = W - K + 1;

	// We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a

	__shared__ float k_s[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float x_s[ImageNum][BLOCK_SIZE][BLOCK_SIZE];
	int output_x = blockIdx.x * blockDim.x + threadIdx.x;
	int output_y = blockIdx.y * blockDim.y + threadIdx.y;
	int tx, ty;
	int numArow = M;
	int numAcolumn = C * K * K;
	int numBrow = numAcolumn;
	int numBcolumn = W_out * H_out;
	int relative_pos = blockIdx.z;


	tx = threadIdx.x;
	ty = threadIdx.y;
	float c_values = 0;
	for (int i = 0; i < ((numAcolumn - 1) / BLOCK_SIZE + 1); i++)
	{
		if (i * BLOCK_SIZE + tx < numAcolumn && output_y < numArow)
		{
			k_s[ty][tx] = device_k[output_y * numAcolumn + i * BLOCK_SIZE + tx];
		}
		else
		{
			k_s[ty][tx] = 0;
		}
		if (i * BLOCK_SIZE + ty < numBrow && output_x < numBcolumn && (sampleID + relative_pos) < B)
		{
			x_s[relative_pos][ty][tx] = unroll_x[relative_pos * (numBcolumn * numBrow) + (i * BLOCK_SIZE + ty) * numBcolumn + output_x];
		}
		else
		{
			x_s[relative_pos][ty][tx] = 0;
		}
		__syncthreads();
		for (int i = 0; i < BLOCK_SIZE; i++)
		{
			c_values += k_s[ty][i] * x_s[relative_pos][i][tx];
		}
		__syncthreads();
	}
	if (output_y < numArow && output_x < numBcolumn && (sampleID + relative_pos) < B)
	{
		device_y[(sampleID + relative_pos) * (M * W_out * H_out) + output_y * (W_out * H_out) + output_x] = c_values;
	}

}

__global__ void Input_matrix_unroll(const float* device_x, float* unrolled_x,
	int C, int K, int W, int H, int sampleID)
{
	const int W_out = W - K + 1;
	const int H_out = H - K + 1;

#define x4d(i3, i2, i1, i0) device_x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + (i0)]
#define unrollx3d(i2,i1,i0) unrolled_x[(i2) * (C*K*K*W_out*H_out) + (i1) * ( W_out * H_out ) + (i0)]
	int w = blockIdx.x * blockDim.x + threadIdx.x;
	int h = blockIdx.y * blockDim.y + threadIdx.y;
	int n = blockIdx.z + sampleID;
	if (w < W_out & h < H_out)
	{
		for (int c = 0; c < C; c++)
		{
			for (int p = 0; p < K; p++)
			{
				for (int q = 0; q < K; q++)
				{
					unrollx3d(blockIdx.z, (c * K * K) + (p * K) + q, h * W_out + w) = x4d(n, c, h + p, w + q);
				}
			}
		}
	}
#undef x4d
#undef unrollx3d
}

// dim3 HMU_gridDim(ceil,ceil,M)
// dim3 HMU_blockDim(BLOCK_SIZE,BLOCK_SIZE,1)

//__global__ void Kernel_unroll(const float* device_k, float* unrolled_k,
//	int C, int K, int M)
//{
//#define k4d(i3, i2, i1, i0) device_k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//#define unrollk2d(i1,i0) unrolled_k[ (i1) * (C*K*K) + (i0)]
//	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
//	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
//	int c = blockIdx.z;
//	if (idx_x < K && idx_y < K)
//	{
//		for (int m = 0; m < M; m++)
//		{
//			unrollk2d(m, c * (K * K) + idx_y * K + idx_x) = k4d(m, c, idx_y, idx_x);
//		}
//	}
//#undef k4d
//#undef unrollk2d
//}




//__global__ void FeatureMap_mat_roll(float* device_y, float* unrolled_y,
//	int M, int W_out, int H_out)
//{
//#define y4d(i3, i2, i1, i0) device_y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//#define unrolly3d(i2,i1,i0)	unrolled_y[(i2)*(M*W_out*H_out) + (i1)*(W_out*H_out) + (i0)]
//	int w = blockIdx.x * blockDim.x + threadIdx.x;
//	int h = blockIdx.y * blockDim.y + threadIdx.y;
//	int n = blockIdx.z;
//	if (w < W_out && h < H_out)
//	{
//		for (int m = 0; m < M; m++)
//		{
//			y4d(n, m, h, w) = unrolly3d(n, m, h * W_out + w);
//		}
//	}
//#undef y4d
//#undef unrolly2d
//	return;
//}


__host__ void GPUInterface::conv_forward_gpu_prolog(float* host_y, float* host_x, float* host_k, float** device_y_ptr, float** device_x_ptr, float** device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
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

	// input unroll
}



// to see acceleration, using nsys statistic in ms form of kernel call
__host__ void GPUInterface::conv_forward_gpu(float* device_y, const float* device_x, const float* device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
	// Set the kernel dimensions and call the kernel
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int Numphoto = 25;

	float* unroll_device_k;
	float* unroll_device_x;
	float* unroll_device_y;

	// cudaMalloc(&unroll_device_k, sizeof(float) * C * K * K * M);
	cudaMalloc(&unroll_device_x, sizeof(float) * ImageNum * C * K * K * W_out * H_out);
	// cudaMalloc(&unroll_device_y, sizeof(float) * B * M * W_out * H_out);

	/*std::cout << "value of K " << K << std::endl;
	std::cout << "value of C " << C << std::endl;
	std::cout << "value of H " << H << std::endl;
	std::cout << "value of W " << W << std::endl;
	std::cout << "value of M " << M << std::endl;*/

	dim3 IMU_gridDim(ceil(W_out / (1.0 * BLOCK_SIZE)), ceil(H_out / (1.0 * BLOCK_SIZE)), ImageNum);
	dim3 IMU_blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);


	// dim3 KU_gridDim(ceil(K / (1.0 * BLOCK_SIZE)), ceil(K / (1.0 * BLOCK_SIZE)), M);
	// dim3 KU_blockDim(BLOCK_SIZE, BLOCK_SIZE,1);
	// Kernel_unroll << < KU_gridDim, KU_blockDim >> > (device_k, unroll_device_k, C, K, M);
	// cudaDeviceSynchronize();

	dim3 CFK_gridDim(ceil((W_out * H_out) / (1.0 * BLOCK_SIZE)), ceil(M / (1.0 * BLOCK_SIZE)), ImageNum);
	dim3 CFK_blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
	for (int sampleID = 0; sampleID < B; sampleID += ImageNum)
	{
		Input_matrix_unroll << < IMU_gridDim, IMU_blockDim >> > (device_x, unroll_device_x, C, K, W, H, sampleID);
		cudaDeviceSynchronize();
		conv_forward_kernel << < CFK_gridDim, CFK_blockDim >> > (device_y, unroll_device_x, device_k, B, M, C, H, W, K, sampleID);
		cudaDeviceSynchronize();
	}

	cudaFree(unroll_device_x);
	// dim3 FMR_gridDim(ceil(W_out * H_out / (1.0 * BLOCK_SIZE)), ceil(M / (1.0 * BLOCK_SIZE)), B);
	// dim3 FMR_blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
	// FeatureMap_mat_roll << < FMR_gridDim, FMR_blockDim >> > (device_y, unroll_device_y, M, W_out, H_out);
	// cudaDeviceSynchronize();


	//cudaFree(unroll_device_y);
	//cudaFree(unroll_device_k);


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
