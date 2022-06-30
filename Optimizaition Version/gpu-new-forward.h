#ifndef SRC_LAYER_GPU_NEW_FORWARD_H
#define SRC_LAYER_GPU_NEW_FORWARD_H

#define NumStream 16
class GPUInterface
{
public:
    void get_device_properties();
    void conv_forward_gpu_prolog(float* host_y, float* host_x, float* host_k, float** device_y_ptr, float** device_x_ptr, float** device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu(float* device_y, const float* device_x, const float* device_k, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu_epilog(float* host_y, float* device_y, float* device_x, float* device_k, const int B, const int M, const int C, const int H, const int W, const int K);

    float* d_Y[NumStream];
    float* d_X[NumStream];
    //float* d_X0, * d_Y0;
    //float* d_X1, * d_Y1;
    //float* d_X2, * d_Y2;
    //float* d_X3, * d_Y3;
    const float* host_x_str;
    float* host_y_str;
};

#endif
