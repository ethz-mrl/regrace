#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

#include "sampling_gpu.h"

//extern THCState *state;


int gather_points_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor){
    const float *points = points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gather_points_kernel_launcher_fast(b, c, n, npoints, points, idx, out, stream);
    return 1;
}


int gather_points_grad_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *grad_points = grad_points_tensor.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gather_points_grad_kernel_launcher_fast(b, c, n, npoints, grad_out, idx, grad_points, stream);
    return 1;
}


int furthest_point_sampling_wrapper(int b, int n, int m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *points = points_tensor.data_ptr<float>();
    float *temp = temp_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx, stream);
    return 1;
}
