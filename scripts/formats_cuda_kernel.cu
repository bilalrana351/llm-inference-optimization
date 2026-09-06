#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

__inline__ __device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// One block computes one output element. Each group of four dense weights stores
// two fp16 values. Two 4-bit index codes share one metadata byte, so the weight
// representation occupies 9 bytes per 8 dense elements, exactly 0.5625x fp16.
__global__ void sparse_2to4_gemv_kernel(
    const half* __restrict__ input,
    const half* __restrict__ values,
    const unsigned char* __restrict__ metadata,
    half* __restrict__ output,
    int k) {
  const int row = blockIdx.x;
  const int groups = k / 4;
  const int values_per_row = k / 2;
  const int metadata_per_row = k / 8;
  const half* row_values = values + static_cast<long long>(row) * values_per_row;
  const unsigned char* row_metadata =
      metadata + static_cast<long long>(row) * metadata_per_row;

  float sum = 0.0f;
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    const unsigned char packed = row_metadata[group >> 1];
    const unsigned char code = (group & 1) ? (packed >> 4) : (packed & 0x0f);
    const int index0 = code & 0x03;
    const int index1 = (code >> 2) & 0x03;
    const half2 pair = reinterpret_cast<const half2*>(row_values)[group];
    const float2 weight = __half22float2(pair);
    const int input_base = group * 4;
    sum += weight.x * __half2float(input[input_base + index0]);
    sum += weight.y * __half2float(input[input_base + index1]);
  }

  sum = warp_sum(sum);
  __shared__ float warp_sums[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < (blockDim.x / 32) ? warp_sums[lane] : 0.0f;
    block_sum = warp_sum(block_sum);
    if (lane == 0) {
      output[row] = __float2half(block_sum);
    }
  }
}

}  // namespace

torch::Tensor sparse_2to4_gemv_cuda(
    torch::Tensor input,
    torch::Tensor values,
    torch::Tensor metadata) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const int n = values.size(0);
  const int k = input.size(0);
  auto output = torch::empty({n}, input.options());
  constexpr int threads = 256;
  sparse_2to4_gemv_kernel<<<n, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(values.data_ptr<at::Half>()),
      metadata.data_ptr<unsigned char>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
