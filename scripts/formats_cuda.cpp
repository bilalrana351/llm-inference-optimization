#include <torch/extension.h>

torch::Tensor sparse_2to4_gemv_cuda(
    torch::Tensor input,
    torch::Tensor values,
    torch::Tensor metadata);

torch::Tensor sparse_2to4_gemv(
    torch::Tensor input,
    torch::Tensor values,
    torch::Tensor metadata) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(metadata.is_cuda(), "metadata must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be fp16");
  TORCH_CHECK(values.scalar_type() == torch::kFloat16, "values must be fp16");
  TORCH_CHECK(metadata.scalar_type() == torch::kUInt8, "metadata must be uint8");
  TORCH_CHECK(input.dim() == 1, "input must have shape [K]");
  TORCH_CHECK(values.dim() == 2, "values must have shape [N, K/2]");
  TORCH_CHECK(metadata.dim() == 2, "metadata must have shape [N, K/8]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(metadata.is_contiguous(), "metadata must be contiguous");
  TORCH_CHECK(input.size(0) % 8 == 0, "K must be divisible by 8");
  TORCH_CHECK(values.size(0) == metadata.size(0), "N must match");
  TORCH_CHECK(values.size(1) * 2 == input.size(0), "values must store K/2 entries");
  TORCH_CHECK(metadata.size(1) * 8 == input.size(0), "metadata must store K/8 bytes");
  return sparse_2to4_gemv_cuda(input, values, metadata);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("sparse_2to4_gemv", &sparse_2to4_gemv,
             "Handwritten 2:4 sparse fp16 GEMV (CUDA)");
}
