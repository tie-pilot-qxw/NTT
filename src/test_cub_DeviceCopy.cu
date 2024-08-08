#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h> // Add this line
#include <thrust/iterator/counting_iterator.h>  // Add this line
#include <thrust/iterator/constant_iterator.h>  // Add this line

struct GetIteratorToRange
{
  __host__ __device__ __forceinline__ auto operator()(uint32_t index)
  {
    return thrust::make_constant_iterator(d_data_in[index]);
  }
  int32_t *d_data_in;
};

struct GetPtrToRange
{
  __host__ __device__ __forceinline__ auto operator()(uint32_t index)
  {
    return d_data_out + d_offsets[index];
  }
  int32_t *d_data_out;
  uint32_t *d_offsets;
};

struct GetRunLength
{
  __host__ __device__ __forceinline__ uint32_t operator()(uint32_t index)
  {
    return d_offsets[index + 1] - d_offsets[index];
  }
  uint32_t *d_offsets;
};

int main() {
  uint32_t num_ranges = 1;
  int32_t *d_data_in;           // e.g., [4, 2, 7, 3, 1]
  int32_t *d_data_out;          // e.g., [0,                ...               ]
  uint32_t *d_offsets;          // e.g., [0, 2, 5, 6, 9, 14]

  int a[] = {0, 2};
  int b[] = {0, 4};
  cudaMalloc(&d_data_in, 1 * sizeof(uint32_t));
  cudaMemcpy(d_data_in, a + 1, 1 * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_data_out, 5 * sizeof(int32_t));
  cudaMemcpy(d_data_out, a, 2 * sizeof(int32_t), cudaMemcpyHostToDevice);

  cudaMalloc(&d_offsets, 2 * sizeof(uint32_t));
  cudaMemcpy(d_offsets, b, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);

  // Returns a constant iterator to the element of the i-th run
  thrust::counting_iterator<uint32_t> iota(0);
  auto iterators_in = thrust::make_transform_iterator(iota, GetIteratorToRange{d_data_in});

  // Returns the run length of the i-th run
  auto sizes = thrust::make_transform_iterator(iota, GetRunLength{d_offsets});

  // Returns pointers to the output range for each run
  auto ptrs_out = thrust::make_transform_iterator(iota, GetPtrToRange{d_data_out + 1, d_offsets});

  // Determine temporary device storage requirements
  void *d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes, iterators_in, ptrs_out, sizes,
  num_ranges);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run batched copy algorithm (used to perform runlength decoding)
  cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes, iterators_in, ptrs_out, sizes,
  num_ranges);

  // d_data_out       <-- [4, 4, 2, 2, 2, 7, 3, 3, 3, 1, 1, 1, 1, 1]
  int c[5];
  cudaMemcpy(c, d_data_out, 5 * sizeof(int32_t), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 5; i++) {
    printf("%d\n", c[i]);
  }
  cudaFree(d_temp_storage);
  cudaFree(d_data_out);
  cudaFree(d_offsets);
  cudaFree(d_data_in);
  return 0;
}
