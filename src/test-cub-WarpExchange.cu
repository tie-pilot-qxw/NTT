#include <cub/cub.cuh>
#include <cuda_runtime.h>

typedef uint u32;

template <uint WORDS, uint io_group>
__global__ void data_exchange_kernel(uint *data) {
    

    int lid = threadIdx.x;
    int lsize = blockDim.x;

    const u32 io_id = lid & (io_group - 1);
    const u32 lid_start = lid - io_id;
    const u32 shared_read_stride = (lsize << 1) + 1;
    const u32 cur_io_group = io_group < lsize ? io_group : lsize;
    const u32 io_per_thread = io_group / cur_io_group;


    constexpr int warp_threads = io_group;
    constexpr int block_threads = 8;
    constexpr int items_per_thread = 8;
    constexpr int warps_per_block = block_threads / warp_threads;
    const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
    // Specialize WarpExchange for a virtual warp of 16 threads owning 4 integer items each
    using WarpExchangeT = cub::WarpExchange<int, items_per_thread, warp_threads>;
    // Allocate shared memory for WarpExchange
    __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];
    // Load a tile of data striped across threads
    int thread_data[items_per_thread];

    u32 counter = 0;
    printf("Thread %d:\n", threadIdx.x);
    printf("cur_io_group: %d\n", cur_io_group);

    // Read data
    for (u32 i = lid_start; i < lid_start + cur_io_group; i++) {
        for (u32 j = 0; j < io_per_thread; j++) {
            u32 io = io_id + j * cur_io_group;
            if (io < WORDS) {
                thread_data[counter++] = data[i * WORDS + io];
            }
            printf("%d\n", thread_data[counter - 1]);
        }
    }    

    if (threadIdx.x == 0) {
        for (u32 i = 0; i < items_per_thread; i++) {
            printf("%d ", thread_data[i]);
        }
        printf("\n");
    }

    // ...
    // Collectively exchange data into a blocked arrangement across threads
    WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);

    if (threadIdx.x == 0) {
        for (u32 i = 0; i < items_per_thread; i++) {
            printf("%d ", thread_data[i]);
        }
        printf("\n");
    }
}

int main() {
    const uint words = 8;
    constexpr uint io_group = 8;
    printf("io_group: %d\n", io_group);
    //1 << ((int)log2(words - 1) + 1);

    uint data[64];
    for (uint i = 0; i < 64; i++) {
        data[i] = i;
    }
    uint *data_d;
    cudaMalloc(&data_d, 64 * sizeof(uint));
    cudaMemcpy(data_d, data, 64 * sizeof(uint), cudaMemcpyHostToDevice);
    
    data_exchange_kernel<words, io_group><<< 1, 8 >>>(data_d);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    return 0;
}