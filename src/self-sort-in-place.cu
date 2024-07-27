#include <iostream>
#include <random>
#include <cassert>
#include <cuda_runtime.h>
#include <ctime>
#include <cstring>

#define P (469762049 ) // 29 * 2^57 + 1ll
#define root (3)

inline long long qpow(long long x, long long y) {
    long long base = 1ll;
    while(y) {
        if (y & 1ll) base = (base * x) % P;
        x = (x * x) % P;
        y >>= 1ll;
    }
    return base;
}

inline long long inv(long long x) {
    return qpow(x, P - 2);
}

void swap(long long &a, long long &b) {
    long long tmp = a;
    a = b;
    b = tmp;
}


void NTT(long long data[], long long reverse[], long long len, long long omega) {
    // rearrange the coefficients
    for (long long i = 0; i < len; i++) {
        if (i < reverse[i]) swap(data[i], data[reverse[i]]);
    }
    
    
    for (long long stride = 1ll; stride < len; stride <<= 1ll) {
        long long gap = qpow(omega, (P - 1ll) / (stride << 1ll));
        for (long long start = 0; start < len; start += (stride << 1ll)) {
            for (long long offset = 0, w = 1ll; offset < stride; offset++, w = (gap * w) % P) {
                long long a = data[start + offset], b = w * data[start + offset + stride] % P;
                data[start + offset] = (a + b) % P;
                data[start + offset + stride] = (a - b + P) % P;
                // printf("%lld %lld\n", w, offset);
            }
        }
    }
    
}

void NTT_dif(long long data[], long long reverse[], uint log_len, long long omega) {
    uint len = 1 << log_len;
    
    for (uint i = log_len; i > 0; i--) {
        uint stride = 1 << (i - 1);
        long long gap = qpow(omega, (P - 1ll) / (stride << 1ll));
        for (uint start = 0; start < len; start += (stride << 1ll)) {
            for (uint offset = 0, w = 1ll; offset < stride; offset++, w = (gap * w) % P) {
                long long a = data[start + offset], b = data[start + offset + stride];
                data[start + offset] = (a + b) % P;
                data[start + offset + stride] = ((a - b + P) % P) * w % P;
            }
        }
        // for (int i = 0; i < len; i++) {
        //     printf("%lld ", data[i]);
        // }
        // printf("\n");
    }

    // rearrange the coefficients
    for (long long i = 0; i < len; i++) {
        if (i < reverse[i]) swap(data[i], data[reverse[i]]);
    }

}

void NTT_pro1(long long data[], uint log_len, long long omega) {
    uint len = 1 << log_len;
    
    for (uint i = log_len; i > log_len / 2; i--) {
        uint stride = 1 << (i - 1);
        long long gap = qpow(omega, (P - 1ll) / (stride << 1ll));
        for (uint start = 0; start < len; start += (stride << 1ll)) {
            for (uint offset = 0, w = 1ll; offset < stride; offset++, w = (gap * w) % P) {
                long long a = data[start + offset], b = data[start + offset + stride];
                data[start + offset] = (a + b) % P;
                data[start + offset + stride] = ((a - b + P) % P) * w % P;
            }
        }
    }

}

void NTT_pro2(long long data[], uint log_len, long long omega) {
    uint len = 1 << log_len;

    for (uint i = log_len / 2; i >= 1; i--) {
        uint stride = 1 << (i - 1);
        uint pair_stride = 1 << (log_len - i);
        printf("%u %u\n", stride, pair_stride);
        long long gap = qpow(omega, (P - 1ll) / (stride << 1));
        for (uint start = 0; start < len; start += (pair_stride << 1)) {
            for (uint offset0 = 0; offset0 < pair_stride; offset0 += (stride << 1)) {
                for (uint offset = 0, w = 1ll; offset < stride; offset++, w = (gap * w) % P) {
                    long long a = data[start + offset0 + offset];
                    long long b = data[start + offset0 + offset + stride];
                    long long c = data[start + offset0 + offset + pair_stride];
                    long long d = data[start + offset0 + offset + pair_stride + stride];
                    data[start + offset0 + offset] = (a + b) % P;
                    data[start + offset0 + offset + stride] = (c + d) % P;
                    data[start + offset0 + offset + pair_stride] = ((a - b + P) % P) * w % P;
                    data[start + offset0 + offset + pair_stride + stride] = ((c - d + P) % P) * w % P;
                }
            }
        }
    }

}

__forceinline__ __device__ long long FIELD_pow_lookup(long long *omegas, uint exponent) {
    long long res = 1ll;
    uint i = 0;
    while(exponent > 0) {
        if (exponent & 1)
            res = (res * omegas[i]) % P;
        exponent = exponent >> 1;
        i++;
    }
    return res;
}

__forceinline__ __device__ long long FIELD_pow (long long base, uint exponent) {
    long long res = 1;
    while(exponent > 0) {
        if (exponent & 1)
        res = (res * base) % P;
        exponent = exponent >> 1;
        base = (base * base) % P;
    }
    return res;
}

__global__ void SSIP_NTT_stage1 (long long * x, // Source buffer
                        long long * pq, // Precalculated twiddle factors
                        long long * omegas, // [omega, omega^2, omega^4, ...]
                        uint n, // Number of elements
                        uint log_stride, // Log2 of `p` (Read more in the link above)
                        uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                        uint max_deg)
{
    extern __shared__ long long u[];

    uint lid = threadIdx.x;
    uint index = blockIdx.x;
    const uint end_stride = 1 << log_stride >> (deg - 1); //stride of the last butterfly
    // printf("%u %u", log_stride, end_stride);

    // each segment is independent
    uint segment_stride = end_stride << deg; // the distance between two segment
    uint segment_num = end_stride; // # of blocks in a segment
    
    uint segment_start = index / segment_num * segment_stride;
    uint segment_id = index & (segment_num - 1);
    
    // uint block_offset = segment_id; // TODO

    uint subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round
    uint subblock_id = segment_id & (end_stride - 1);

    x += segment_start + subblock_id;

    uint group_id = lid & (subblock_sz - 1);

    uint gpos = group_id * (end_stride << 1);

    //printf("%u %u %u %u %u %u\n", blockIdx.x, threadIdx.x, gpos, group_id, segment_start, subblock_id);
    

    u[(lid << 1)] = x[gpos];
    u[(lid << 1) + 1] = x[gpos + end_stride];

     __syncthreads();
    // if (threadIdx.x == 0 && blockIdx.x == 3) {
    //         for (int i = 0; i < blockDim.x << 1; i++) {
    //             printf("%lld ", u[i]);
    //         }
    //         printf("\n");
    // }
    const uint pqshift = max_deg - deg;
    for(uint rnd = 0; rnd < deg; rnd++) {
        const uint bit = subblock_sz >> rnd;
        uint i = lid;
        const uint di = i & (bit - 1);
        const uint i0 = (i << 1) - di;
        const uint i1 = i0 + bit;
        long long tmp = u[i0];
        u[i0] = (u[i0] + u[i1]) % P;
        u[i1] = (tmp + P - u[i1]) % P;
        if(di != 0) u[i1] = (pq[di << rnd << pqshift] * u[i1]) % P;

        __syncthreads();

        // if (threadIdx.x == 0 && blockIdx.x == 1) {
        //     for (int i = 0; i < blockDim.x << 1; i++) {
        //         printf("%lld ", u[i]);
        //     }
        //     printf("\n");
        // }
    }


    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     for (int i = 0; i < blockDim.x * 4; i++) {
    //         printf("%lld ", s[i]);
    //     }
    // }
    uint k = index & (end_stride - 1);
    long long twiddle = FIELD_pow_lookup(omegas, (n >> (log_stride - deg + 1) >> deg) * k);

    //printf("%u %u %u %u %u %u\n", blockIdx.x, threadIdx.x, (n >> log_stride >> (deg - 1)) * k,lid << 1, __brev(lid << 1) >> (32 - deg),__brev((lid << 1) + 1) >> (32 - deg));

    long long t1 = FIELD_pow(twiddle, __brev(lid << 1) >> (32 - deg));
    long long t2 = FIELD_pow(twiddle, __brev((lid << 1) + 1) >> (32 - deg));
    x[gpos] = t1 * u[(lid << 1)] % P;
    x[gpos + end_stride] = t2 * u[(lid << 1) + 1] % P;
}

// __global__ void SSIP_NTT_stage2 (long long data[], uint log_len, uint log_stride, uint deg) {
//     extern __shared__ long long s[];

//     uint lid = threadIdx.x;
//     uint index = blockIdx.x;
//     uint start_stride = 1 << log_stride; // stride of the first butterfly
//     uint end_stride = start_stride >> (deg - 1); //stride of the last butterfly
//     uint start_pair_stride = 1 << (log_len - log_stride - 1); // the stride between the first pair of butterfly
//     uint end_pair_stride = start_pair_stride << (deg - 1); // the stride between the last pair of butterfly

//     // each segment is independent
//     uint segment_stride = end_pair_stride << 1; // the distance between two segment
//     uint segment_num = segment_stride >> (deg << 1); // # of blocks in a segment
    
//     uint segment_start = index / segment_num * segment_stride;
//     uint segment_id = index & (segment_num - 1);
    
//     // uint block_offset = segment_id; // TODO

//     uint subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round
//     uint subblock_offset = (segment_id / (end_stride)) * (2 *subblock_sz * end_stride);
//     uint subblock_id = segment_id & (end_stride - 1);

//     data += segment_start + subblock_offset + subblock_id;

//     uint group_offset = (lid / subblock_sz) * (start_pair_stride);

//     uint group_id = lid & (subblock_sz - 1);

//     uint gpos = group_offset + group_id * (end_stride << 1);

//     printf("%u %u %u %u %u %u %u %u\n", blockIdx.x, threadIdx.x, gpos, group_offset, group_id, segment_start, subblock_offset, subblock_id);

//     s[(lid << 1)] = data[gpos];
//     s[(lid << 1) + 1] = data[gpos + end_stride];
//     s[(lid << 1) + (blockDim.x << 1)] = data[gpos + end_pair_stride];
//     s[(lid << 1) + (blockDim.x << 1) + 1] = data[gpos + end_pair_stride + end_stride];

//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         for (int i = 0; i < blockDim.x * 4; i++) {
//             printf("%lld ", s[i]);
//         }
//     }
// }

void run_kernel(long long * data, uint log_len) {
    uint deg = 3;
    uint log_stride = 4;
    uint n = 1 << log_len;

    dim3 block(1 << deg >> 1);
    dim3 grid(n / 2 / block.x);
    printf("%d %d\n", block.x, grid.x);

    SSIP_NTT_stage1 <<< grid, block, sizeof(long long) * block.x * 2>>>(data, data, data, n, log_stride, deg, 0);


    dim3 block1(1 << (deg << 1) >> 2);
    dim3 grid1(n / 4 / block1.x);
    printf("%d %d\n", block1.x, grid1.x);

    // SSIP_NTT_stage2<<< grid1, block1, sizeof(long long) * block1.x * 4 >>>(data, log_len, log_stride, deg);

    cudaDeviceSynchronize();
}

#define MAX_LOG2_RADIX 11u
void bellperson_baseline(long long *x,long long omega, uint log_n) {

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);



    uint n = 1 << log_n;

    omega = qpow(omega, (P - 1ll) / n);
    // for (uint i = 0; i <= n; i++) printf("%lld ", qpow(omega, i));
    // return;
    // All usages are safe as the buffers are initialized from either the host or the GPU
    // before they are read.
    // let mut src_buffer = unsafe { program.create_buffer::<F>(n)? };
    // let mut dst_buffer = unsafe { program.create_buffer::<F>(n)? };
    // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
    int max_deg = std::min(MAX_LOG2_RADIX, log_n);

    // Precalculate:
    // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
    long long *pq, *pq_d;
    long long *omegas, *omegas_d;
    pq = new long long[1 << max_deg >> 1];
    memset (pq, 0, sizeof(long long) * (1 << max_deg >> 1));
    pq[0] = 1;
    long long twiddle = qpow(omega, ((long long)n) >> (1ll*max_deg));
    if (max_deg > 1) {
        pq[1] = twiddle;
        for (uint i = 2; i < (1 << max_deg >> 1) ; i++ ) {
            pq[i] = pq[i - 1];
            pq[i] = pq[i] *(twiddle)%P;
        }
    }
    cudaMalloc(&pq_d, sizeof(long long) * (1 << max_deg >> 1));
    cudaMemcpy(pq_d, pq, sizeof(long long) * (1 << max_deg >> 1), cudaMemcpyHostToDevice);

    // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
    omegas = new long long[32];
    memset (omegas, 0, sizeof(long long) * 32);
    omegas[0] = omega;
    for (uint i  = 1; i < 32; i++) {
        omegas[i] = omegas[i - 1] * omegas[i - 1] % P;
    }
    cudaMalloc(&omegas_d, sizeof(long long) * 32);
    cudaMemcpy(omegas_d, omegas, sizeof(long long) * 32, cudaMemcpyHostToDevice);
    long long *res = new long long[n];

    // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
    int log_p = log_n - 1;
    
    cudaEventRecord(start);

    // Each iteration performs a FFT round
    while (log_p >= 0) {

        // 1=>radix2, 2=>radix4, 3=>radix8, ...
        uint deg = std::min(max_deg, log_p + 1);

        uint n = 1u << log_n;
        dim3 block(1 << std::min(deg - 1, 10u) );
        dim3 grid(n >> deg);

        // printf("%d %d %d\n", block.x, grid.x, deg);

        SSIP_NTT_stage1 <<< grid, block, sizeof(long long) * (1 << deg) >>>(x, pq_d, omegas_d, n, log_p, deg, max_deg);

        log_p -= deg;
        // cudaMemcpy(res, x, sizeof(*res) * n, cudaMemcpyDeviceToHost);
        // for (int i = 0; i < n; i++) printf("%lld ", res[i]);
        // printf("\n");
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float t;
    cudaEventElapsedTime(&t, start, end);
    delete [] res;

    printf("bellman: %fms\n", t);
    free(pq);
    free(omegas);
    cudaFree(pq_d);
    cudaFree(omegas_d);
}

int main() {
    long long *data, *reverse, *data_copy;
    long long l,length = 1ll;
    int bits = 0;

    //scanf("%lld", &l);
    l = qpow(2, 24);

    while (length < l) {
        length <<= 1ll;
        bits ++;
    }

    assert(length == (1ll << bits));
    assert(bits <= 57);

    data = new long long[length];
    data_copy = new long long[length];
    reverse = new long long [length];

    for (long long i = 0; i < length; i++) {
        reverse[i] = (reverse[i >> 1ll] >> 1ll) | ((i & 1ll) << (bits - 1ll) ); //reverse the bits
    }

    std::random_device rd;
    std::mt19937_64 gen(rd());
    for (long long i = 0; i < length; i++) {
        data[i] = i; std::abs((long long)gen()) % P;
        data_copy[i] = data[i];
    }

    // cpu implementation
    {
        clock_t start = clock();

        NTT(data, reverse, length, root);

        clock_t end = clock();
        printf("cpu: %lfms\n",(double)(end - start) / CLOCKS_PER_SEC * 1000);
    }

    long long *data_new;
    data_new = new long long[length];
    memcpy(data_new, data_copy, length * sizeof(long long));

    // Cpu SSIP
    {
        clock_t start = clock();

        // NTT_pro1(data_new, bits, root);
        // NTT_pro2(data_new, bits, root);
        NTT_dif(data_new, reverse, bits, root);

        clock_t end = clock();
        printf("cpu: %lfms\n",(double)(end - start) / CLOCKS_PER_SEC * 1000);

        for (long long i = 0; i < length; i++) {
            if (data_new[i] != data[i]) {
                printf("error\n");
                return 0;
            }
        }
        printf("correct\n");
    }

    long long * data_d;
    cudaMalloc(&data_d, sizeof(long long) * length);
    cudaMemcpy(data_d, data_copy, sizeof(long long) * length, cudaMemcpyHostToDevice);

    bellperson_baseline(data_d, root, bits);

    cudaMemcpy(data_new, data_d, length * sizeof(long long), cudaMemcpyDeviceToHost);
    // rearrange the coefficients
    for (long long i = 0; i < length; i++) {
        if (i < reverse[i]) swap(data_new[i], data_new[reverse[i]]);
    }

    for (long long i = 0; i < length; i++) {
        if (data_new[i] != data[i]) {
            printf("error\n");
            return 0;
        }
    }
    printf("correct\n");
    // run_kernel(data_d, bits);
    return 0;
}