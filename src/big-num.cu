#include <iostream>
#include <random>
#include <cassert>
#include <cuda_runtime.h>
#include <ctime>
#include <gmp.h>
#include "cgbn/cgbn.h"

#define P (469762049      ) // 4179340454199820289 29 * 2^57 + 1ll
#define root (3)
#define TPI 4
#define BITS 128

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

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

void NTT(long long data[], uint reverse[], long long len, long long omega) {

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

template <const uint tpi, const uint bits>
__global__ void rearrange(cgbn_mem_t<bits> * data, uint2 * reverse, uint len) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    uint2 r = reverse[index];
    cgbn_mem_t<bits> tmp = data[r.x];
    data[r.x] = data[r.y];
    data[r.y] = tmp;
}

template <const uint tpi, const uint bits>
__global__ void naive(cgbn_mem_t<bits> data[], long long len, cgbn_mem_t<bits> roots[], uint stride, cgbn_mem_t<bits> prime) {
    typedef cgbn_context_t<tpi> context_t;
    typedef cgbn_env_t<context_t, bits> env_t;

    context_t bn_context;
    env_t bn_env(bn_context);

    uint id = (blockDim.x * blockIdx.x + threadIdx.x) / tpi;
    if (id << 1 >= len) return;
    uint offset = id % stride;
    uint pos = ((id - offset) << 1ll) + offset;

    typename env_t::cgbn_t  a, b, w, ra, rb, mod;
    typename env_t::cgbn_wide_t wtmp;

    cgbn_load(bn_env, mod, &prime);
    cgbn_load(bn_env, a, &data[pos]);
    cgbn_load(bn_env, b, &data[pos + stride]);
    cgbn_load(bn_env, w, &roots[offset * len / (stride << 1ll)]);

    // b = w * data[pos + stride] % P
    cgbn_mul_wide(bn_env, wtmp, w, b);
    cgbn_rem_wide(bn_env, b, wtmp, mod);

    // data[pos] = (a + b) % P
    cgbn_add(bn_env, ra, a, b);
    cgbn_rem(bn_env, ra, ra, mod);
    cgbn_store(bn_env, &data[pos], ra);

    // data[pos + stride] = (a - b + P) % P
    cgbn_add(bn_env, rb, a, mod);
    cgbn_sub(bn_env, rb, rb, b);
    cgbn_rem(bn_env, rb, rb, mod);
    cgbn_store(bn_env, &data[pos + stride], rb);
}

template <const uint tpi, const uint bits>
void NTT_GPU_Naive(cgbn_mem_t<bits> data[], uint len, uint2 reverse[], uint reverse_len, cgbn_mem_t<bits> prime, cgbn_mem_t<bits> omega) {
    assert(bits / 32 <= tpi);
    typedef cgbn_context_t<tpi> context_t;
    typedef cgbn_env_t<context_t, bits> env_t;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    cgbn_mem_t<bits> *roots, *roots_d;
    roots = (cgbn_mem_t<bits> *)malloc(sizeof(*roots) * len);
    memset(roots, 0, sizeof(*roots) * len);

    context_t bn_context;
    env_t bn_env(bn_context);

    typename env_t::cgbn_t gap, mod, exponent;
    typename env_t::cgbn_wide_t wtmp;

    // long long gap = qpow(omega, (P - 1) / len); 
    
    cgbn_load(bn_env, mod, &prime);
    cgbn_load(bn_env, gap, &omega);
    cgbn_sub_ui32(bn_env, exponent, mod, 1);
    cgbn_div_ui32(bn_env, exponent, exponent, len);
    cgbn_modular_power(bn_env, gap, gap, exponent, mod);

    // roots[i] = roots[i - 1] * gap % P;
    roots[0]._limbs[0] = 1;
    typename env_t::cgbn_t tmp;
    cgbn_load(bn_env, tmp, &roots[0]);

    for (uint i = 1; i < len; i++) {
        cgbn_mul_wide(bn_env, wtmp, gap, tmp);
        cgbn_rem_wide(bn_env, tmp, wtmp, mod);
        cgbn_store(bn_env, &roots[i], tmp);
    }
        
    cudaMalloc(&roots_d, len * sizeof(*roots_d));
    cudaMemcpy(roots_d, roots, len * sizeof(*roots_d), cudaMemcpyHostToDevice);

    dim3 block(96);
    dim3 grid((reverse_len - 1) / block.x + 1);
    dim3 grid1((len / 2 - 1) / block.x + 1);
    block.x *= tpi;
    rearrange<tpi, bits> <<< grid, block >>>(data, reverse, reverse_len);
    for (long long stride = 1ll; stride < len; stride <<= 1ll) {
        naive<tpi, bits> <<< grid1, block >>>(data, len, roots_d, stride, prime);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float t;
    cudaEventElapsedTime(&t, start, end);

    printf("naive: %fms\n", t);

    cudaFree(roots_d);
    free(roots);
}

template <const uint tpi, const uint bits>
__global__ void GZKP (cgbn_mem_t<bits> data[],  uint len, cgbn_mem_t<bits> roots[], uint stride, uint G, uint csize, uint B, cgbn_mem_t<bits> prime) {
    typedef cgbn_context_t<tpi> context_t;
    typedef cgbn_env_t<context_t, bits> env_t;

    extern __shared__ typename env_t::cgbn s[];

    context_t bn_context;
    env_t bn_env(bn_context);

    uint start = (G * blockIdx.x) % stride + (G * blockIdx.x / stride) * stride * csize * 2;

    uint id = threadIdx.x / tpi;;

    uint iogroup = id / G; // io group
    uint ioid = id % G; // id in the io group

    uint spos; // load to
    uint gpos; // load from

    // (stride >= G) 
    spos = ioid * csize * 2 + 2 * iogroup;
    gpos = start + ioid + iogroup * 2 * stride;

    // load to shared memory
    if (gpos + stride >= len) return;

    cgbn_load(bn_env, s[spos], &data[gpos]);
    cgbn_load(bn_env, s[spos + 1], &data[gpos + stride]);
    
    uint cid = id % csize; // id of the compute group
    uint base =  (id - cid) << 1; // base pos of the compute group

    typename env_t::cgbn_t w, a, b, mod;
    typename env_t::cgbn_wide_t wtmp;
    cgbn_load(bn_env, mod, &prime);

    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i  = 0; i < blockDim.x / tpi * 2; i++) {
            printf("%d ", s[i]._limbs[0]);
        }
        printf("\n");
    }
    
    for (uint i = 1, step = 1; i <= B && stride * step < len; i++, step *= 2) {
        __syncthreads();

        uint offset = step * 2 * ((uint)(cid / step)) + cid % step;

        uint io_thread_id = id / csize + offset / 2 * G;

        iogroup = (io_thread_id) / G;
        ioid = (io_thread_id) - iogroup * G;

        uint tmp = start + ioid + iogroup * 2 * stride + (offset & 1) * stride;
        tmp = tmp % (stride * step * 2);
        tmp = tmp * (len / (stride * step * 2));

        cgbn_load(bn_env, w, &roots[tmp]);

        // a = s[base + offset]
        cgbn_set(bn_env, a, s[base + offset]);
        // b = w * s[base + offset + step] % P
        cgbn_mul_wide(bn_env, wtmp, w, s[base + offset + step]);
        cgbn_rem_wide(bn_env, b, wtmp, mod);

        // s[base + offset] = (a + b) % P;
        cgbn_add(bn_env, s[base + offset], a, b);
        cgbn_rem(bn_env, s[base + offset], s[base + offset], mod);

        // s[base + offset + step] = (a - b + P) % P;
        cgbn_add(bn_env, s[base + offset + step], a, mod);
        cgbn_sub(bn_env, s[base + offset + step], s[base + offset + step], b);
        cgbn_rem(bn_env, s[base + offset + step], s[base + offset + step], mod);
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i  = 0; i < blockDim.x / tpi * 2; i++) {
            printf("%d ", s[i]._limbs[0]);
        }
    }

    cgbn_store(bn_env, &data[gpos], s[spos]);
    cgbn_store(bn_env, &data[gpos + stride], s[spos + 1]);

}

template <const uint tpi, const uint bits>
void NTT_GZKP(cgbn_mem_t<bits> data[], uint len, uint2 reverse[], uint reverse_len, cgbn_mem_t<bits> prime, cgbn_mem_t<bits> omega, uint B, uint G) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 block(qpow(2,B)*G/2);
    assert(block.x * tpi <= 1024);
    assert(bits / 32 <= tpi);


    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    //assert(deviceProp.sharedMemPerBlock >= sizeof(long long) * qpow(2,B) * G * 2 + sizeof(long long) * len);

    cudaEventRecord(start);

    cgbn_mem_t<bits> *roots, *roots_d;
    roots = (cgbn_mem_t<bits> *)malloc(sizeof(*roots) * len);
    memset(roots, 0, sizeof(*roots) * len);

    typedef cgbn_context_t<tpi> context_t;
    typedef cgbn_env_t<context_t, bits> env_t;

    context_t bn_context;
    env_t bn_env(bn_context);

    typename env_t::cgbn_t gap, mod, exponent;
    typename env_t::cgbn_wide_t wtmp;

    // long long gap = qpow(omega, (P - 1) / len);
    cgbn_load(bn_env, mod, &prime);
    cgbn_load(bn_env, gap, &omega);
    cgbn_sub_ui32(bn_env, exponent, mod, 1);
    cgbn_div_ui32(bn_env, exponent, exponent, len);
    cgbn_modular_power(bn_env, gap, gap, exponent, mod);

    // roots[i] = roots[i - 1] * gap % P;
    roots[0]._limbs[0] = 1;
    typename env_t::cgbn_t tmp;
    cgbn_load(bn_env, tmp, &roots[0]);
    for (uint i = 1; i < len; i++) {
        cgbn_mul_wide(bn_env, wtmp, gap, tmp);
        cgbn_rem_wide(bn_env, tmp, wtmp, mod);
        cgbn_store(bn_env, &roots[i], tmp);
    }
    
    cudaMalloc(&roots_d, len * sizeof(*roots_d));
    cudaMemcpy(roots_d, roots, len * sizeof(*roots_d), cudaMemcpyHostToDevice);

    dim3 block0(768);
    dim3 grid0((reverse_len - 1) / block0.x + 1);
    dim3 grid1((len / 2 - 1) / block.x + 1);
    block.x *= tpi;

    // rearrange<tpi, bits> <<< grid0, block0 >>>(data, reverse, reverse_len);

    uint stride = 1;
    for (; stride < G; stride <<= 1) {
        // naive<tpi, bits> <<< grid1, block >>>(data, len, roots_d, stride, prime);
    }

    for (; stride << B <= len; stride <<= B) {
        GZKP<tpi, bits> <<< grid1, block, sizeof(cgbn_mem_t<bits>) * qpow(2,B) * G>>>(data, len, roots_d, stride, G, qpow(2,B)/2, B, prime);
    }

    B = 0;
    uint stride_c = stride;
    while (stride_c < len) {
        stride_c <<= 1;
        B++;
    }
    if (B != 0) {
        block = dim3(qpow(2,B)*G/2);
        grid1 = dim3((len / 2 - 1) / block.x + 1);
        block.x *= tpi;
        GZKP<tpi, bits> <<< grid1, block, sizeof(cgbn_mem_t<bits>) * qpow(2,B) * G>>>(data, len, roots_d, stride, G, qpow(2,B)/2, B, prime);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float t;
    cudaEventElapsedTime(&t, start, end);

    printf("GZKP: %fms\n",t);

    cudaFree(roots_d);
    free(roots);
}

template <const uint bits>
cgbn_mem_t<bits> int64_to_cgbn(const long long &x) {
    cgbn_mem_t<bits> r;
    r._limbs[0] = x & 0xffffffff;
    r._limbs[1] = x >> 32;
    for (int i = 2; i < bits / 32; i++) r._limbs[i] = 0;
    return r;
}

template <const uint bits>
long long cgbn_to_int64(const cgbn_mem_t<bits> &x) {
    return ((long long)x._limbs[0]) | ((long long)x._limbs[1] << 32ll);
}

int main() {
    long long *data;
    uint *reverse;
    cgbn_mem_t<BITS> *data_copy;
    uint l,length = 1ll;
    int logl = 0;

    //scanf("%lld", &l);
    l = 16;qpow(2, 24);

    while (length < l) {
        length <<= 1ll;
        logl ++;
    }

    assert(length == (1ll << logl));
    assert(logl <= 26);

    data = new long long[length];
    data_copy = (cgbn_mem_t<BITS> *)malloc(sizeof(*data_copy) * length);
    reverse = new uint[length];

    for (uint i = 0; i < length; i++) {
        reverse[i] = (reverse[i >> 1ll] >> 1ll) | ((i & 1ll) << (logl - 1ll) ); //reverse the bits
    }

    std::random_device rd;
    std::mt19937_64 gen(rd());
    for (long long i = 0; i < length; i++) {
        data[i] = i;//std::abs((long long)gen()) % P;
        data_copy[i] = int64_to_cgbn<BITS>(data[i]);
    }

    // cpu implementation
    {
        clock_t start = clock();

        NTT(data, reverse, length, root);

        clock_t end = clock();
        printf("cpu: %lfms\n",(double)(end - start) / CLOCKS_PER_SEC * 1000);
    }

    // revise the reverse array to elimate branch divergence
    uint2 *reverse2 = (uint2 *)malloc(length / 2 * sizeof(uint2));
    uint reverse_num =  0;
    
    for (uint i = 0; i < length; i++) {
        if (reverse[i] < i) {
            reverse2[reverse_num] = make_uint2(reverse[i], i);
            reverse_num++;
        }
    }
    uint2 *reverse2_d;

    cudaMalloc(&reverse2_d, length / 2 * sizeof(*reverse2_d));
    cudaMemcpy(reverse2_d, reverse2, length / 2 * sizeof(*reverse2), cudaMemcpyHostToDevice);

    cgbn_mem_t<BITS> *data_d;

    cudaMalloc(&data_d, length * sizeof(*data_d));
    cudaMemcpy(data_d, data_copy, length * sizeof(*data_d), cudaMemcpyHostToDevice);

    cgbn_mem_t<BITS> prime, omega;
    prime = int64_to_cgbn<BITS>(P);
    omega = int64_to_cgbn<BITS>(root);

    // naive gpu approach
    NTT_GPU_Naive<TPI, BITS>(data_d, length, reverse2_d, reverse_num, prime, omega);

    cgbn_mem_t<BITS> *tmp;
    tmp = (cgbn_mem_t<BITS> *)malloc(sizeof(*tmp) * length);

    cudaMemcpy(tmp, data_d, sizeof(*data_d) * length, cudaMemcpyDeviceToHost);

    for (long long i = 0; i < length; i++) {
        long long cur = cgbn_to_int64(tmp[i]);
        if (data[i] != cur) {
            printf("%lld %lld %lld\n", data[i], cur, i);
        }
    }


    // GZKP approach

    // cudaMemcpy(data_d, data_copy, length * sizeof(*data_d), cudaMemcpyHostToDevice);

    // NTT_GZKP<TPI, BITS>(data_d, length, reverse2_d, reverse_num, prime, omega, 2, 4);

    // cudaMemcpy(tmp, data_d, sizeof(*data_d) * length, cudaMemcpyDeviceToHost);

    // for (long long i = 0; i < length; i++) {
    //     long long cur = cgbn_to_int64(tmp[i]);
    //     if (data[i] != cur) {
    //         printf("%lld %lld %lld\n", data[i], cur, i);
    //     }
    // }

    
    // NTT(data, reverse, length, inv(root));

    // long long co = inv(length);
    // for (long long i = 0; i < length; i++) {
    //     if (data[i] * co % P != data_copy[i]) {
    //         std::cout << data[i] * co % P << " " << data_copy[i] << " " << i << std::endl;
    //     }
    // }
    cudaFree(data_d);
    cudaFree(reverse2_d);

    delete [] data;
    free(data_copy);
    delete [] reverse;
    free(tmp);
    return 0;
}