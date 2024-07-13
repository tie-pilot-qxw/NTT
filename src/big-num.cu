#include <iostream>
#include <random>
#include <cassert>
#include <cuda_runtime.h>
#include <ctime>
#include "cgbn/cgbn.h"

#define P (4179340454199820289) // 29 * 2^57 + 1ll
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

template <uint tpi, uint bits>
__global__ void rearrange(cgbn_mem_t<bits> * data, long long * reverse, long long len) {
    typedef cgbn_context_t<tpi> context_t;
    typedef cgbn_env_t<context_t, bits> env_t;

    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    long long rid = reverse[index];
    if (index < rid) {
        context_t bn_context();
        env_t bn_env(bn_context);
        long long tmp = data[index];
        data[index] = data[rid];
        data[rid] = tmp;
    }
}

__global__ void naive(long long data[], long long len, long long roots[], long long stride) {

    long long id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id << 1 >= len) return;
    long long offset = id % stride;
    long long pos = ((id - offset) << 1ll) + offset;

    long long w = roots[offset * len / (stride << 1ll)];

    long long a = data[pos], b = w * data[pos + stride] % P;
    data[pos] = (a + b) % P;
    data[pos + stride] = (a - b + P) % P;
}

void NTT_GPU_Naive(long long data[], long long reverse[], long long len, long long omega) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    long long *roots, *roots_d;
    roots = new long long [len];
    long long gap = qpow(omega, (P - 1ll) / len);
    roots[0] = 1;
    for (long long i = 1; i < len; i++) roots[i] = roots[i - 1] * gap % P;
    
    cudaMalloc(&roots_d, len * sizeof(*roots_d));
    cudaMemcpy(roots_d, roots, len * sizeof(*roots_d), cudaMemcpyHostToDevice);

    long long *tmp;
    tmp = new long long [len];

    dim3 block(768);
    dim3 grid((len - 1) / block.x + 1);
    dim3 grid1((len / 2 - 1) / block.x + 1);
    rearrange <<< grid, block >>>(data, reverse, len);
    for (long long stride = 1ll; stride < len; stride <<= 1ll) {
        naive <<< grid1, block >>>(data, len, roots_d, stride);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float t;
    cudaEventElapsedTime(&t, start, end);

    printf("naive: %fms\n", t);

    cudaFree(roots_d);
    delete [] roots;
    delete [] tmp;
}

__global__ void GZKP (long long data[],  long long len, long long roots[], long long stride, int G, int csize, int B) {
    extern __shared__ long long s[];
    long long *global = s + csize * 2 * G;

    long long start = (G * blockIdx.x) % stride + ((long long) G * blockIdx.x / stride) * stride * csize * 2;

    int iogroup = threadIdx.x / G; // io group
    int ioid = threadIdx.x % G; // id in the io group

    int spos; // load to
    long long gpos; // load from
    long long gpos_c = start + threadIdx.x % stride + ((long long) threadIdx.x / stride) * stride * 2; // the global position of the computing element

    if (stride < G) {
        spos = threadIdx.x * 2;
        gpos = gpos_c;
    } else {
        spos = ioid * csize * 2 + 2 * iogroup;
        gpos = start + ioid + iogroup * 2 * stride;
    }

    // load to shared memory
    if (gpos + stride >= len) return;

    // printf("%d %lld %lld\n",spos, gpos, start);

    s[spos] = data[gpos];
    global[spos] = gpos;
    s[spos + 1] = data[gpos + stride];
    global[spos + 1] = gpos + stride;

    // if (threadIdx.x == 0 && blockIdx.x == 0){
    //     for (int i = 0; i < blockDim.x*2; i++) {
    //         printf("%lld ", s[i]);
    //     }
    //     printf("\n");
    // }


    int cid = threadIdx.x % csize; // id of the compute group
    int base =  (threadIdx.x - cid) << 1; // base pos of the compute group
    
    for (int i = 1, step = 1; i <= B && stride * step < len; i++, step *= 2) {
        __syncthreads();

        int offset = step * 2 * ((int)(cid / step)) + cid % step;

        gpos_c = global[base + offset];

        long long tmp = gpos_c % (stride * step * 2) * len / (stride * step * 2);

        long long w = roots[tmp];

        long long a = s[base + offset], b = w * s[base + offset + step] % P;
        s[base + offset] = (a + b) % P;
        s[base + offset + step] = (a - b + P) % P;

        // printf("%d %d %lld %lld %lld %lld\n", base, offset, s[base + offset], s[base + offset + step], tmp, gpos_c);

    }
    
    __syncthreads();
    data[gpos] = s[spos];
    data[gpos + stride] = s[spos + 1];
}

void NTT_GZKP(long long data[], long long reverse[], long long len, long long omega, int B, int G) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 block(qpow(2,B)*G/2);
    assert(block.x <= 1024);

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    //assert(deviceProp.sharedMemPerBlock >= sizeof(long long) * qpow(2,B) * G * 2 + sizeof(long long) * len);

    cudaEventRecord(start);

    long long *roots, *roots_d;
    roots = new long long [len];
    long long gap = qpow(omega, (P - 1ll) / len);
    roots[0] = 1;
    for (long long i = 1; i < len; i++) roots[i] = roots[i - 1] * gap % P;
    
    cudaMalloc(&roots_d, len * sizeof(*roots_d));
    cudaMemcpy(roots_d, roots, len * sizeof(*roots_d), cudaMemcpyHostToDevice);


    long long *tmp;
    tmp = new long long [len];

    dim3 grid0((len - 1) / block.x + 1);
    dim3 grid1((len / 2 - 1) / block.x + 1);

    rearrange <<< grid0, block >>>(data, reverse, len);

    long long stride = 1ll;
    for (; stride < G; stride <<= 1) {
        naive <<< grid1, block >>>(data, len, roots_d, stride);
    }

    for (; stride << B <= len; stride <<= B) {
        GZKP <<< grid1, block, sizeof(long long) * qpow(2,B) * G * 2>>>(data, len, roots_d, stride, G, qpow(2,B)/2, B);
    }

    B = 0;
    long long stride_c = stride;
    while (stride_c < len) {
        stride_c <<= 1;
        B++;
    }
    if (B != 0) {
        block = dim3(qpow(2,B)*G/2);
        grid1 = dim3((len / 2 - 1) / block.x + 1);
        GZKP <<< grid1, block, sizeof(long long) * qpow(2,B) * G * 2>>>(data, len, roots_d, stride, G, qpow(2,B)/2, B);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float t;
    cudaEventElapsedTime(&t, start, end);

    printf("GZKP: %fms\n",t);

    cudaFree(roots_d);
    delete [] roots;

    delete [] tmp;
}

int main() {
    long long *data, *reverse, *data_copy;
    long long l,length = 1ll;
    int bits = 0;

    //scanf("%lld", &l);
    l = 10000000ll;

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
        data[i] = i;//std::abs((long long)gen()) % P;
        data_copy[i] = data[i];
    }

    // cpu implementation
    {
        clock_t start = clock();

        NTT(data, reverse, length, root);

        clock_t end = clock();
        printf("cpu: %lfms\n",(double)(end - start) / CLOCKS_PER_SEC * 1000);
    }

    long long *data_d, *reverse_d;

    cudaMalloc(&data_d, length * sizeof(*data_d));
    cudaMalloc(&reverse_d, length * sizeof(*reverse_d));
    cudaMemcpy(data_d, data_copy, length * sizeof(*data_d), cudaMemcpyHostToDevice);
    cudaMemcpy(reverse_d, reverse, length * sizeof(*reverse_d), cudaMemcpyHostToDevice);

    // naive gpu approach
    NTT_GPU_Naive(data_d, reverse_d, length, root);

    long long *tmp;
    tmp = new long long [length];

    cudaMemcpy(tmp, data_d, sizeof(*data_d) * length, cudaMemcpyDeviceToHost);

    for (long long i = 0; i < length; i++) {
        if (data[i] != tmp[i]) {
            printf("%lld %lld %lld\n", data[i], data_copy[i], i);
        }
    }


    // GZKP approach

    cudaMemcpy(data_d, data_copy, length * sizeof(*data_d), cudaMemcpyHostToDevice);

    NTT_GZKP(data_d, reverse_d, length, root, 6, 8);

    cudaMemcpy(tmp, data_d, sizeof(*data_d) * length, cudaMemcpyDeviceToHost);

    for (long long i = 0; i < length; i++) {
        if (data[i] != tmp[i]) {
            printf("%lld %lld %lld\n", data[i], tmp[i], i);
        }
    }

    
    // NTT(data, reverse, length, inv(root));

    // long long co = inv(length);
    // for (long long i = 0; i < length; i++) {
    //     if (data[i] * co % P != data_copy[i]) {
    //         std::cout << data[i] * co % P << " " << data_copy[i] << " " << i << std::endl;
    //     }
    // }
    cudaFree(data_d);
    cudaFree(reverse_d);

    delete [] data;
    delete [] data_copy;
    delete [] reverse;
    delete [] tmp;
    return 0;
}