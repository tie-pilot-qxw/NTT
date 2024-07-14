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

__global__ void rearrange(long long * data, longlong2 * reverse, long long len) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    longlong2 r = reverse[index];
    long long tmp = data[r.x];
    data[r.x] = data[r.y];
    data[r.y] = tmp;
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

void NTT_GPU_Naive(long long data[], longlong2 reverse[], long long len, long long omega, long long reverse_num) {
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
    dim3 grid((reverse_num - 1) / block.x + 1);
    dim3 grid1((len / 2 - 1) / block.x + 1);
    rearrange <<< grid, block >>>(data, reverse, reverse_num);
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

__global__ void GZKP (long long data[],  int len, long long roots[], int stride, int G, int csize, int B) {
    extern __shared__ long long s[];

    int start = (G * blockIdx.x) % stride + ((int) G * blockIdx.x / stride) * stride * csize * 2;

    int iogroup = threadIdx.x / G; // io group
    int ioid = threadIdx.x % G; // id in the io group

    int spos; // load to
    int gpos; // load from

    // (stride >= G) 
    spos = ioid * csize * 2 + 2 * iogroup;
    gpos = start + ioid + iogroup * 2 * stride;

    // load to shared memory
    if (gpos + stride >= len) return;

    s[spos] = data[gpos];
    s[spos + 1] = data[gpos + stride];
    __syncthreads();
    
    int cid = threadIdx.x % csize; // id of the compute group
    int base =  (threadIdx.x - cid) << 1; // base pos of the compute group
    
    for (int i = 1, step = 1; i <= B && stride * step < len; i++, step *= 2) {

        int offset = step * 2 * ((int)(cid / step)) + cid % step;

        int io_thread_id = threadIdx.x / csize + offset / 2 * G;

        iogroup = (io_thread_id) / G;
        ioid = (io_thread_id) - iogroup * G;

        int tmp = start + ioid + iogroup * 2 * stride + (offset & 1) * stride;
        tmp = tmp % (stride * step * 2);
        tmp = tmp * (len / (stride * step * 2));

        long long w = roots[tmp];

        long long a = s[base + offset], b = w * s[base + offset + step] % P;
        s[base + offset] = (a + b) % P;
        s[base + offset + step] = (a - b + P) % P;

    }
    
    __syncthreads();
    data[gpos] = s[spos];
    data[gpos + stride] = s[spos + 1];
}

void NTT_GZKP(long long data[], longlong2 reverse[], long long len, long long omega, int B, int G, long long reverse_num) {
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

    dim3 block0(768);
    dim3 grid0((reverse_num - 1) / block0.x + 1);
    dim3 grid1((len / 2 - 1) / block.x + 1);

    rearrange <<< grid0, block0 >>>(data, reverse, reverse_num);

    long long stride = 1ll;
    for (; stride < G; stride <<= 1) {
        naive <<< grid1, block >>>(data, len, roots_d, stride);
    }

    for (; stride << B <= len; stride <<= B) {
        GZKP <<< grid1, block, sizeof(long long) * qpow(2,B) * G>>>(data, len, roots_d, stride, G, qpow(2,B)/2, B);
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
        GZKP <<< grid1, block, sizeof(long long) * qpow(2,B) * G>>>(data, len, roots_d, stride, G, qpow(2,B)/2, B);
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
    l = 60000000;

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
        data[i] = std::abs((long long)gen()) % P;
        data_copy[i] = data[i];
    }

    // cpu implementation
    {
        clock_t start = clock();

        NTT(data, reverse, length, root);

        clock_t end = clock();
        printf("cpu: %lfms\n",(double)(end - start) / CLOCKS_PER_SEC * 1000);
    }

    // revise the reverse array to elimate branch divergence
    longlong2 *reverse2 = (longlong2 *)malloc(length / 2 * sizeof(longlong2));
    long long reverse_num =  0;
    
    for (long long i = 0; i < length; i++) {
        if (reverse[i] < i) {
            reverse2[reverse_num] = make_longlong2(reverse[i], i);
            reverse_num++;
        }
    }
    longlong2 *reverse2_d;
    long long *data_d;

    cudaMalloc(&data_d, length * sizeof(*data_d));
    cudaMalloc(&reverse2_d, length / 2 * sizeof(*reverse2_d));
    cudaMemcpy(data_d, data_copy, length * sizeof(*data_d), cudaMemcpyHostToDevice);
    cudaMemcpy(reverse2_d, reverse2, length / 2 * sizeof(*reverse2), cudaMemcpyHostToDevice);

    // naive gpu approach
    NTT_GPU_Naive(data_d, reverse2_d, length, root, reverse_num);

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

    NTT_GZKP(data_d, reverse2_d, length, root, 5, 64, reverse_num);

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
    cudaFree(reverse2_d);

    delete [] data;
    delete [] data_copy;
    delete [] reverse;
    delete [] tmp;
    free(reverse2);

    return 0;
}