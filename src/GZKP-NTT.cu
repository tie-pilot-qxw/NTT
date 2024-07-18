#include <iostream>
#include <random>
#include <cassert>
#include <cuda_runtime.h>
#include <ctime>

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
    cudaEventRecord(start);
    rearrange <<< grid, block >>>(data, reverse, reverse_num);
    for (long long stride = 1ll; stride < len; stride <<= 1ll) {
        naive <<< grid1, block >>>(data, len, roots_d, stride);
        // cudaMemcpy(tmp, data, sizeof(*tmp) * len, cudaMemcpyDeviceToHost);
        // for (int i = 0; i < len; i++) printf("%lld ", tmp[i]);
        // printf("\n");
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
        if (step > 32) __syncthreads();

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
    cudaEventRecord(start);

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

__global__ void naive_no_swap(long long x[], long long y[], long long len, long long roots[], uint stride) {
    uint i = blockDim.x * blockIdx.x + threadIdx.x;
    uint t = len / 2;
    if(i + t >= len) return;
    uint k = i & (stride - 1);

    long long a = x[i], b = x[i + t];
    long long w = roots[k * (len / (stride << 1))];
    b = b * w % P;
    long long tmp = a + P - b;
    if (tmp >= P) tmp -= P;
    a += b;
    if (a >= P) a -= P;
    b = tmp;

    uint j = (i << 1) - k;
    y[j] = a;
    y[j + stride] = b;
}

long long * No_Swap(long long *x, long long *y, long long len, long long omega) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);


    long long *roots, *roots_d;
    roots = new long long [len];
    long long gap = qpow(omega, (P - 1ll) / len);
    roots[0] = 1;
    for (long long i = 1; i < len; i++) roots[i] = roots[i - 1] * gap % P;
    
    cudaMalloc(&roots_d, len * sizeof(*roots_d));
    cudaMemcpy(roots_d, roots, len * sizeof(*roots_d), cudaMemcpyHostToDevice);

    cudaEventRecord(start);


    dim3 block(768);
    dim3 grid1((len / 2 - 1) / block.x + 1);
    long long *res = new long long [len];

    for (long long stride = 1ll; stride < len; stride <<= 1ll) {
        naive_no_swap <<< grid1, block >>>(x, y, len, roots_d, stride);
        long long *tmp = x;
        x = y;
        y = tmp;
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float t;
    cudaEventElapsedTime(&t, start, end);

    printf("no_swap: %fms\n", t);

    cudaFree(roots_d);
    delete [] roots;
    return x;
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

/*
 * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
 */
__global__ void FIELD_radix_fft(long long * x, // Source buffer
                      long long * y, // Destination buffer
                      long long * pq, // Precalculated twiddle factors
                      long long * omegas, // [omega, omega^2, omega^4, ...]
                      uint n, // Number of elements
                      uint lgp, // Log2 of `p` (Read more in the link above)
                      uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                      uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
{
// CUDA doesn't support local buffers ("shared memory" in CUDA lingo) as function arguments,
// ignore that argument and use the globally defined extern memory instead.

    // There can only be a single dynamic shared memory item, hence cast it to the type we need.
    extern __shared__ long long u[];

    uint lid = threadIdx.x;//GET_LOCAL_ID();
    uint lsize = blockDim.x;//GET_LOCAL_SIZE();
    uint index = blockIdx.x;//GET_GROUP_ID();
    uint t = n >> deg;
    uint p = 1 << lgp;
    uint k = index & (p - 1);

    x += index;
    y += ((index - k) << deg) + k;

    uint count = 1 << deg; // 2^deg
    uint counth = count >> 1; // Half of count

    uint counts = count / lsize * lid;
    uint counte = counts + count / lsize;

    // Compute powers of twiddle
    const long long twiddle = FIELD_pow_lookup(omegas, (n >> lgp >> deg) * k);
    long long tmp = FIELD_pow(twiddle, counts);
    for(uint i = counts; i < counte; i++) {
      u[i] = (tmp * x[i*t]) % P;
      tmp = (tmp * twiddle) % P;
    }

    __syncthreads();

    const uint pqshift = max_deg - deg;
    for(uint rnd = 0; rnd < deg; rnd++) {
      const uint bit = counth >> rnd;
      for(uint i = counts >> 1; i < counte >> 1; i++) {
        const uint di = i & (bit - 1);
        const uint i0 = (i << 1) - di;
        const uint i1 = i0 + bit;
        tmp = u[i0];
        u[i0] = (u[i0] + u[i1]) % P;
        u[i1] = (tmp + P - u[i1]) % P;
        if(di != 0) u[i1] = (pq[di << rnd << pqshift] * u[i1]) % P;
      }

      __syncthreads();
    }
    

    for(uint i = counts >> 1; i < counte >> 1; i++) {
        y[i*p] = u[__brev(i) >> (32 - deg)];
        y[(i+counth)*p] = u[__brev(i + counth) >> (32 - deg)];
    }
}

#define MAX_LOG2_RADIX 8u
long long * bellperson_baseline(long long *x, long long *y,long long omega, uint log_n) {

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);



    uint n = 1 << log_n;

    omega = qpow(omega, (P - 1ll) / n);
    // All usages are safe as the buffers are initialized from either the host or the GPU
    // before they are read.
    // let mut src_buffer = unsafe { program.create_buffer::<F>(n)? };
    // let mut dst_buffer = unsafe { program.create_buffer::<F>(n)? };
    // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
    uint max_deg = std::min(MAX_LOG2_RADIX, log_n);

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
    uint log_p = 0u;
    
    cudaEventRecord(start);

    // Each iteration performs a FFT round
    while (log_p < log_n) {

        // 1=>radix2, 2=>radix4, 3=>radix8, ...
        uint deg = std::min(max_deg, log_n - log_p);

        uint n = 1u << log_n;
        dim3 block(1 << std::min(deg - 1, 10u) );
        uint grid(n >> deg);

        FIELD_radix_fft <<< grid, block, sizeof(long long) * (1 << deg) >>>(x, y, pq_d, omegas_d, n, log_p, deg, max_deg);

        log_p += deg;
        long long * tmp = x;
        x = y;
        y = tmp;
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
    return x;
}


int main() {
    long long *data, *reverse, *data_copy;
    long long l,length = 1ll;
    int bits = 0;

    //scanf("%lld", &l);
    l = qpow(2, 26);

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

    NTT_GZKP(data_d, reverse2_d, length, root, 8, 8, reverse_num);

    cudaMemcpy(tmp, data_d, sizeof(*data_d) * length, cudaMemcpyDeviceToHost);

    for (long long i = 0; i < length; i++) {
        if (data[i] != tmp[i]) {
            printf("%lld %lld %lld\n", data[i], tmp[i], i);
        }
    }

    // NO swap
    long long *data_p;
    cudaMalloc(&data_p, length * sizeof(*data_p));

    cudaMemcpy(data_d, data_copy, length * sizeof(*data_d), cudaMemcpyHostToDevice);
    long long *res = No_Swap(data_d, data_p, length, root);
    cudaMemcpy(tmp, res, sizeof(*res) * length, cudaMemcpyDeviceToHost);

    for (long long i = 0; i < length; i++) {
        if (data[i] != tmp[i]) {
            printf("%lld %lld %lld\n", data[i], tmp[i], i);
        }
    }

    // bellperson
    cudaMemcpy(data_d, data_copy, length * sizeof(*data_d), cudaMemcpyHostToDevice);
    res = bellperson_baseline(data_d, data_p, root, bits);
    cudaMemcpy(tmp, res, sizeof(*res) * length, cudaMemcpyDeviceToHost);
    for (long long i = 0; i < length; i++) {
        if (data[i] != tmp[i]) {
            printf("%lld %lld %lld\n", data[i], tmp[i], i);
        }
    }

    cudaFree(data_p);
    
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