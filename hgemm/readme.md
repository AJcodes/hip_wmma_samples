# Understanding HIP and WMMA Intrinsics

This project is a personal exploration of HIP programming and the RDNA3 Wave Matrix Multiply-Accumulate (WMMA) intrinsic. The primary goal was to deepen my understanding of the WMMA intrinsic and extend the fixed-size example provided in the [GPUOpen tutorial](https://gpuopen.com/learn/wmma_on_rdna3/) to support arbitrary matrix dimensions. While this project is primarily for personal learning, it may also serve as a helpful reference for others interested in exploring the WMMA intrinsic.

**Note:** The WMMA intrinsic is specific to RDNA3 GPUs for now, so running this project requires an RDNA3-compatible GPU. A future feature may include testing this implementation on RDNA4 hardware when it becomes available. For production-grade GPU matrix multiplication, it is highly recommended to use [rocWMMA](https://github.com/ROCm/rocWMMA), which provides a robust and optimized abstraction over the WMMA functionality.

## Objectives
This project aims to:
1. Provide a simple example of HIP programming and WMMA usage for GPU-accelerated computation
2. Extend beyond the fixed-size example in the GPUOpen tutorial by supporting arbitrary matrix dimensions (M, N, K)
3. Enhance understanding of the WMMA intrinsic's mechanics, especially around data loading and storing

## Features

- **Flexible Matrix Dimensions:** Supports arbitrary matrix sizes (M, N, K) beyond the basic 16x16 example
- **Multiple Implementations:**
  - Basic WMMA implementation
  - Shared memory optimized WMMA
  - Traditional shared memory implementation (for comparison)
- **Performance Benchmarking:** Built-in benchmarking capabilities for comparing different implementations
- **Correctness Verification:** CPU reference implementation for result validation

## Performance Results

Performance measured on AMD Radeon RX 7900 GRE on Windows (HIP SDK 6.2.4). All implementations use half precision (FP16).

### Performance for 1024x1024 Matrix Multiplication
| Implementation | Time (ms) | TFLOPs/s |
|----------------|-----------|-----------|
| Shared Memory | 0.561 | 3.82 |
| WMMA Naive | 0.406 | 5.28 |
| WMMA + Shared Memory | 0.185 | 11.57 |
| WMMA + Shared Memory + Warp Tiling | 0.284 | 7.54 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.209 | 10.23 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.072 | 29.72 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.067 | 31.95 |
| WMMA Prefetch | 0.065 | 32.91 |
| rocWMMA | 0.083 | 25.82 |
| rocBLAS | 0.053 | 40.46 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Time (ms) | TFLOPs/s |
|----------------|-----------|-----------|
| Shared Memory | 4.537 | 3.78 |
| WMMA Naive | 3.191 | 5.37 |
| WMMA + Shared Memory | 1.336 | 12.83 |
| WMMA + Shared Memory + Warp Tiling | 0.934 | 18.35 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.965 | 17.76 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.426 | 40.23 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.335 | 51.19 |
| WMMA Prefetch | 0.351 | 49.00 |
| rocWMMA | 0.416 | 41.20 |
| rocBLAS | 0.341 | 50.25 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Time (ms) | TFLOPs/s |
|----------------|-----------|-----------|
| Shared Memory | 35.73 | 3.84 |
| WMMA Naive | 21.94 | 6.25 |
| WMMA + Shared Memory | 10.49 | 13.06 |
| WMMA + Shared Memory + Warp Tiling | 5.73 | 23.92 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 5.51 | 24.89 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.46 | 55.75 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.24 | 61.23 |
| WMMA Prefetch | 2.35 | 58.40 |
| rocWMMA | 2.76 | 49.68 |
| rocBLAS | 2.31 | 59.36 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Time (ms) | TFLOPs/s |
|----------------|-----------|-----------|
| Shared Memory | 309.09 | 3.52 |
| WMMA Naive | 195.27 | 5.57 |
| WMMA + Shared Memory | 95.36 | 11.41 |
| WMMA + Shared Memory + Warp Tiling | 47.79 | 22.77 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 47.41 | 22.95 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 25.96 | 41.92 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 18.61 | 58.47 |
| WMMA Prefetch | 19.05 | 57.21 |
| rocWMMA | 22.95 | 47.37 |
| rocBLAS | 21.29 | 51.07 |

Key observations:
1. Each optimization step provides significant performance improvements
2. Global vectorized loads provide the largest single performance boost
3. The most optimized WMMA implementation achieves performance comparable to or better than rocBLAS for larger matrices
4. Performance scaling improves with matrix size:
   - At 1024x1024: ~32 TFLOPs/s peak
   - At 2048x2048: ~51 TFLOPs/s peak
   - At 4096x4096: ~61 TFLOPs/s peak
   - At 8192x8192: ~58 TFLOPs/s peak
5. Smaller matrices (1024x1024) show more variance between implementations

## Known Issues

- The WMMA HGEMM kernels using shared memory have stability issues when K > M, N

## Usage

Run the executable after building:
```bash
# Assumes you're currently in /build directory
./hgemm/hgemm
```

### Customizing Tests

- Edit `.verify_sizes` in main.cpp to add specific matrix sizes for correctness validation
- Edit `.benchmark_sizes` to specify sizes for performance testing

## Future Improvements

1. **WMMA HGEMM Optimization:**
   - Explore additional optimization techniques
   - Implement advanced tiling strategies
   - Investigate various data loading patterns
   - Improve shared memory utilization

2. **Implementation Enhancements:**
   - Add support for different data layouts
   - Implement more sophisticated benchmarking tools
   - Enhance performance measurement and analysis

3. **Bug Fixes:**
   - Address WMMA + shared memory issues when K > M, N
