#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 900
#endif

#include <cublasdx.hpp>
#include <cusolverdx.hpp>

#include <cooperative_groups.h>
#include <cuda/atomic>

#undef uint64_t
#define uint64_t unsigned long
#ifdef USE_NVSHMEM
#include <device/nvshmem_defines.h>
#include <device/nvshmemx_defines.h>
#endif

namespace cg = cooperative_groups;

// Default parameters for Cube8P2E on Hopper
#ifndef GROUP_SIZE
#define GROUP_SIZE 8
#define DUP_PER_RANK 2
#define SM_Ver 900
#define BLOCK_DIM 128
#endif

template <int N>
__device__ void gaussian_elimination_solve(float a[N][N], float b[N]) {
  int status;
  decltype(cusolverdx::Size<N>() +
           cusolverdx::Function<cusolverdx::function::posv>() +
           cusolverdx::Arrangement<cusolverdx::row_major,
                                   cusolverdx::row_major>() +
           cusolverdx::SM<SM_Ver>() + cusolverdx::Block() +
           cusolverdx::FillMode<cusolverdx::lower>() +
           cusolverdx::BlockDim<BLOCK_DIM>())()
      .execute(a[0], b, &status);
}

template <int M, int N, int K>
__device__ void matmulNT(float *a, float *b, float *c) {
  decltype(cublasdx::Size<M, N, K>() +
           cublasdx::Function<cublasdx::function::MM>() +
           cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
           cublasdx::SM<SM_Ver>() + cublasdx::Block() +
           cublasdx::BlockDim<BLOCK_DIM>())()
      .execute(1.f, a, b, 0.f, c);
}

template <int M, int N, int K>
__device__ void matmulNN(float *a, float *b, float *c) {
  decltype(cublasdx::Size<M, N, K>() +
           cublasdx::Function<cublasdx::function::MM>() +
           cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>() +
           cublasdx::SM<SM_Ver>() + cublasdx::Block() +
           cublasdx::BlockDim<BLOCK_DIM>())()
      .execute(1.f, a, b, 0.f, c);
}

constexpr int NC = GROUP_SIZE + GROUP_SIZE * DUP_PER_RANK;
constexpr int NV = GROUP_SIZE * DUP_PER_RANK * 2 + GROUP_SIZE + 2;

struct smem_variables {
  float dup_workload[GROUP_SIZE][DUP_PER_RANK];
  float b[NC];
  float a[NC][NV];
  float c[NV];
  float ax2[NC][NV];
  float ax2a[NC][NC];
  float x[NV];
  float ax2c[NC];
  float r[NV];
  float d[NV];
  float alpha;
  bool avail_flag;
};

extern "C" __global__ void get_solve_smem_size(int *size_output) {
  *size_output = sizeof(smem_variables);
}

extern "C" __global__ void kernel_solve(
    const int *workload, float *global_workload, const int *r2o,
    const int *phy2log, int n_experts_per_var, int n_experts_fixed,
    int *avail_num, float *result
#ifdef USE_NVSHMEM
    ,
    float *workload_buf_inter, uint64_t *workload_sig_inter,
    float **workload_buf_intra,
    cuda::atomic<uint32_t, cuda::thread_scope_system> **workload_sig_intra,
    nvshmem_team_t internode_team, int self_device, int node_size
#endif
) {
  extern __shared__ smem_variables smem[];

  const int n_groups = gridDim.x;

  const int pid = blockIdx.x;
  const int tid = threadIdx.x;
  const int dim = blockDim.x;

  const int n_experts_per_rank =
      DUP_PER_RANK * n_experts_per_var + n_experts_fixed;
  const int n_phy_experts_per_rank =
      n_experts_per_rank + DUP_PER_RANK * n_experts_per_var;
  const int n_experts_per_group = GROUP_SIZE * n_experts_per_rank;
  const int n_experts = n_groups * n_experts_per_group;
#ifdef USE_NVSHMEM
  if (internode_team != NVSHMEM_TEAM_INVALID) {
    // Allreduce workload across nodes
    int n_nodes = nvshmem_team_n_pes(internode_team);
    int self_node = nvshmem_team_my_pe(internode_team);
    // Use only thread 0 for IBRC to reduce message count
    if (pid == 0) {
      // 1. Copy workload -> workload_buf_inter using block 0
      for (int i = tid; i < n_experts; i += dim)
        workload_buf_inter[self_node * n_experts + i] = workload[i];
      __syncthreads();
      // 2. In-place allgather of workload_buf_inter
      if (tid == 0) {
        for (int stride = 1; stride < n_nodes; stride++) {
          int remote_node = (self_node + stride) % n_nodes;
          nvshmem_putmem_signal_nbi(
              &workload_buf_inter[self_node * n_experts],
              &workload_buf_inter[self_node * n_experts],
              n_experts * sizeof(float), workload_sig_inter, 1,
              NVSHMEM_SIGNAL_ADD,
              nvshmem_team_translate_pe(internode_team, remote_node,
                                        NVSHMEM_TEAM_WORLD));
        }
        nvshmem_signal_wait_until(workload_sig_inter, NVSHMEM_CMP_GE,
                                  workload_sig_inter[1] + n_nodes - 1);
        workload_sig_inter[1] += n_nodes - 1;
      }
      __syncthreads();
      // 3. Sum workload_buf_inter across nodes into
      // workload_buf_intra[self_device]
      for (int i = tid; i < n_experts; i += dim) {
        float sum = 0.0f;
        for (int node = 0; node < n_nodes; node++)
          sum += workload_buf_inter[node * n_experts + i];
        workload_buf_intra[self_device][i] = sum;
      }
      __syncthreads();
      if (tid == 0) {
        for (int i = 0; i < node_size; ++i)
          workload_sig_intra[i]->fetch_add(1, cuda::std::memory_order_release);
        auto target = workload_sig_intra[self_device][1] + node_size;
        while (workload_sig_intra[self_device]->load(
                   cuda::memory_order_acquire) < target)
          ;
        workload_sig_intra[self_device][1] = target;
      }
      __syncthreads();
      // 4. Allreduce workload_buf_intra to global_workload
      for (int i = tid; i < n_experts; i += dim) {
        float sum = 0.0f;
        for (int device = 0; device < node_size; device++)
          sum += workload_buf_intra[(device + self_device) % node_size][i];
        global_workload[i] = sum;
      }
      __syncthreads();
      // 5. Scale global_workload to have max value of 1
      if (tid < 32) {
        float workload_max = 0.0f;
        for (int i = tid; i < n_experts; i += 32) {
          workload_max = fmaxf(workload_max, global_workload[i]);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
          workload_max = fmaxf(
              workload_max, __shfl_xor_sync(0xffffffff, workload_max, offset));
        for (int i = tid; i < n_experts; i += 32)
          global_workload[i] /= workload_max;
      }
    }
    cg::this_grid().sync();
  } else {
#endif
    // Still need to scale workload
    if (pid == 0 && tid < 32) {
      float workload_max = 0.0f;
      for (int i = tid; i < n_experts; i += 32)
        workload_max = fmaxf(workload_max, workload[i]);
      for (int offset = 16; offset > 0; offset >>= 1)
        workload_max = fmaxf(workload_max,
                             __shfl_xor_sync(0xffffffff, workload_max, offset));
      for (int i = tid; i < n_experts; i += 32)
        global_workload[i] = workload[i] / workload_max;
#ifdef DEBUG_DUMP
      printf("tid=%d workload_max=%f\n", tid, workload_max);
#endif
    }
    cg::this_grid().sync();
#ifdef USE_NVSHMEM
  }
#endif

  result += pid * GROUP_SIZE * DUP_PER_RANK;
  phy2log += pid * GROUP_SIZE * n_phy_experts_per_rank;
  auto &&dup_workload = smem->dup_workload;

  for (int i = tid; i < GROUP_SIZE * DUP_PER_RANK; i += dim) {
    int rank = i / DUP_PER_RANK;
    int dup = i % DUP_PER_RANK;
    dup_workload[rank][dup] = 0;
    for (int j = 0; j < n_experts_per_var; ++j) {
      dup_workload[rank][dup] +=
          global_workload[phy2log[rank * n_phy_experts_per_rank +
                                  j * DUP_PER_RANK + dup]];
#ifdef DEBUG_DUMP
      if (pid == 0)
        printf("rank=%d dup=%d global_workload[phy2log[%d] = %d] = %f\n", rank,
               dup, rank * n_phy_experts_per_rank + j * DUP_PER_RANK + dup,
               phy2log[rank * n_phy_experts_per_rank + j * DUP_PER_RANK + dup],
               global_workload[phy2log[rank * n_phy_experts_per_rank +
                                       j * DUP_PER_RANK + dup]]);
#endif
    }
  }
  __syncthreads();

  auto &&b = smem->b;
  for (int i = tid; i < NC; i += dim) {
    if (i < GROUP_SIZE) {
      b[i] = 0;
      for (int j = 0; j < n_experts_fixed; ++j)
        b[i] -= global_workload[phy2log[i * n_phy_experts_per_rank +
                                        DUP_PER_RANK * n_experts_per_var + j]];
    } else {
      b[i] = 1;
    }
  }
  __syncthreads();

  auto &&a = smem->a;
  for (int i = tid; i < NC * NV; i += dim) {
    int ic = i / NV, iv = i % NV;
    if (ic < GROUP_SIZE) {
      if (iv < GROUP_SIZE * DUP_PER_RANK) {
        int var_rank = iv / DUP_PER_RANK, var_dup = iv % DUP_PER_RANK;
        // Ratio assigning expert workload to original
        if (ic == var_rank)
          a[ic][iv] = dup_workload[var_rank][var_dup];
        // Experts not related to current rank
        else
          a[ic][iv] = 0;
      } else if (iv < GROUP_SIZE * DUP_PER_RANK * 2) {
        int var_rank = iv / DUP_PER_RANK - GROUP_SIZE,
            var_dup = iv % DUP_PER_RANK;
        // Ratio assigning expert workload to replica
        if (var_rank == r2o[ic * DUP_PER_RANK + var_dup])
          a[ic][iv] = dup_workload[var_rank][var_dup];
        // Experts not related to current rank
        else
          a[ic][iv] = 0;
      } else if (iv < GROUP_SIZE * DUP_PER_RANK * 2 + GROUP_SIZE) {
        // Slack variable representing difference between max load in group and
        // current rank's load (coefficient 1)
        a[ic][iv] = (iv - GROUP_SIZE * DUP_PER_RANK * 2 == ic);
      } else if (iv == GROUP_SIZE * DUP_PER_RANK * 2 + GROUP_SIZE) {
        // Max value variable (coefficient -1)
        a[ic][iv] = -1;
      }
    } else {
      // Constraints for replica assignment
      if (iv == ic - GROUP_SIZE ||
          iv - GROUP_SIZE * DUP_PER_RANK == ic - GROUP_SIZE)
        a[ic][iv] = 1;
      else
        a[ic][iv] = 0;
    }
  }
  __syncthreads();
  // Column required for Big M method to ensure A @ vec(1) = b
  for (int ic = tid; ic < NC; ic += dim) {
    float *a_M_col = &a[ic][GROUP_SIZE * DUP_PER_RANK * 2 + GROUP_SIZE + 1];
    *a_M_col = b[ic];
    for (int iv = 0; iv < GROUP_SIZE * DUP_PER_RANK * 2 + GROUP_SIZE + 1; ++iv)
      *a_M_col -= a[ic][iv];
  }
  __syncthreads();

  auto &c = smem->c;
  for (int iv = tid; iv < NV; iv += dim) {
    if (iv < NV - 2)
      c[iv] = 0;
    else if (iv == NV - 2)
      c[iv] = 1;
    else
      c[iv] = 1000;
  }
  __syncthreads();

#ifdef DEBUG_DUMP
  if (tid == 0 && pid == 0) {
    // print all of dup_workload
    printf("dup_workload:\n");
    for (int dup = 0; dup < DUP_PER_RANK; dup++) {
      for (int rank = 0; rank < GROUP_SIZE; rank++)
        printf("%.2f ", dup_workload[rank][dup]);
      printf("\n");
    }
    // print all of a, b, c
    printf("A:\n");
    for (int ic = 0; ic < NC; ic++) {
      for (int iv = 0; iv < NV; iv++)
        printf("%.2f ", a[ic][iv]);
      printf("\n");
    }
    printf("b:\n");
    for (int ic = 0; ic < NC; ic++)
      printf("%.2f ", b[ic]);
    printf("\n");
    printf("c:\n");
    for (int iv = 0; iv < NV; iv++)
      printf("%.2f ", c[iv]);
    printf("\n");
  }
  __syncthreads();
#endif

  auto &&ax2 = smem->ax2;
  auto &&ax2a = smem->ax2a;
  auto &&x = smem->x;
  auto &&ax2c = smem->ax2c;
  auto &&r = smem->r;
  auto &&d = smem->d;
  auto &&alpha = smem->alpha;
  float d_max, max_residual;
  for (int j = tid; j < NV; j += dim)
    x[j] = 1;
  __syncthreads();

  for (int step = 0; step < 5; step++) {
    for (int ij = tid; ij < NC * NV; ij += dim) {
      int i = ij / NV, j = ij % NV;
      ax2[i][j] = a[i][j] * x[j] * x[j];
    }
    __syncthreads();

    // ax2a = ax2 @ a.T
    matmulNT<NC, NC, NV>(ax2[0], a[0], ax2a[0]);

    // ax2c = ax2 @ c.unsqueeze(1)
    matmulNT<NC, 1, NV>(ax2[0], c, ax2c);

    // solve(ax2a, ax2c);
    gaussian_elimination_solve<NC>(ax2a, ax2c);

    // r = ax2c.unsqueeze(0) @ a
    matmulNN<1, NV, NC>(ax2c, a[0], r);

    if (tid < 32) {
      d_max = 0.f;
      for (int j = tid; j < NV; j += 32)
        d_max = d[j] = x[j] * (c[j] - r[j]);
      for (int offset = 16; offset > 0; offset >>= 1)
        d_max = fmaxf(d_max, __shfl_xor_sync(0xffffffff, d_max, offset));
      if (tid == 0)
        alpha = 0.999 / d_max;
    }
    __syncthreads();
    for (int j = tid; j < NV; j += dim)
      x[j] *= 1 - alpha * d[j];
    __syncthreads();
  }

  // ax2c -> residual
  matmulNT<NC, 1, NV>(a[0], x, ax2c);
  if (tid < 32) {
    max_residual = 0;
    for (int i = tid; i < NC; i += 32)
      max_residual = fmaxf(max_residual, fabsf(ax2c[i] - b[i]));
    for (int offset = 16; offset > 0; offset >>= 1)
      max_residual = fmaxf(max_residual,
                           __shfl_down_sync(0xffffffff, max_residual, offset));
  }
  auto &&avail_flag = smem->avail_flag;
  if (tid == 0) {
    avail_flag = (d_max < 0.1 && 0 <= x[NV - 1] && x[NV - 1] < 1e-4 &&
                  max_residual < 0.05);
    atomicAdd(avail_num, (int)avail_flag);
  }
  __syncthreads();
#ifdef DEBUG_DUMP
  if (tid == 0 && pid == 0) {
    // print all of x
    printf("x:\n");
    for (int iv = 0; iv < NV; iv++)
      printf("%.2f ", x[iv]);
    printf("\n");
  }
#endif
  if (!avail_flag)
    for (int i = tid; i < GROUP_SIZE * DUP_PER_RANK; i += dim)
      result[i] = 0.5;
  else
    for (int i = tid; i < GROUP_SIZE * DUP_PER_RANK; i += dim)
      result[i] = x[i];
  __syncthreads();
}

int split_and_align(int n, int par_rank, int par_size, int align) {
  int n_per_rank = (n + par_size - 1) / par_size;
  n_per_rank = (n_per_rank + align - 1) / align * align;
  return min(n, n_per_rank * par_rank);
}

extern "C" __global__ void kernel_count_idx(const long *idx,
                                            const int n_elements,
                                            const int n_experts, int *counts) {
  const int pid = blockIdx.x;
  const int n_prog = gridDim.x;
  const int block_size = blockDim.x;
  const int tid = threadIdx.x;

  extern __shared__ int smem_buffer[];
  int *smem_counts = smem_buffer + 16; // allow accessing smem_counts[-1]

  // init smem_counts to 0
  for (int i = tid; i < n_experts; i += block_size)
    smem_counts[i] = 0;
  __syncthreads();

  const int start = split_and_align(n_elements, pid, n_prog, block_size);
  const int end = split_and_align(n_elements, pid + 1, n_prog, block_size);
  for (int i = start + tid; i < end; i += block_size) {
    assert(idx[i] >= -1);
    assert(idx[i] < n_experts);
    atomicAdd(&smem_counts[idx[i]], 1);
  }
  __syncthreads();

  for (int i = tid; i < n_experts; i += block_size)
    counts[pid * n_experts + i] = smem_counts[i];
  cg::this_grid().sync();

  if (tid < 32)
    for (int i = pid * 32 + tid; i < n_experts; i += n_prog * 32)
      for (int sm = 1; sm < n_prog; sm++)
        counts[sm * n_experts + i] += counts[(sm - 1) * n_experts + i];
}

extern "C" __global__ void
kernel_map_idx(const long *mapping_idx, const float *o_weight,
               const int *local_workload_by_sm, const int *o2r,
               const int *phy2log, const int n_elements, const int n_group,
               const int n_combined_experts, const int n_local_experts,
               long *mapping_idx_out) {
  const int pid = blockIdx.x;
  const int n_prog = gridDim.x;
  const int block_size = blockDim.x;
  const int tid = threadIdx.x;

  const int n_local_logical_experts =
      n_local_experts - DUP_PER_RANK * n_combined_experts;
  const int n_logical_experts = n_group * GROUP_SIZE * n_local_logical_experts;

  extern __shared__ int smem_buffer[];
  int *smem_total_count = smem_buffer;
  int *smem_expected_count = smem_total_count + n_logical_experts;
  int *smem_current_count = smem_expected_count + n_logical_experts;
  int *smem_log2r = smem_current_count + n_logical_experts;

  for (int i = tid; i < n_logical_experts; i += block_size) {
    smem_total_count[i] = smem_expected_count[i] =
        local_workload_by_sm[(n_prog - 1) * n_logical_experts + i];
    smem_current_count[i] =
        pid == 0 ? 0 : local_workload_by_sm[(pid - 1) * n_logical_experts + i];
    int i_with_redundants = (i / n_local_logical_experts) * n_local_experts +
                            (i % n_local_logical_experts);
    smem_log2r[phy2log[i_with_redundants] * 2] = i_with_redundants;
    smem_log2r[i * 2 + 1] = -1;
  }
  __syncthreads();

  for (int i = tid;
       i < n_group * GROUP_SIZE * n_combined_experts * DUP_PER_RANK;
       i += block_size) {
    int ep_rank = i / (n_combined_experts * DUP_PER_RANK);
    int local_expert_idx = i % (n_combined_experts * DUP_PER_RANK);
    int logical_expert = phy2log[ep_rank * n_local_experts + local_expert_idx];
    smem_expected_count[logical_expert] =
        o_weight[ep_rank * DUP_PER_RANK + local_expert_idx % DUP_PER_RANK] *
        local_workload_by_sm[(n_prog - 1) * n_logical_experts + logical_expert];

    int i_group = ep_rank / GROUP_SIZE;
    int group_rank = ep_rank % GROUP_SIZE;
    smem_log2r[logical_expert * 2 + 1] =
        (i_group * GROUP_SIZE +
         o2r[group_rank * DUP_PER_RANK + local_expert_idx % DUP_PER_RANK] + 1) *
            n_local_experts -
        n_combined_experts * DUP_PER_RANK + local_expert_idx;
  }
  __syncthreads();

  const int start = split_and_align(n_elements, pid, n_prog, block_size);
  const int end = split_and_align(n_elements, pid + 1, n_prog, block_size);
  for (int i = start + tid; i < end; i += block_size) {
    auto idx = mapping_idx[i];
    int idx_out;
    if (idx != -1) {
      auto computed_count = atomicAdd(&smem_current_count[idx], 1);
      // 499 is prime, so it permutes positions unless the count is a
      // multiple of 499. Use a rotation for that case. Unsigned arithmetic
      // avoids signed overflow; widen only when the product cannot fit.
      const auto total_count = static_cast<unsigned int>(smem_total_count[idx]);
      const auto ordinal = static_cast<unsigned int>(computed_count);
      constexpr auto max_fast_count = (~0u - 41u) / 499u + 1u;
      if (total_count % 499u == 0)
        computed_count = (ordinal + 41u) % total_count;
      else if (total_count <= max_fast_count)
        computed_count = (ordinal * 499u + 41u) % total_count;
      else
        computed_count = (static_cast<unsigned long long>(ordinal) * 499u + 41u) % total_count;
      idx_out =
          smem_log2r[mapping_idx[i] * 2 +
                     (computed_count >= smem_expected_count[mapping_idx[i]])];
    } else {
      idx_out = -1;
    }
    mapping_idx_out[i] = idx_out;
  }
  __syncthreads();
}
