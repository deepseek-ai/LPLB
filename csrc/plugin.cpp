
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include <pybind11/pybind11.h>

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/extension.h>

#include <filesystem>
#include <random>

#ifdef USE_NVSHMEM
#include "deepep_rt_slim.h"
#endif

#define NVRTC_CHECK(x)                                                         \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      std::ostringstream oss;                                                  \
      oss << "\nerror: " #x " failed with error "                              \
          << nvrtcGetErrorString(result) << '\n';                              \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t result = x;                                                    \
    if (result != cudaSuccess) {                                               \
      std::ostringstream oss;                                                  \
      oss << "\nerror: " #x " failed with error " << cudaGetErrorName(result)  \
          << '\n';                                                             \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

#define NVJITLINK_CHECK(h, x)                                                  \
  do {                                                                         \
    nvJitLinkResult result = x;                                                \
    if (result != NVJITLINK_SUCCESS) {                                         \
      std::ostringstream oss;                                                  \
      oss << "\nerror: " #x " failed with error " << result << '\n';           \
      size_t lsize;                                                            \
      result = nvJitLinkGetErrorLogSize(h, &lsize);                            \
      if (result == NVJITLINK_SUCCESS && lsize > 0) {                          \
        char *log = (char *)malloc(lsize);                                     \
        result = nvJitLinkGetErrorLog(h, log);                                 \
        if (result == NVJITLINK_SUCCESS) {                                     \
          oss << "error: " << log << '\n';                                     \
          free(log);                                                           \
        }                                                                      \
      }                                                                        \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

#define LPLB_ASSERT(x)                                                         \
  do {                                                                         \
    if (!(x))                                                                  \
      throw std::runtime_error("assertion failed: " #x);                       \
  } while (0)

#ifdef USE_NVSHMEM

void sync_current_to_module(cudaLibrary_t module, const char *symbol_name) {
  auto lib_handle = dlopen(DEEP_EP_SO, RTLD_LAZY | RTLD_LOCAL);
  if (!lib_handle)
    throw std::runtime_error(std::string("Cannot load library: ") + dlerror());
  dlerror(); // Clear any existing error
  const void *symbol = dlsym(lib_handle, symbol_name);
  void *devPtr_A;
  CUDA_CHECK(cudaGetSymbolAddress(&devPtr_A, symbol));
  void *devPtr_B;
  size_t size_B;
  CUDA_CHECK(cudaLibraryGetGlobal(&devPtr_B, &size_B, module, symbol_name));
  CUDA_CHECK(cudaMemcpy(devPtr_B, devPtr_A, size_B, cudaMemcpyDeviceToDevice));
}

#endif

struct vector_string_hash {
  std::size_t operator()(const std::vector<std::string> &vec) const {
    std::size_t seed = vec.size();
    for (const auto &str : vec) {
      // Combine hashes of individual strings
      seed ^= std::hash<std::string>()(str) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
    }
    return seed;
  }
};

void shared_mkdir(const std::string name) { mkdir(name.c_str(), 0775); }

struct compiled_solver {
  cudaKernel_t kernel_solve = nullptr, kernel_map_idx = nullptr,
               kernel_count_idx = nullptr;
  cudaLibrary_t module = nullptr;
  int n_group;
  int group_size;
  int dup_per_rank;
  int block_dim;
  int smem_size = -1;

  int n_nodes = 1, node_size = 1, self_node = 0, self_device = 0;
#ifdef USE_NVSHMEM
  nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
#endif

  float *workload_buf_internode = nullptr;
  uint64_t *workload_sig_internode = nullptr;

  float **workload_buf_intranode = nullptr;
  uint32_t **workload_sig_intranode = nullptr;
  std::vector<float *> workload_buf_intranode_cpu;
  std::vector<uint32_t *> workload_sig_intranode_cpu;

  int n_experts, n_local_experts, n_combined_experts;

  c10::intrusive_ptr<c10d::ProcessGroup> pg;

  std::string compile_cubin(const std::string &resource_path) {
    // Dynamically determine the arch to link for
    int device = c10::cuda::current_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int arch = prop.major * 10 + prop.minor;

    auto name = "minilp_solve_" + std::to_string(group_size) + "x" +
                std::to_string(dup_per_rank) + ".cu";

    // Create an instance of nvrtcProgram with the code string.
    std::ifstream kernel_file(resource_path + "/csrc-tmpl/minilp.cu");
    std::string kernel_source =
        std::string((std::istreambuf_iterator<char>(kernel_file)),
                    std::istreambuf_iterator<char>());
    kernel_file.close();

    std::vector<std::string> to_hash{name, kernel_source};

    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog,                 // prog
                                   kernel_source.c_str(), // buffer
                                   name.c_str(),          // name
                                   0,                     // numHeaders
                                   NULL,                  // headers
                                   NULL));                // includeNames

    const std::string mathdx_path = resource_path + "/mathdx/";

    // specify that LTO IR should be generated for LTO operation
    const char *cuda_home = std::getenv("CUDA_HOME");
    if (!cuda_home || !*cuda_home)
      cuda_home = std::getenv("CUDA_PATH");
    if (!cuda_home || !*cuda_home)
      cuda_home = "/usr/local/cuda";
    const auto cuda_include = std::filesystem::path(cuda_home) / "include";

    std::vector<std::string> opts{
        "-dlto",
        "--relocatable-device-code=true",
        "-I" + mathdx_path + "include",
        "-I" + mathdx_path + "external/cutlass/include",
        "-DGROUP_SIZE=" + std::to_string(group_size),
        "-DDUP_PER_RANK=" + std::to_string(dup_per_rank),
        "-DSM_Ver=" + std::to_string(arch * 10),
        "-DBLOCK_DIM=" + std::to_string(block_dim),
        "-I" + cuda_include.string(),
        "-I" + (cuda_include / "cccl").string(),
        "-default-device",
        "-arch=sm_" + std::to_string(arch),
#ifdef USE_NVSHMEM
        "-I" NVSHMEM_DIR "/include",
        "-DUSE_NVSHMEM",
#endif
    };
    std::vector<const char *> opts_c;
    opts_c.reserve(opts.size());
    for (size_t i = 0; i < opts.size(); ++i) {
      to_hash.push_back(opts[i]);
      opts_c.push_back(opts[i].c_str());
    }

    auto hash = vector_string_hash()(to_hash);
    std::ostringstream hash_ss;
    hash_ss << std::setw(16) << std::setfill('0') << std::hex << hash;

    std::string cache_path;
    if (auto env = getenv("LPLB_CACHE_PATH"))
      cache_path = env;
    else
      cache_path = DEFAULT_CACHE_DIR;
    shared_mkdir(cache_path);
    shared_mkdir(cache_path + "/cache");
    shared_mkdir(cache_path + "/tmp");

    auto cache_dir = cache_path + "/cache/" + name + "_" + hash_ss.str();

    if (std::filesystem::exists(cache_dir + "/cubin") &&
        std::filesystem::exists(cache_dir + "/cu") &&
        std::filesystem::exists(cache_dir + "/options")) {
      std::string cached_kernel_source;
      {
        std::ifstream cached_kernel_file(cache_dir + "/cu");
        cached_kernel_source =
            std::string((std::istreambuf_iterator<char>(cached_kernel_file)),
                        std::istreambuf_iterator<char>());
      }
      LPLB_ASSERT(cached_kernel_source == kernel_source &&
                  "Hash collides but source doesn't match");

      // read options by line and compare with opts
      {
        std::ifstream cached_options_file(cache_dir + "/options");
        std::string line;
        unsigned i = 0;
        while (std::getline(cached_options_file, line))
          LPLB_ASSERT(line == opts[i++] &&
                      "Hash collides but options doesn't match");
        LPLB_ASSERT(i == opts.size() &&
                    "Hash collides but options doesn't match");
      }

      std::cout << "Using cached kernel in " << cache_dir << std::endl;
      return cache_dir + "/cubin";
    }

    char hostname[256] = {0};
    LPLB_ASSERT(gethostname(hostname, sizeof(hostname)) == 0);
    char pid[256] = {0};
    sprintf(pid, "%d", getpid());
    auto tmp_cache_dir = cache_path + "/tmp/" + name + "_" + hash_ss.str() +
                         "_" + hostname + "_" + pid;
    std::filesystem::create_directory(tmp_cache_dir);

    std::cout << "Compiling " << name << std::endl;
    nvrtcResult compileResult = nvrtcCompileProgram(prog,          // prog
                                                    opts_c.size(), // numOptions
                                                    opts_c.data()); // options

    // Obtain compilation log from the program.
    size_t logSize;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = new char[logSize];
    NVRTC_CHECK(nvrtcGetProgramLog(prog, log));
    if (compileResult != NVRTC_SUCCESS)
      throw std::runtime_error("NVRTC compile for " + name + " failed:\n" +
                               log);

    // Obtain generated LTO IR from the program.
    size_t LTOIRSize;
    NVRTC_CHECK(nvrtcGetLTOIRSize(prog, &LTOIRSize));
    char *LTOIR = new char[LTOIRSize];
    NVRTC_CHECK(nvrtcGetLTOIR(prog, LTOIR));
    // Destroy the program.
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));

    // Load the generated LTO IR and the LTO IR generated offline
    // and link them together.
    nvJitLinkHandle handle;
    char smbuf[16];
    sprintf(smbuf, "-arch=sm_%d", arch);
    const char *lopts[] = {"-lto", smbuf};
    NVJITLINK_CHECK(handle, nvJitLinkCreate(&handle, 2, lopts));

    // The fatbinary contains LTO IR generated offline using nvcc
    NVJITLINK_CHECK(
        handle,
        nvJitLinkAddFile(handle, NVJITLINK_INPUT_FATBIN,
                         (mathdx_path + "/lib/libcusolverdx.fatbin").c_str()));
#ifdef USE_NVSHMEM
    NVJITLINK_CHECK(handle,
                    nvJitLinkAddFile(handle, NVJITLINK_INPUT_LIBRARY,
                                     NVSHMEM_DIR "/lib/libnvshmem_device.a"));
#endif
    NVJITLINK_CHECK(handle,
                    nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR,
                                     (void *)LTOIR, LTOIRSize, "lto_online"));

    // The call to nvJitLinkComplete causes linker to link together the two
    // LTO IR modules (offline and online), do optimization on the linked LTO
    // IR, and generate cubin from it.
    NVJITLINK_CHECK(handle, nvJitLinkComplete(handle));
    size_t cubinSize;
    NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
    std::vector<char> cubin(cubinSize);
    NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubin(handle, cubin.data()));
    NVJITLINK_CHECK(handle, nvJitLinkDestroy(&handle));

    // save cubin to file
    {
      std::ofstream cubin_file(tmp_cache_dir + "/cubin", std::ios::binary);
      cubin_file.write(cubin.data(), cubinSize);
      std::ofstream cu_file(tmp_cache_dir + "/cu");
      cu_file << kernel_source;
      std::ofstream opts_file(tmp_cache_dir + "/options");
      for (auto &opt : opts)
        opts_file << opt << std::endl;
    }
    // try reduce some contention on FS, but not have to be atomic
    if (!std::filesystem::exists(cache_dir))
      std::filesystem::rename(tmp_cache_dir, cache_dir);
    else
      std::filesystem::remove_all(tmp_cache_dir);
    std::cout << "Saved to " << cache_dir << std::endl;

    return cache_dir + "/cubin";
  }

  void prepare_module(const std::string &cubin) {
    CUDA_CHECK(cudaLibraryLoadFromFile(&module, cubin.c_str(), nullptr, nullptr,
                                       0, nullptr, nullptr, 0));
    CUDA_CHECK(cudaLibraryGetKernel(&kernel_solve, module, "kernel_solve"));
    CUDA_CHECK(cudaLibraryGetKernel(&kernel_map_idx, module, "kernel_map_idx"));
    CUDA_CHECK(
        cudaLibraryGetKernel(&kernel_count_idx, module, "kernel_count_idx"));

    cudaKernel_t get_solve_smem_size;
    CUDA_CHECK(cudaLibraryGetKernel(&get_solve_smem_size, module,
                                    "get_solve_smem_size"));
    // launch get_solve_smem_size(&smem_size) once to get smem size for kernel
    int *h_smem_size;
    CUDA_CHECK(cudaHostAlloc(&h_smem_size, sizeof(int), cudaHostAllocMapped));
    int *d_smem_size;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_smem_size, h_smem_size, 0));
    void *args[] = {&d_smem_size};
    CUDA_CHECK(cudaLaunchKernel(get_solve_smem_size, 1, 1, args, 0, nullptr));
    CUDA_CHECK(cudaStreamSynchronize(0));
    smem_size = *h_smem_size;
    CUDA_CHECK(cudaKernelSetAttributeForDevice(
        kernel_solve, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size,
        c10::cuda::current_device()));
  }

  bool init_comm_done = false;

#ifdef USE_NVSHMEM
  void init_comm(const c10::Device &device, bool nvshmem_multiplane,
                 bool do_nvshmem_init) {
    LPLB_ASSERT(pg);
    if (init_comm_done)
      return;
    init_comm_done = true;

    if (do_nvshmem_init) {
      nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
      nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
      nvshmemx_get_uniqueid(&id);
      nvshmemx_set_attr_uniqueid_args(0, 1, &id, &attr);
      nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    }

    sync_current_to_module(module, "nvshmemi_device_state_d");
    sync_current_to_module(module, "nvshmemi_device_lib_version_d");
    sync_current_to_module(module, "nvshmemi_ibgda_device_state_d");

    int n_pes = n_group * group_size;
    if (nvshmem_multiplane) {
      n_nodes = nvshmem_n_pes();
      cpu_rdma_team = NVSHMEM_TEAM_WORLD;
    } else {
      LPLB_ASSERT(deep_ep::internode::cpu_rdma_team != NVSHMEM_TEAM_INVALID);
      n_nodes = nvshmem_team_n_pes(deep_ep::internode::cpu_rdma_team);
      cpu_rdma_team = deep_ep::internode::cpu_rdma_team;
    }
    node_size = n_pes / n_nodes;
    self_node = pg->getRank() / node_size;
    self_device = pg->getRank() % node_size;

    int n_logical_experts =
        (n_local_experts - dup_per_rank * n_combined_experts) * n_pes;
    workload_buf_internode = (float *)nvshmem_align(
        128, n_logical_experts * n_nodes * sizeof(float));
    workload_sig_internode = (uint64_t *)nvshmem_calloc(2, sizeof(uint64_t));

    float *workload_buf_intranode_local;
    CUDA_CHECK(cudaMalloc(&workload_buf_intranode_local,
                          n_logical_experts * n_nodes * sizeof(float)));
    uint32_t *workload_sig_intranode_local;
    CUDA_CHECK(cudaMalloc(&workload_sig_intranode_local, 2 * sizeof(uint32_t)));
    CUDA_CHECK(
        cudaMemset(workload_sig_intranode_local, 0, 2 * sizeof(uint32_t)));

    cudaIpcMemHandle_t buf_handle, sig_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&buf_handle, workload_buf_intranode_local));
    CUDA_CHECK(cudaIpcGetMemHandle(&sig_handle, workload_sig_intranode_local));

    auto local_handles_tensor =
        at::empty({2 * sizeof(cudaIpcMemHandle_t)}, torch::kByte);
    cudaIpcMemHandle_t *local_handles = reinterpret_cast<cudaIpcMemHandle_t *>(
        local_handles_tensor.data_ptr<uint8_t>());
    local_handles[0] = buf_handle;
    local_handles[1] = sig_handle;
    local_handles_tensor = local_handles_tensor.to(device);
    auto global_handles_tensor =
        at::empty({pg->getSize(), 2 * sizeof(cudaIpcMemHandle_t)},
                  torch::dtype(torch::kByte).device(device));
    pg->_allgather_base(global_handles_tensor, local_handles_tensor)->wait();
    global_handles_tensor = global_handles_tensor.to(at::kCPU);
    cudaIpcMemHandle_t *global_handles = reinterpret_cast<cudaIpcMemHandle_t *>(
        global_handles_tensor.data_ptr<uint8_t>());

    workload_buf_intranode_cpu.resize(node_size);
    workload_sig_intranode_cpu.resize(node_size);
    for (int i = self_node * node_size; i < (self_node + 1) * node_size; i++) {
      if (i == pg->getRank()) {
        workload_buf_intranode_cpu[i % node_size] =
            workload_buf_intranode_local;
        workload_sig_intranode_cpu[i % node_size] =
            workload_sig_intranode_local;
      } else {
        CUDA_CHECK(cudaIpcOpenMemHandle(
            (void **)&workload_buf_intranode_cpu[i % node_size],
            global_handles[i * 2], cudaIpcMemLazyEnablePeerAccess));
        CUDA_CHECK(cudaIpcOpenMemHandle(
            (void **)&workload_sig_intranode_cpu[i % node_size],
            global_handles[i * 2 + 1], cudaIpcMemLazyEnablePeerAccess));
      }
    }
    CUDA_CHECK(
        cudaMalloc(&workload_buf_intranode, node_size * sizeof(float *)));
    CUDA_CHECK(cudaMemcpy(workload_buf_intranode,
                          workload_buf_intranode_cpu.data(),
                          node_size * sizeof(float *), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMalloc(&workload_sig_intranode, node_size * sizeof(uint64_t *)));
    CUDA_CHECK(
        cudaMemcpy(workload_sig_intranode, workload_sig_intranode_cpu.data(),
                   node_size * sizeof(uint64_t *), cudaMemcpyHostToDevice));

    if (pg->getRank() == 0) {
      std::cout << "LPLB communicator initialized: " << std::endl;
      std::cout << "  n_nodes: " << n_nodes << std::endl;
      std::cout << "  node_size: " << node_size << std::endl;
      std::cout << "  nvshmem_multiplane: " << nvshmem_multiplane << std::endl;
    }
  }
#endif

  compiled_solver(const std::string &resource_path, int n_group, int group_size,
                  int dup_per_rank, int block_dim, int n_local_experts,
                  int n_combined_experts,
                  c10::intrusive_ptr<c10d::ProcessGroup> pg)
      : n_group(n_group), group_size(group_size), dup_per_rank(dup_per_rank),
        block_dim(block_dim),
        n_experts((n_local_experts - dup_per_rank * n_combined_experts) *
                  group_size * n_group),
        n_local_experts(n_local_experts),
        n_combined_experts(n_combined_experts), pg(pg) {
    prepare_module(compile_cubin(resource_path));
  }

  ~compiled_solver() {
#ifdef USE_NVSHMEM
    if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
      cudaFree(workload_buf_intranode);
      cudaFree(workload_sig_intranode);
      for (int i = 0; i < n_nodes; i++)
        if (i != pg->getRank() % node_size) {
          cudaIpcCloseMemHandle(workload_buf_intranode_cpu[i]);
          cudaIpcCloseMemHandle(workload_sig_intranode_cpu[i]);
        }
      pg->barrier();
      nvshmem_free(workload_buf_intranode_cpu[pg->getRank() % node_size]);
      nvshmem_free(workload_sig_intranode_cpu[pg->getRank() % node_size]);
      nvshmem_free(workload_buf_internode);
      nvshmem_free(workload_sig_internode);
    }
#endif
  }

  std::pair<at::Tensor, at::Tensor> solve(at::Tensor local_workload,
                                          at::Tensor r2o, at::Tensor phy2log,
                                          at::Tensor avail_num) {
    LPLB_ASSERT(local_workload.dim() == 3);
    int n_group = local_workload.size(0);
    LPLB_ASSERT(group_size == local_workload.size(1));
    int n_logical_experts_per_rank = local_workload.size(2);
    int n_pes = n_group * group_size;
    LPLB_ASSERT(local_workload.dtype() == torch::kInt32);
    LPLB_ASSERT(local_workload.is_contiguous());

    void *p_workload = local_workload.data_ptr();

    auto global_workload = at::empty_like(
        local_workload, local_workload.options().dtype(torch::kFloat32));
    void *p_global_workload = global_workload.data_ptr();

    LPLB_ASSERT(r2o.dim() == 2);
    LPLB_ASSERT(r2o.size(0) == group_size);
    LPLB_ASSERT(r2o.size(1) == dup_per_rank);
    LPLB_ASSERT(r2o.dtype() == torch::kInt32);
    void *p_r2o = r2o.contiguous().data_ptr();

    LPLB_ASSERT(phy2log.dim() == 1);
    LPLB_ASSERT(phy2log.size(0) % n_pes == 0);
    LPLB_ASSERT(phy2log.dtype() == torch::kInt32);
    int n_physical_experts_per_rank = phy2log.size(0) / n_pes;
    LPLB_ASSERT((n_physical_experts_per_rank - n_logical_experts_per_rank) %
                    dup_per_rank ==
                0);
    void *p_phy2log = phy2log.contiguous().data_ptr();
    int n_experts_per_var =
        (n_physical_experts_per_rank - n_logical_experts_per_rank) /
        dup_per_rank;
    int n_experts_fixed =
        n_logical_experts_per_rank - dup_per_rank * n_experts_per_var;

    LPLB_ASSERT(avail_num.numel() == 1);
    LPLB_ASSERT(avail_num.dtype() == torch::kInt32);
    void *p_avail_num = avail_num.contiguous().data_ptr();

    at::Tensor result =
        at::empty({n_group, group_size, dup_per_rank},
                  local_workload.options().dtype(torch::kFloat32));
    void *p_result = result.data_ptr();

    void *args[] = {&p_workload,
                    &p_global_workload,
                    &p_r2o,
                    &p_phy2log,
                    &n_experts_per_var,
                    &n_experts_fixed,
                    &p_avail_num,
                    &p_result,
#ifdef USE_NVSHMEM
                    &workload_buf_internode,
                    &workload_sig_internode,
                    &workload_buf_intranode,
                    &workload_sig_intranode,
                    &cpu_rdma_team,
                    &self_device,
                    &node_size
#endif
    };
    cudaStream_t cuda_stream = c10::cuda::getCurrentCUDAStream().stream();
#ifdef USE_NVSHMEM
    if (cpu_rdma_team == NVSHMEM_TEAM_INVALID)
#endif
      CUDA_CHECK(cudaLaunchCooperativeKernel(
          kernel_solve,
          /* gridDims */ {(unsigned)n_group, 1, 1},
          /* blockDims */ {(unsigned)block_dim, 1, 1},
          /* kernelParams */ args,
          /* sharedMemBytes */ smem_size,
          /* stream */ cuda_stream));
#ifdef USE_NVSHMEM
    else
      LPLB_ASSERT(0 == nvshmemx_collective_launch(
                           kernel_solve,
                           /* gridDims */ {(unsigned)n_group, 1, 1},
                           /* blockDims */ {(unsigned)block_dim, 1, 1},
                           /* kernelParams */ args,
                           /* sharedMemBytes */ smem_size,
                           /* stream */ cuda_stream));
#endif

    return {result, global_workload};
  }

  std::pair<at::Tensor, at::Tensor> count_idx(at::Tensor idx, int n_sms,
                                              int block_dim) {
    LPLB_ASSERT(idx.dtype() == torch::kInt64);
    LPLB_ASSERT(idx.device().is_cuda());
    LPLB_ASSERT(idx.is_contiguous());

    auto out =
        at::empty({n_sms, n_experts}, idx.options().dtype(torch::kInt32));

    void *idx_ptr = idx.data_ptr();
    int n_elements = idx.numel();
    void *out_ptr = out.data_ptr();

    void *args[] = {&idx_ptr, &n_elements, &n_experts, &out_ptr};
    int count_idx_smem_size = (16 + n_experts) * sizeof(int);
    cudaStream_t cuda_stream = c10::cuda::getCurrentCUDAStream().stream();
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        kernel_count_idx, {(unsigned)n_sms, 1, 1}, {(unsigned)block_dim, 1, 1},
        args, count_idx_smem_size, cuda_stream));
    return {out.index({-1}), out};
  }

  at::Tensor map_idx(at::Tensor mapping_idx, at::Tensor o_weight,
                     at::Tensor local_workload_split_by_sm, at::Tensor o2r,
                     at::Tensor phy2log, int n_sms, int block_dim) {
    LPLB_ASSERT(mapping_idx.dtype() == torch::kInt64);
    LPLB_ASSERT(mapping_idx.device().is_cuda());
    LPLB_ASSERT(mapping_idx.is_contiguous());

    LPLB_ASSERT(o_weight.dtype() == torch::kFloat32);
    LPLB_ASSERT(o_weight.device().is_cuda());
    LPLB_ASSERT(o_weight.is_contiguous());
    LPLB_ASSERT(o_weight.dim() == 3);
    LPLB_ASSERT(o_weight.size(0) == n_group);
    LPLB_ASSERT(o_weight.size(1) == group_size);
    LPLB_ASSERT(o_weight.size(2) == dup_per_rank);

    LPLB_ASSERT(o2r.dtype() == torch::kInt32);
    LPLB_ASSERT(o2r.device().is_cuda());
    LPLB_ASSERT(o2r.is_contiguous());
    LPLB_ASSERT(o2r.dim() == 2);
    LPLB_ASSERT(o2r.size(0) == group_size);
    LPLB_ASSERT(o2r.size(1) == dup_per_rank);

    LPLB_ASSERT(phy2log.dtype() == torch::kInt32);
    LPLB_ASSERT(phy2log.device().is_cuda());
    LPLB_ASSERT(phy2log.is_contiguous());
    LPLB_ASSERT(phy2log.dim() == 1);
    LPLB_ASSERT(phy2log.size(0) == group_size * n_group * n_local_experts);

    at::Tensor mapping_idx_out = at::empty_like(mapping_idx);
    int n_elements = mapping_idx.numel();

    void *mapping_idx_ptr = mapping_idx.data_ptr();
    void *o_weight_ptr = o_weight.data_ptr();
    void *local_workload_split_by_sm_ptr =
        local_workload_split_by_sm.data_ptr();
    void *o2r_ptr = o2r.data_ptr();
    void *phy2log_ptr = phy2log.data_ptr();
    void *mapping_idx_out_ptr = mapping_idx_out.data_ptr();

    void *args[] = {&mapping_idx_ptr,
                    &o_weight_ptr,
                    &local_workload_split_by_sm_ptr,
                    &o2r_ptr,
                    &phy2log_ptr,
                    &n_elements,
                    &n_group,
                    &n_combined_experts,
                    &n_local_experts,
                    &mapping_idx_out_ptr};
    int n_logical_experts =
        n_group * group_size *
        (n_local_experts - dup_per_rank * n_combined_experts);
    int map_idx_smem_size = 5 * n_logical_experts * sizeof(int);
    cudaStream_t cuda_stream = c10::cuda::getCurrentCUDAStream().stream();
    CUDA_CHECK(cudaLaunchKernel(kernel_map_idx, {(unsigned)n_sms, 1, 1},
                                {(unsigned)block_dim, 1, 1}, args,
                                map_idx_smem_size, cuda_stream));
    return mapping_idx_out;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  py::class_<compiled_solver>(m, "CompiledSolver")
      .def(py::init<const std::string &, int, int, int, int, int, int,
                    c10::intrusive_ptr<c10d::ProcessGroup>>())
#ifdef USE_NVSHMEM
      .def("init_comm", &compiled_solver::init_comm, "device"_a,
           "nvshmem_multiplane"_a, "do_nvshmem_init"_a)
#endif
      .def("solve", &compiled_solver::solve, "local_workload"_a, "r2o"_a,
           "phy2log"_a, "avail_num"_a)
      .def("count_idx", &compiled_solver::count_idx, "idx"_a, "n_sms"_a,
           "block_dim"_a)
      .def("map_idx", &compiled_solver::map_idx, "mapping_idx"_a, "o_weight"_a,
           "local_workload_split_by_sm"_a, "o2r"_a, "phy2log"_a, "n_sms"_a,
           "block_dim"_a);
}
