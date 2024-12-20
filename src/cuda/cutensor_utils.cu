#ifndef __CUTENSOR_UTILS_ZIG__
#define __CUTENSOR_UTILS_ZIG__

#include <array>
#include <algorithm>
#include <cutensor/types.h>
#include <unordered_map>

#include "decls.h"
#include "cuda_helpers.cu"

cutensorDataType_t cutensor_data_type(dtype id) {
  switch (id) {
    case SINGLE: return cutensorDataType_t::CUTENSOR_R_32F;
    case DOUBLE: return cutensorDataType_t::CUTENSOR_R_64F;
    default: 
      SYSTEM_EXIT("Invalid data type");
      return {}; // silence warning
  }
}
  
cutensorComputeDescriptor_t cutensor_compute_type(dtype id) {
  switch (id) {
    case SINGLE: return CUTENSOR_COMPUTE_DESC_32F;
    case DOUBLE: return CUTENSOR_COMPUTE_DESC_64F;
    default: 
      SYSTEM_EXIT("Invalid data type");
      return {}; // silence warning
  }
}

template<class T>
struct BoundedArray {
  constexpr static len_t MAX_SIZE = 16;
  T data[MAX_SIZE];
  uint32_t size = 0;

  using iterator = typename std::array<T, MAX_SIZE>::iterator;
  using const_iterator = typename std::array<T, MAX_SIZE>::const_iterator;

  iterator begin() {
    return iterator{ &this->data[0] };
  }
  iterator end() {
    return this->begin() + this->size;
  }

  const_iterator begin() const {
    return const_iterator{ &this->data[0] };
  }
  const_iterator end() const {
    return this->begin() + this->size;
  }

  T* ptr() {
    return &this->data[0];
  }

  void reverse() {
    std::reverse(this->begin(), this->end());
  }

  void sort() {
    std::sort(this->begin(), this->end());
  }

  void append(const T* vals, len_t n) {
    CHECK_INVARIANT((n + this->size <= MAX_SIZE), "Bounded Array Overflow");

    for (len_t i = 0; i < n; ++i) {
      this->data[this->size] = vals[i];
      ++this->size;
    }
  }

  void append(T val) {
    CHECK_INVARIANT((1 + this->size <= MAX_SIZE), "Bounded Array Overflow");
    this->data[this->size] = val;
    ++this->size;
  }
};

struct CutensorMapKey {
  cutensorDataType_t data_type;
  BoundedArray<len_t> dims;
  
  bool operator==(const CutensorMapKey& b) const {
    return (this->data_type == b.data_type) && std::equal(this->dims.begin(), this->dims.end(), b.dims.begin(), b.dims.end());
  }

  struct Hash {
    // Taken from: https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
    std::size_t operator()(CutensorMapKey const& key) const {
      std::size_t seed = key.dims.size;
      for (len_t x: key.dims) {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };
};


class CutensorBackend {

  public:
    using i64 = int64_t;
    using CutensorPlanMap = std::unordered_map<CutensorMapKey, cutensorPlan_t, CutensorMapKey::Hash>;

    cudaStream_t stream{nullptr};
    cutensorHandle_t handle{nullptr};
    CutensorPlanMap reduce_sum_map{};
    CutensorPlanMap reduce_max_map{};

    static CutensorBackend* unwrap(CutensorWrapper wrapper) {
      return static_cast<CutensorBackend*>(wrapper.ptr);
    }
    static CutensorWrapper wrap(CutensorBackend* backend) {
      return { .ptr = backend };
    }

    CutensorBackend(void* _stream) {
      this->stream = static_cast<cudaStream_t>(_stream);
      CUTENSOR_ASSERT(cutensorCreate(&this->handle));
    }

    ~CutensorBackend() {
      CUTENSOR_ASSERT(cutensorDestroy(this->handle));
    }

    cutensorPlan_t get_sum_over_dim_plan(
      dtype id,
      const len_t* dims,
      len_t dim_len,
      len_t rdx_dim,
      void** scratch,
      len_t* scratch_len
    ) {
      return this->get_reduce_plan(id, dims, dim_len, &rdx_dim, 1, scratch, scratch_len, this->reduce_sum_map, CUTENSOR_OP_ADD);
    }

    cutensorPlan_t get_max_over_dim_plan(
      dtype id,
      const len_t* dims,
      len_t dim_len,
      len_t rdx_dim,
      void** scratch,
      len_t* scratch_len
    ) {
      return this->get_reduce_plan(id, dims, dim_len, &rdx_dim, 1, scratch, scratch_len, this->reduce_max_map, CUTENSOR_OP_MAX);
    }

  private:
  
    cutensorPlan_t get_reduce_plan(
      dtype id,
      const len_t* src_dims,
      len_t src_dims_len,
      const len_t* rdx_idxs,
      len_t rdx_idxs_len,
      void** scratch,
      len_t* scratch_len,
      CutensorPlanMap& plan_map,
      cutensorOperator_t redux
    ) {
      CHECK_INVARIANT(rdx_idxs_len <= src_dims_len, "Reduction dimension out of bounds");
      CHECK_INVARIANT(0 < src_dims_len, "Zero length dimensions passed to reduce");
      CHECK_INVARIANT(0 < rdx_idxs_len, "Zero length dimensions passed to reduce");
  
      const auto data_type = cutensor_data_type(id);
  
      CutensorMapKey key;
      key.dims.append(src_dims, src_dims_len);
      key.dims.append(rdx_idxs, rdx_idxs_len);
      key.data_type = data_type;
  
      const auto itr = plan_map.find(key);
  
      if (itr != plan_map.end()) {
        return itr->second;
      }
      
      BoundedArray<i64> a_dims;
      BoundedArray<i64> b_dims;
      BoundedArray<int> a_syms;
      BoundedArray<int> b_syms;
  
      BoundedArray<len_t> r_idxs;
      r_idxs.append(rdx_idxs, rdx_idxs_len);
      r_idxs.sort();
  
      {
          int sym = 'i';
          len_t r_pos = 0;
  
          for (len_t i = 0; i < src_dims_len; ++i, ++sym) {
  
            a_dims.append(static_cast<i64>(src_dims[i]));
            a_syms.append(sym);
  
            // skip every index indicated for reduction...
            if (r_pos < r_idxs.size && i == r_idxs.data[r_pos]) {
              r_pos += 1;
              continue;
            };
  
            b_dims.append(static_cast<i64>(src_dims[i]));
            b_syms.append(sym);
          }
  
          a_dims.append(1);
          b_dims.append(1);
          a_syms.append('z');
          b_syms.append('z');
          a_dims.reverse();
          b_dims.reverse();
          a_syms.reverse();
          b_syms.reverse();
      }
  
  
      /**********************
       * Create Tensor Descriptors
       **********************/
      const uint32_t alignment = 128; // Alignment of the global-memory device pointers (bytes)
  
      cutensorTensorDescriptor_t descA;
      CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
                  this->handle,
                  &descA,
                  a_dims.size,
                  a_dims.data,
                  NULL,/*stride*/
                  data_type, alignment));
  
      cutensorTensorDescriptor_t descB;
      CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
                  this->handle,
                  &descB,
                  b_dims.size,
                  b_dims.data,
                  NULL,/*stride*/
                  data_type, alignment));
  
      /*******************************
       * Create Contraction Descriptor
       *******************************/
  
      cutensorOperationDescriptor_t desc;
      CUTENSOR_ASSERT(cutensorCreateReduction(
                  this->handle, &desc,
                  descA, a_syms.data, CUTENSOR_OP_IDENTITY,
                  descB, b_syms.data, CUTENSOR_OP_IDENTITY,
                  descB, b_syms.data, redux,
                  cutensor_compute_type(id)));
  
      /**************************
       * Set the algorithm to use
       ***************************/
  
      const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
  
      cutensorPlanPreference_t plan_preference;
      CUTENSOR_ASSERT(cutensorCreatePlanPreference(
                  this->handle,
                  &plan_preference,
                  algo,
                  CUTENSOR_JIT_MODE_NONE));
  
      /**********************
       * Query workspace estimate
       **********************/
  
      CUTENSOR_ASSERT(cutensorEstimateWorkspaceSize(
                  this->handle,
                  desc,
                  plan_preference,
                  CUTENSOR_WORKSPACE_DEFAULT,
                  scratch_len));
  
      cutensorPlan_t plan;
      CUTENSOR_ASSERT(cutensorCreatePlan(
                  this->handle,
                  &plan,
                  desc,
                  plan_preference,
                  *scratch_len));
  
      plan_map.insert(std::make_pair(key, plan));
  
      return plan;
    } 
};

#endif
  
