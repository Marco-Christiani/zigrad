#ifndef __CUTENSOR_UTILS_ZIG__
#define __CUTENSOR_UTILS_ZIG__
#include "decls.h"
#include "cuda_helpers.cu"
#include <cutensor/types.h>

#include <array>
#include <algorithm>
#include <unordered_map>
#include <initializer_list>
#include <typeinfo>
#include <variant>

static const u32 cutensor_alignment = 128; // Alignment of the global-memory device poi32ers (bytes)

cutensorOperator_t cutensor_op_type(BINARY_OP op) {
  switch (op) {
    case BINARY_OP::ADD: return cutensorOperator_t::CUTENSOR_OP_ADD;
    case BINARY_OP::MIN: return cutensorOperator_t::CUTENSOR_OP_MIN;
    case BINARY_OP::MAX: return cutensorOperator_t::CUTENSOR_OP_MAX;
    case BINARY_OP::MUL: return cutensorOperator_t::CUTENSOR_OP_MUL;
    default: 
      SYSTEM_EXIT("Invalid reduce operation");
      return {}; // silence warning
  }
}

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

// Taken from: https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
template <class T>
std::size_t __seq_hash(const T* key, std::size_t n) {
  std::size_t seed = n;
  for (std::size_t i = 0; i < n; ++i) {
    T x = key[i];
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

template<class T, std::size_t N = 8>
struct BoundedArray {
  T data[N];
  std::size_t size = 0;

  using iterator = typename std::array<T, N>::iterator;
  using const_iterator = typename std::array<T, N>::const_iterator;

  BoundedArray() = default;

  template<class U>
  BoundedArray(const U* vals, std::size_t n) : BoundedArray() {
    this->append(vals, n);
  }

  template<class U>
  BoundedArray(const U* vals, std::size_t n, bool reverse) : BoundedArray(vals, n) {
    if (reverse) this->reverse();
  }

  BoundedArray(std::initializer_list<std::size_t> vals) : BoundedArray() {
    this->append(vals);
  }

  std::size_t hash() const {
    return __seq_hash(this->ptr(), this->size);
  }

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

  const T* ptr() const {
    return &this->data[0];
  }

  void reverse() {
    std::reverse(this->begin(), this->end());
  }

  void sort() {
    std::sort(this->begin(), this->end());
  }

  template<class U>
  void append(const U* vals, len_t n) {
    for (len_t i = 0; i < n; ++i) {
      this->append(vals[i]);
    }
  }

  template<class ... Ts>
  void append(std::initializer_list<Ts...> vals) {
    for (auto x: vals) {
      this->append(x);
    }
  }

  template <class U>
  void append(U val) {
    CHECK_INVARIANT((1 + this->size <= N), "Bounded Array Overflow");
    this->data[this->size] = static_cast<T>(val);
    ++this->size;
  }
};

/////////////////////////////////////////////////////////
// Cutensor Data Component Cleanup //////////////////////

void destroy(cutensorPlan_t ptr) {
  CUTENSOR_ASSERT(cutensorDestroyPlan(ptr));
}
void destroy(cutensorPlanPreference_t ptr) {
  CUTENSOR_ASSERT(cutensorDestroyPlanPreference(ptr));
}
void destroy(cutensorTensorDescriptor_t ptr) {
  CUTENSOR_ASSERT(cutensorDestroyTensorDescriptor(ptr));
}
void destroy(cutensorOperationDescriptor_t ptr) {
  CUTENSOR_ASSERT(cutensorDestroyOperationDescriptor(ptr));
}

template<class T>
struct ManagedComponent {
  T ptr;
  ManagedComponent(T _ptr = nullptr) : ptr{_ptr} {}
  ManagedComponent(ManagedComponent const&) = delete;
  ManagedComponent& operator=(const ManagedComponent&) = delete;
  ManagedComponent(ManagedComponent&& other) noexcept {
    this->ptr = std::exchange(other.ptr, nullptr);
  };
  ManagedComponent& operator=(ManagedComponent&& other) noexcept {
    this->ptr = std::exchange(other.ptr, nullptr);
  };
  ~ManagedComponent() {
    if (this->ptr) {
      destroy(this->ptr);
    }
  }
};

typedef ManagedComponent<cutensorPlan_t> ManagedPlan;
typedef ManagedComponent<cutensorPlanPreference_t> ManagedPlanPreference;
typedef ManagedComponent<cutensorTensorDescriptor_t> ManagedTensorDescriptor;
typedef ManagedComponent<cutensorOperationDescriptor_t> ManagedOperationDescriptor;

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////

struct ReducePlan {
  ManagedPlan plan;
  ManagedPlanPreference plan_pref;
  ManagedOperationDescriptor op_desc;
  ManagedTensorDescriptor x_desc;
  ManagedTensorDescriptor y_desc;
  len_t scratch_len;
};

struct PermutatePlan {
  ManagedPlan plan;
  ManagedPlanPreference plan_pref;
  ManagedOperationDescriptor op_desc;
  ManagedTensorDescriptor x_desc;
  ManagedTensorDescriptor y_desc;
  len_t scratch_len;
};

struct ContractionPlan {
  ManagedPlan plan;
  ManagedPlanPreference plan_pref;
  ManagedOperationDescriptor op_desc;
  ManagedTensorDescriptor x_desc;
  ManagedTensorDescriptor y_desc;
  ManagedTensorDescriptor z_desc;
  len_t scratch_len;
};

struct BinaryPlan {
  ManagedPlan plan;
  ManagedPlanPreference plan_pref;
  ManagedOperationDescriptor op_desc;
  ManagedTensorDescriptor x_desc;
  ManagedTensorDescriptor y_desc;
  len_t scratch_len;
};

typedef std::variant<
  BinaryPlan,
  ContractionPlan,
  PermutatePlan,
  ReducePlan
> PlanVariant;

typedef std::shared_ptr<PlanVariant> MapPtr;

class PlanManager {
private:
  
  struct MapKey {
    BoundedArray<len_t, 8> data;
  
    struct Hash{
      std::size_t operator()(MapKey key) const { return key.data.hash(); }
    };
    bool operator==(const MapKey& b) const {
      return std::equal(this->data.begin(), this->data.end(), b.data.begin(), b.data.end());
    }
  };

  typedef std::unordered_map<MapKey, MapPtr, MapKey::Hash> MapType;

  MapType map{};

public:

  PlanManager() = default;
  PlanManager(const PlanManager&) = delete;
  PlanManager(PlanManager&&) noexcept = default;

  template<class T>
  MapKey make_key(cutensorDataType_t dtype, std::initializer_list<std::size_t> vals) {
    MapKey key;
    key.data.append(vals);
    key.data.append(dtype);
    key.data.append(typeid(T).hash_code());
    return key;
  }

  MapType::iterator begin() {
    return this->map.begin();
  }
  MapType::iterator end() {
    return this->map.end();
  }

  template<class T>
  T* find(MapKey const& k) {
    auto itr = this->map.find(k);
    if (itr == this->map.end()) {
      return nullptr;
    }
    return &std::get<T>(*itr->second);
  }

  template<class T>
  void insert(MapKey key, T&& val) {
    this->map.insert(std::make_pair(key, std::make_shared<PlanVariant>(std::move(val))));
  }
};


struct CutensorBackend {
    PlanManager manager;
    cudaStream_t stream{nullptr};
    cutensorHandle_t handle{nullptr};

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
};
  
// binary is used for broadcasting reverses
// not intended to be used with 
//CutensorPlanWrapper get_binary_plan(
//  CutensorWrapper wrapper,
//  dtype id,
//  const len_t* src_dims,
//  const u8* src_syms,
//  len_t src_dims_len,
//  const len_t* dst_dims,
//  const u8* dst_syms,
//  len_t dst_dims_len,
//  BINARY_OP op
//) {
//  CHECK_INVARIANT(0 < src_dims_len, "Zero length dimensions passed to permutate");
//  CHECK_INVARIANT(src_dims_len <= dst_dims_len, "Source dimensions length greater than destination");
//  
//  const auto data_type = cutensor_data_type(id);
//  const auto op_type = cutensor_op_type(op);
//
//  auto ct = CutensorBackend::unwrap(wrapper);
//
//  auto key = ct->manager.make_key<BinaryPlan>(
//    data_type,
//    {
//      __seq_hash(src_dims, src_dims_len),
//      __seq_hash(src_syms, src_dims_len),
//      __seq_hash(dst_dims, src_dims_len),
//      __seq_hash(dst_syms, src_dims_len),
//      static_cast<std::size_t>(op),
//    }
//  );
//
//  if (auto entry = ct->manager.find<PermutatePlan>(key); entry)
//    return { .ptr = entry->plan.ptr, .scratch_len = entry->scratch_len };
//
//  BoundedArray<i64> a_dims(src_dims, src_dims_len, true);
//  BoundedArray<i32> a_syms(src_syms, src_dims_len, true);
//  BoundedArray<i64> b_dims(dst_dims, src_dims_len, true);
//  BoundedArray<i32> b_syms(dst_syms, src_dims_len, true);
//  
//  cutensorTensorDescriptor_t x_desc;
//  CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
//              ct->handle,
//              &x_desc,
//              a_dims.size,
//              a_dims.data,
//              NULL,/*stride*/
//              data_type, cutensor_alignment));
//  
//  cutensorTensorDescriptor_t y_desc;
//  CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
//              ct->handle,
//              &y_desc,
//              b_dims.size,
//              b_dims.data,
//              NULL,/*stride*/
//              data_type, cutensor_alignment));
//  
//  cutensorOperationDescriptor_t op_desc;
//  CUTENSOR_ASSERT(cutensorCreateElementwiseBinary(
//              ct->handle, &op_desc,
//              x_desc, a_syms.data, CUTENSOR_OP_IDENTITY,
//              y_desc, b_syms.data, CUTENSOR_OP_IDENTITY,
//              y_desc, b_syms.data,
//              op_type, cutensor_compute_type(id)));
//
//  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
//  
//  cutensorPlanPreference_t plan_pref;
//  CUTENSOR_ASSERT(cutensorCreatePlanPreference(
//              ct->handle,
//              &plan_pref,
//              algo,
//              CUTENSOR_JIT_MODE_NONE));
//  
//  
//  cutensorPlan_t plan;
//  CUTENSOR_ASSERT(cutensorCreatePlan(
//              ct->handle,
//              &plan,
//              op_desc,
//              plan_pref,
//              0 /* not required */));
//  
//  ct->manager.insert(
//    key,
//    BinaryPlan{
//      .plan = plan,
//      .plan_pref = plan_pref,
//      .x_desc = x_desc,
//      .y_desc = y_desc,
//    }
//  );
//
//  return { .ptr = plan };
//}

// binary is used for broadcasting reverses
// not intended to be used with 
//CutensorPlanWrapper get_reduce_bwds_trinary(
//  CutensorWrapper wrapper,
//  dtype id,
//  const len_t* src_dims,
//  const u8* src_syms,
//  len_t src_dims_len,
//  const len_t* dst_dims,
//  const u8* dst_syms,
//  len_t dst_dims_len,
//  BINARY_OP op
//) {
//  CHECK_INVARIANT(0 < src_dims_len, "Zero length dimensions passed to permutate");
//  CHECK_INVARIANT(src_dims_len <= dst_dims_len, "Source dimensions length greater than destination");
//
//  auto ct = CutensorBackend::unwrap(wrapper);
//  
//  const auto data_type = cutensor_data_type(id);
//  const auto op_type = cutensor_op_type(op);
//
//  auto key = ct->manager.make_key<BinaryPlan>(
//    data_type,
//    {
//      __seq_hash(src_dims, src_dims_len),
//      __seq_hash(src_syms, src_dims_len),
//      __seq_hash(dst_dims, src_dims_len),
//      __seq_hash(dst_syms, src_dims_len),
//      static_cast<std::size_t>(op),
//    }
//  );
//
//  if (auto entry = ct->manager.find<PermutatePlan>(key); entry) {
//    return entry->plan.ptr;
//  }
//
//  BoundedArray<i64> a_dims(src_dims, src_dims_len, true);
//  BoundedArray<i32> a_syms(src_syms, src_dims_len, true);
//  BoundedArray<i64> b_dims(dst_dims, src_dims_len, true);
//  BoundedArray<i32> b_syms(dst_syms, src_dims_len, true);
//  
//  cutensorTensorDescriptor_t x_desc;
//  CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
//              ct->handle,
//              &x_desc,
//              a_dims.size,
//              a_dims.data,
//              NULL,/*stride*/
//              data_type, cutensor_alignment));
//  
//  cutensorTensorDescriptor_t y_desc;
//  CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
//              ct->handle,
//              &y_desc,
//              b_dims.size,
//              b_dims.data,
//              NULL,/*stride*/
//              data_type, cutensor_alignment));
//  
//  cutensorOperationDescriptor_t op_desc;
//  CUTENSOR_ASSERT(cutensorCreateElementwiseBinary(
//              ct->handle, &op_desc,
//              x_desc, a_syms.data, CUTENSOR_OP_IDENTITY,
//              y_desc, b_syms.data, CUTENSOR_OP_IDENTITY,
//              y_desc, b_syms.data,
//              op_type, cutensor_compute_type(id)));
//
//  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
//  
//  cutensorPlanPreference_t plan_pref;
//  CUTENSOR_ASSERT(cutensorCreatePlanPreference(
//              ct->handle,
//              &plan_pref,
//              algo,
//              CUTENSOR_JIT_MODE_NONE));
//  
//  
//  cutensorPlan_t plan;
//  CUTENSOR_ASSERT(cutensorCreatePlan(
//              ct->handle,
//              &plan,
//              op_desc,
//              plan_pref,
//              0 /* not required */));
//  
//  ct->manager.insert(
//    key,
//    BinaryPlan{
//      .plan = plan,
//      .plan_pref = plan_pref,
//      .x_desc = x_desc,
//      .y_desc = y_desc,
//    }
//  );
//
//  return { .ptr = plan, .scratch_len = scratch_len };
//}
  
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

#endif
