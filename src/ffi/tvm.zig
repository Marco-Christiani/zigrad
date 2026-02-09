//! TVM C API FFI bindings.
//!
//! Provides bindings to TVM's C API (libtvm_ffi.so, libtvm.so)

const build_options = @import("build_options");

// Conditional import of TVM C API
const tvm_enabled = build_options.enable_tvm;

// When TVM is enabled, import the real C API
// When disabled, provide stub types
const Impl = if (tvm_enabled)
    @cImport({
        @cInclude("tvm/ffi/c_api.h");
    })
else
    struct {
        // Stub types for when TVM is disabled
        pub const TVMFFIAny = extern struct {
            _dummy: u8 = 0,
        };
        pub const TVMFFIObjectHandle = ?*anyopaque;
        pub const TVMFFIByteArray = extern struct {
            data: [*]const u8,
            size: usize,
        };
        pub const TVMFFIErrorCell = opaque {};
        pub const TVMFFIFieldInfo = opaque {};
        pub const TVMFFIObject = opaque {};
        pub const TVMFFITypeInfo = opaque {};
        pub const DLTensor = opaque {};
        pub const DLManagedTensor = opaque {};
        pub const kTVMFFISmallStr: c_int = 0;
    };

// Re-export all types and functions
// Types
pub const TVMFFIAny = Impl.TVMFFIAny;
pub const TVMFFIObjectHandle = Impl.TVMFFIObjectHandle;
pub const TVMFFIByteArray = Impl.TVMFFIByteArray;
pub const TVMFFIErrorCell = if (tvm_enabled) Impl.TVMFFIErrorCell else Impl.TVMFFIErrorCell;
pub const TVMFFIFieldInfo = if (tvm_enabled) Impl.TVMFFIFieldInfo else Impl.TVMFFIFieldInfo;
pub const TVMFFIObject = if (tvm_enabled) Impl.TVMFFIObject else Impl.TVMFFIObject;
pub const TVMFFITypeInfo = if (tvm_enabled) Impl.TVMFFITypeInfo else Impl.TVMFFITypeInfo;
pub const DLTensor = if (tvm_enabled) Impl.DLTensor else Impl.DLTensor;
pub const DLManagedTensor = if (tvm_enabled) Impl.DLManagedTensor else Impl.DLManagedTensor;

// Functions
pub const TVMFFIObjectDecRef = if (tvm_enabled) Impl.TVMFFIObjectDecRef else undefined;  // NOTE: audit use of undefined
pub const TVMFFIObjectIncRef = if (tvm_enabled) Impl.TVMFFIObjectIncRef else undefined;  // NOTE: audit use of undefined
pub const TVMFFIFunctionCreate = if (tvm_enabled) Impl.TVMFFIFunctionCreate else undefined; // NOTE: audit use of undefined
pub const TVMFFIFunctionCall = if (tvm_enabled) Impl.TVMFFIFunctionCall else undefined; // NOTE: audit use of undefined
pub const TVMFFIFunctionGetGlobal = if (tvm_enabled) Impl.TVMFFIFunctionGetGlobal else undefined; // NOTE: audit use of undefined
pub const TVMFFIFunctionSetGlobal = if (tvm_enabled) Impl.TVMFFIFunctionSetGlobal else undefined; // NOTE: audit use of undefined
pub const TVMFFIGetLastError = if (tvm_enabled) Impl.TVMFFIGetLastError else undefined; // NOTE: audit use of undefined
pub const TVMFFIErrorMoveFromRaised = if (tvm_enabled) Impl.TVMFFIErrorMoveFromRaised else undefined; // NOTE: audit use of undefined
pub const TVMFFIGetTypeInfo = if (tvm_enabled) Impl.TVMFFIGetTypeInfo else undefined; // NOTE: audit use of undefined
pub const TVMFFIStringFromByteArray = if (tvm_enabled) Impl.TVMFFIStringFromByteArray else undefined; // NOTE: audit use of undefined
pub const TVMFFITensorFromDLPack = if (tvm_enabled) Impl.TVMFFITensorFromDLPack else undefined; // NOTE: audit use of undefined

// Constants
pub const kTVMFFINone = if (tvm_enabled) Impl.kTVMFFINone else 0;
pub const kTVMFFIInt = if (tvm_enabled) Impl.kTVMFFIInt else 0;
pub const kTVMFFIFloat = if (tvm_enabled) Impl.kTVMFFIFloat else 0;
pub const kTVMFFIBool = if (tvm_enabled) Impl.kTVMFFIBool else 0;
pub const kTVMFFIRawStr = if (tvm_enabled) Impl.kTVMFFIRawStr else 0;
pub const kTVMFFITensor = if (tvm_enabled) Impl.kTVMFFITensor else 0;
pub const kTVMFFIOpaquePtr = if (tvm_enabled) Impl.kTVMFFIOpaquePtr else 0;
pub const kTVMFFIModule = if (tvm_enabled) Impl.kTVMFFIModule else 0;
pub const kTVMFFIFunction = if (tvm_enabled) Impl.kTVMFFIFunction else 0;
pub const kTVMFFISmallStr = if (tvm_enabled) Impl.kTVMFFISmallStr else 0;
pub const kTVMFFIStr = if (tvm_enabled) Impl.kTVMFFIStr else 0;
pub const kTVMFFIDataType = if (tvm_enabled) Impl.kTVMFFIDataType else 0;
pub const kTVMFFIDevice = if (tvm_enabled) Impl.kTVMFFIDevice else 0;
pub const kTVMFFIStaticObjectBegin = if (tvm_enabled) Impl.kTVMFFIStaticObjectBegin else 0;

// DLPack constants
pub const kDLCPU = if (tvm_enabled) Impl.kDLCPU else 1;
pub const kDLCUDA = if (tvm_enabled) Impl.kDLCUDA else 2;
pub const kDLFloat = if (tvm_enabled) Impl.kDLFloat else 2;
