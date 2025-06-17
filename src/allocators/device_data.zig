//! Device Memory... a lot to say for such a short file.
//!
//! DeviceData objects are "contextual" data objects in
//! two way:
//!
//! 1) They may belong to different address spaces
//!    that are not safe to directly work with. You
//!    might call that the "device" context. Since Zigrad
//!    deals with device memory, you must be careful when
//!    assuming it's safe to access te "raw" data elements
//!    directly. This is only safe given that you are aware
//!    of the context it was created in. For example, any
//!    data allocated by the HostDevice is safe to access
//!    directly. This is not true for the CudaDevice as the
//!    slice may point to a location on your GPU - accessing
//!    it like a typical slice (ex. data.raw[0]) can cause
//!    your progrma to segfault - doing so assumes that the
//!    address resides in host-accessible memory (such as RAM).
//!
//! 2) They can optionally store data in their `ctx` field.
//!    The `ctx` field is an ambigously purposed integer
//!    that can store information useful to the parent allocator.
//!    Like any allocation, it should only be returned to its
//!    allocator-of-origin.
//!
//! When working with devices other than the HostDevice,
//! use the device api or higher-level objects such as
//! NDTensor to do your data manipulation. Likewise, be
//! careful about sharing data across device instances.
//! Two CudaDevices maybe be assigned to separate pieces of
//! hardware via their "device_number" and thus have
//! independent physical memory.
//!
//! In total, it's safe to use DeviceData objects with the devices
//! that created them (otherwise, additional steps may be required
//! (such as calling "mem_transfer").
//!
//! And again, never free a DeviceData object on a different device
//! instance than the one that created it - same with allocators.
//!
//! With all out of the way... behold...
//!
pub fn DeviceData(T: type) type {
    return struct {
        // Slice of device-specific memory.
        raw: []T,
        // The use of this integer is context dependent.
        // It could be a pointer, number, flags... etc...
        ctx: usize,
    };
}

/// Common way that devices signal out of device memory.
pub const Error = error{DeviceOOM};
