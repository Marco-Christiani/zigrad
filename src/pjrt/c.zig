/// Centralized C imports for PJRT
///
/// Multiple @cImport calls create incompatible types, so we centralize here.
pub const c = @cImport({
    @cInclude("dlfcn.h");
    @cInclude("pjrt/c/pjrt_c_api.h");
});

// Re-export commonly used types
pub const PJRT_Api = c.PJRT_Api;
pub const PJRT_Client = c.PJRT_Client;
pub const PJRT_Device = c.PJRT_Device;
pub const PJRT_LoadedExecutable = c.PJRT_LoadedExecutable;
pub const PJRT_Buffer = c.PJRT_Buffer;
pub const PJRT_Event = c.PJRT_Event;
pub const PJRT_Error = c.PJRT_Error;
