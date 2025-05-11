import os
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some strings.')    
    parser.add_argument('backend', type=str)    
    args = parser.parse_args()

    here = Path(__file__).parent.resolve()
    root = here.parent / Path("src/device/root.zig")

    roots = {
        "HOST": "host_device.zig",
        "CUDA": "cuda_device.zig",
    }

    zigrad_backend = "HOST"
    
    if args.backend is None:
        print("ZIGRAD_BACKEND not specified, using HOST...")
    else:
        print(f"ZIGRAD_BACKEND specified as {args.backend}...")
        zigrad_backend = args.backend
    
    imports = [
        "pub const Backend = enum{ HOST, CUDA };",
        f"pub const backend: Backend = .{zigrad_backend};",
        f'pub const device = @import("{roots[zigrad_backend]}");',
        'pub const ReduceType = @import("device_common.zig").ReduceType;',
        'pub const SmaxType = @import("device_common.zig").SmaxType;',
        'pub const RandType = @import("device_common.zig").RandType;',
        'pub const using_mkl = @import("host_device.zig").using_mkl;',
        'pub const SystemAllocator = @import("device_common.zig").SystemAllocator;',
        'pub const opspec = @import("opspec.zig");',
     ]
    
    with open(root, 'w') as file:
        file.write('\n'.join(imports))
        
