# Hand-edited because zon2nix 0.1.3 cannot parse the 0.16-format `build.zig.zon`.
# When this list shifts, replace each `hash` with `lib.fakeHash` and run
# `nix build`. Nix will print the expected sha256, paste it back here.

{ linkFarm, fetchzip, fetchgit }:

linkFarm "zig-packages" [
  {
    name = "cova-0.10.1-_OE4R2j0BADMJ-wKl1l_-07Z2XDulhqFqHiciaX4dPR2";
    path = fetchgit {
      url = "https://github.com/Marco-Christiani/cova";
      rev = "919ca89145a801a3f278cba2641e1d39e1859761";
      hash = "sha256-61l7VRZshV72dMFV9NhthFMb4qI3RafoLa1qhEupqJ4=";
    };
  }
  {
    name = "safetensors_zg-0.0.1-dRXUiNvtAADkooBkmhGsycmgwajaskWj3MiB6feJBMUS";
    path = fetchgit {
      url = "https://github.com/Marco-Christiani/safetensors-zg";
      rev = "9cc91a16eb93e86f9953e24410065d77968f6578";
      hash = "sha256-goZHJReIJBb2wKJ3nYHrtEXwzdsgd0m+fGs+yNF1sOM=";
    };
  }
  {
    name = "protobuf-5.0.0-0e82ahSVKACt8CYbD8lZ3nG0HwxsAzx0DMnwUH15DPiT";
    path = fetchgit {
      url = "https://github.com/Marco-Christiani/zig-protobuf";
      rev = "ce3735fa4ee099a6c767ac87301ffdeb5c484a40";
      hash = "sha256-5AWxFzJuQdulj/AQbL9G9MCql51q8RDIw7duqqyunQE=";
    };
  }
  {
    # Lazy dep of `protobuf`: the protoc binary for linux x86_64. Only the
    #  platform we build for needs to be present in the link farm.
    name = "N-V-__8AAGKbngAmNuaBMSXq_WgmQi6N8WVWVKp0moFSTvoJ";
    path = fetchzip {
      url = "https://github.com/protocolbuffers/protobuf/releases/download/v32.1/protoc-32.1-linux-x86_64.zip";
      hash = "sha256-+nzX+bcCBfIewSNHxE/bez8STbL+LyIdXOfxrNAyknE=";
      stripRoot = false;
    };
  }
]
