{
  fetchgit,
  fetchzip,
  lib,
  linkFarm,
  stdenv,
  withPjrt ? false,
}: let
  safetensors = {
    name = "safetensors_zg-0.0.1-dRXUiNvtAADkooBkmhGsycmgwajaskWj3MiB6feJBMUS";
    path = fetchgit {
      url = "https://github.com/Marco-Christiani/safetensors-zg";
      rev = "9cc91a16eb93e86f9953e24410065d77968f6578";
      hash = "sha256-goZHJReIJBb2wKJ3nYHrtEXwzdsgd0m+fGs+yNF1sOM=";
    };
  };

  protobuf = {
    name = "protobuf-5.0.0-0e82ahSVKACt8CYbD8lZ3nG0HwxsAzx0DMnwUH15DPiT";
    path = fetchgit {
      url = "https://github.com/Marco-Christiani/zig-protobuf";
      rev = "ce3735fa4ee099a6c767ac87301ffdeb5c484a40";
      hash = "sha256-5AWxFzJuQdulj/AQbL9G9MCql51q8RDIw7duqqyunQE=";
    };
  };

  protocAssets = {
    x86_64-linux = {
      name = "N-V-__8AAGKbngAmNuaBMSXq_WgmQi6N8WVWVKp0moFSTvoJ";
      url = "https://github.com/protocolbuffers/protobuf/releases/download/v32.1/protoc-32.1-linux-x86_64.zip";
      hash = "sha256-+nzX+bcCBfIewSNHxE/bez8STbL+LyIdXOfxrNAyknE=";
    };
    aarch64-linux = {
      name = "N-V-__8AAJKMngA8y82sENkRg-JF100BtRa7GQxoBIfU3c3_";
      url = "https://github.com/protocolbuffers/protobuf/releases/download/v32.1/protoc-32.1-linux-aarch_64.zip";
      hash = "sha256-HPNufPYkFYW9c75ZuVwQC9I7hC5pMty2P0CBrIqxud0=";
    };
  };

  protocAsset =
    protocAssets.${stdenv.hostPlatform.system}
    or (throw "zig-dependencies: unsupported protoc host ${stdenv.hostPlatform.system}");
  protoc = {
    inherit (protocAsset) name;
    path = fetchzip {
      inherit (protocAsset) url hash;
      stripRoot = false;
    };
  };
in
  linkFarm "zig-packages" (
    [safetensors]
    ++ lib.optionals withPjrt [
      protobuf
      protoc
    ]
  )
