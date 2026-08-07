{pkgs, ...}: {
  projectRootFile = "flake.nix";

  programs.alejandra.enable = true;
  programs.zig = {
    enable = true;
    package = pkgs.zig;
  };
  programs.shfmt.enable = true;

  settings.formatter.zig.excludes = [
    "src/c/xla/proto/*.pb.zig"
  ];

  # TODO(formatting): Choose and check in the C/C++/CUDA style before enabling
  #  clang-format. CUDA sources use the C++ formatter.
  # programs.clang-format.enable = true;

  # TODO(formatting): Choose repository policies before enabling additional
  #  language formatters.
  # programs.ruff-format.enable = true;
  # programs.prettier.enable = true;
  # programs.mdformat.enable = true;
}
