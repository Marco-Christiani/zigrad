{lib, root}: let
  fs = lib.fileset;
in
  fs.toSource {
    root = root;
    fileset = fs.unions [
      (root + /build.zig)
      (root + /build.zig.zon)
      (root + /src)
    ];
  }
