{
  lib,
  rustPlatform,
  src,
  sourceRoot,
  lockFile,
}: let
  baseLock = builtins.readFile lockFile;

  lockFor = pname:
    builtins.toFile "${pname}-Cargo.lock" (
      builtins.replaceStrings
      ["name = \"abstract_subexpr\""]
      ["name = \"${pname}\""]
      baseLock
    );

  mkRustLib = {
    pname,
    crateDir,
  }: let
    crateLock = lockFor pname;
  in
    rustPlatform.buildRustPackage {
      inherit pname src;
      version = "0.1.0";
      sourceRoot = "${sourceRoot}/${crateDir}";

      cargoLock.lockFile = crateLock;

      # The package removes the unused pyo3 dependency, so the matching lock
      # is a package input. buildRustPackage also requires that lock in the
      # unpacked crate root before its post-patch consistency check.
      prePatch = ''
        install -m 0644 ${crateLock} Cargo.lock
      '';

      postPatch = ''
        substituteInPlace Cargo.toml \
          --replace-fail \
            'pyo3 = { version = "0.25", features = ["extension-module"] }' \
            ""
      '';

      installPhase = ''
        runHook preInstall

        library="$(find target -name "lib${pname}.so" -type f -print -quit)"
        test -n "$library"
        mkdir -p "$out/lib"
        install -Dm755 "$library" "$out/lib/lib${pname}.so"

        runHook postInstall
      '';

      meta = {
        description = "Mirage Rust cdylib: ${pname}";
        license = lib.licenses.asl20;
        platforms = lib.platforms.linux;
      };
    };
in {
  abstract_subexpr = mkRustLib {
    pname = "abstract_subexpr";
    crateDir = "src/search/abstract_expr/abstract_subexpr";
  };

  formal_verifier = mkRustLib {
    pname = "formal_verifier";
    crateDir = "src/search/verification/formal_verifier_equiv";
  };
}
