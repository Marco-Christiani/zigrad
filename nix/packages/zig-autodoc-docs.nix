{
  patch,
  runCommand,
  zig,
}:
runCommand "zig-autodoc-docs-${zig.version}" {
  nativeBuildInputs = [patch];
} ''
  mkdir -p "$out"

  for path in ${zig}/lib/*; do
    if [ "$(basename "$path")" != "docs" ]; then
      ln -s "$path" "$out/$(basename "$path")"
    fi
  done

  cp -R ${zig}/lib/docs "$out/docs"
  chmod -R u+w "$out/docs"
  patch --batch --forward --fuzz=0 --strip=1 --directory="$out/docs" < ${../patches/zig-autodoc-math.patch}
  patch --batch --forward --fuzz=0 --strip=1 --directory="$out/docs" < ${../patches/zig-autodoc-namespace-alias.patch}
''
