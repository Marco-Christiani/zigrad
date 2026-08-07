{
  coreutils,
  jq,
  lib,
  runCommand,
  revision,
  zigradAutodoc,
}: let
  assets = [
    {
      name = "index.html";
      contentType = "text/html; charset=utf-8";
    }
    {
      name = "main.js";
      contentType = "application/javascript; charset=utf-8";
    }
    {
      name = "main.wasm";
      contentType = "application/wasm";
    }
    {
      name = "sources.tar";
      contentType = "application/x-tar";
    }
    {
      name = "zg-logo.svg";
      contentType = "image/svg+xml";
    }
  ];
  assetSpec = builtins.toFile "zigrad-autodoc-assets.json" (builtins.toJSON assets);
  validRevision = builtins.match "[0-9a-f]{40}" revision != null;
in
  assert lib.assertMsg validRevision "zigrad-autodoc-candidate requires a clean Git revision";
    runCommand "zigrad-autodoc-candidate-${builtins.substring 0 12 revision}" {
      nativeBuildInputs = [coreutils jq];
      passthru = {
        bundleId = revision;
        zigVersion = zigradAutodoc.zigVersion;
      };
    } ''
      set -euo pipefail

      cp -R ${zigradAutodoc}/autodoc "$out"
      chmod -R u+w "$out"

      assets_json="$TMPDIR/assets.json"
      printf '{}\n' > "$assets_json"

      while IFS=$'\t' read -r name content_type; do
        path="$out/$name"
        if [ ! -f "$path" ]; then
          echo "missing required autodoc asset: $path" >&2
          exit 1
        fi

        sha256="$(sha256sum "$path" | cut -d' ' -f1)"
        size="$(stat --format=%s "$path")"
        key="autodoc/v1/bundles/${revision}/$name"

        jq \
          --arg name "$name" \
          --arg key "$key" \
          --arg sha256 "$sha256" \
          --argjson size "$size" \
          --arg content_type "$content_type" \
          '. + {($name): {
            key: $key,
            sha256: $sha256,
            size: $size,
            content_type: $content_type
          }}' \
          "$assets_json" > "$assets_json.next"
        mv "$assets_json.next" "$assets_json"
      done < <(jq -r '.[] | [.name, .contentType] | @tsv' ${assetSpec})

      jq \
        --null-input \
        --arg bundle_id "${revision}" \
        --arg zig_version "${zigradAutodoc.zigVersion}" \
        --slurpfile assets "$assets_json" \
        '{
          schema_version: 1,
          bundle_id: $bundle_id,
          commit: $bundle_id,
          zig_version: $zig_version,
          assets: $assets[0]
        }' > "$out/manifest.json"
    ''
