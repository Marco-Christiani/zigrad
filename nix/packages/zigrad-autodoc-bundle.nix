{
  coreutils,
  jq,
  lib,
  runCommand,
  revision,
  zigradAutodoc,
  localPreview ? false,
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
  sourceSpec = builtins.toFile "zigrad-autodoc-sources.json" (builtins.toJSON (
    zigradAutodoc.autodocSources
    // {
      zigrad = {
        repository = "https://github.com/Marco-Christiani/zigrad";
        inherit revision;
        revision_url = "https://github.com/Marco-Christiani/zigrad/commit/${revision}";
        file_url_template = "https://github.com/Marco-Christiani/zigrad/blob/${revision}/src/{path}";
      };
    }
  ));
  validRevision = revision != null && builtins.match "[0-9a-f]{40}" revision != null;
  targetName =
    if localPreview
    then "zigrad-autodoc-preview"
    else "zigrad-autodoc-candidate";
in
  assert lib.assertMsg validRevision "${targetName} requires a Git revision";
    runCommand "${targetName}-${builtins.substring 0 12 revision}" {
      nativeBuildInputs = [coreutils jq];
      passthru = {
        inherit localPreview revision;
        zigVersion = zigradAutodoc.zigVersion;
      };
    } ''
      set -euo pipefail

      cp -R ${zigradAutodoc}/autodoc "$out"
      chmod -R u+w "$out"

      if ${
        if localPreview
        then "true"
        else "false"
      }; then
        bundle_id="$(${coreutils}/bin/sha256sum "$out"/* | ${coreutils}/bin/sha256sum | ${coreutils}/bin/cut -c1-40)"
      else
        bundle_id="${revision}"
      fi

      assets_json="$TMPDIR/assets.json"
      printf '{}\n' > "$assets_json"

      while IFS=$'\t' read -r name content_type; do
        path="$out/$name"
        if [ ! -f "$path" ]; then
          echo "missing required autodoc asset: $path" >&2
          exit 1
        fi

        sha256="$(${coreutils}/bin/sha256sum "$path" | ${coreutils}/bin/cut -d' ' -f1)"
        size="$(${coreutils}/bin/stat --format=%s "$path")"
        key="autodoc/source/v1/bundles/$bundle_id/$name"

        ${jq}/bin/jq \
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
      done < <(${jq}/bin/jq -r '.[] | [.name, .contentType] | @tsv' ${assetSpec})

      ${jq}/bin/jq \
        --null-input \
        --arg bundle_id "$bundle_id" \
        --arg zig_version "${zigradAutodoc.zigVersion}" \
        --argjson local_preview ${
        if localPreview
        then "true"
        else "false"
      } \
        --slurpfile assets "$assets_json" \
        --slurpfile sources ${sourceSpec} \
        '{
          schema_version: 1,
          bundle_id: $bundle_id,
          zig_version: $zig_version,
          local_preview: $local_preview,
          sources: $sources[0],
          assets: $assets[0]
        }' > "$out/manifest.json"
    ''
