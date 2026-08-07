{pkgs}: let
  specifications = builtins.fromJSON (builtins.readFile ./external-sources.json);
  fetch = name: specification: let
    src =
      if specification.type == "github"
      then
        pkgs.fetchFromGitHub {
          inherit (specification) owner repo rev hash;
          fetchSubmodules = specification.fetch_submodules or false;
        }
      else if specification.type == "url"
      then
        pkgs.fetchurl {
          inherit (specification) url hash;
          name = "${name}-${specification.rev}.tar.gz";
        }
      else throw "unsupported external source type '${specification.type}' for ${name}";
  in
    specification
    // {
      inherit src;
    };
in
  builtins.mapAttrs fetch specifications
