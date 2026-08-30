{
  lib,
  nodes,
}: let
  nodeFor = name:
    nodes.${name}
    or (throw "unknown Zigrad build dependency '${name}'");

  normalize = names:
    lib.sort builtins.lessThan (lib.unique names);

  close = demands: let
    current = normalize demands;
    expanded = normalize (
      current
      ++ lib.concatMap (name: (nodeFor name).requires or []) current
    );
  in
    if expanded == current
    then current
    else close expanded;

  conflictFor = resolved:
    lib.findFirst
    (entry: entry != null)
    null
    (lib.concatMap
      (name:
        map
        (other:
          if lib.elem other resolved
          then "${name} conflicts with ${other}"
          else null)
        ((nodeFor name).conflicts or []))
      resolved);

  invalidReference =
    lib.findFirst
    (entry: entry != null)
    null
    (lib.concatMap
      (name:
        map
        (reference:
          if builtins.hasAttr reference nodes
          then null
          else "${name} references unknown dependency '${reference}'")
        ((nodes.${name}.requires or []) ++ (nodes.${name}.conflicts or [])))
      (builtins.attrNames nodes));
in {
  inherit close conflictFor invalidReference normalize;
}
