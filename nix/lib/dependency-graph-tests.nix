{lib}: let
  nodes = {
    root.requires = [
      "middle"
      "leaf"
    ];
    middle.requires = ["leaf"];
    leaf = {};
    incompatible.conflicts = ["leaf"];
  };
  graph = import ./dependency-graph.nix {inherit lib nodes;};
in {
  testTransitiveClosure = {
    expr = graph.close ["root"];
    expected = [
      "leaf"
      "middle"
      "root"
    ];
  };

  testDuplicateDemandsAreNormalized = {
    expr = graph.close [
      "leaf"
      "leaf"
    ];
    expected = ["leaf"];
  };

  testUnknownDemandFails = {
    expr = (builtins.tryEval (graph.close ["missing"])).success;
    expected = false;
  };

  testConflictIsReported = {
    expr = graph.conflictFor [
      "incompatible"
      "leaf"
    ];
    expected = "incompatible conflicts with leaf";
  };

  testUnknownNodeReferenceIsReported = let
    invalid = import ./dependency-graph.nix {
      inherit lib;
      nodes.root.requires = ["missing"];
    };
  in {
    expr = invalid.invalidReference;
    expected = "root references unknown dependency 'missing'";
  };
}
