pub const impl = @cImport({
    @cInclude("cuda_utils.h");
    @cInclude("blas_conflux.h");
    @cInclude("nn_conflux.h");
});
