# Ops

## Overview

We want acceptable parity on basic ops across host and CUDA device backends. We should maintain a kernel table.

The following table highlights issues with how things are organized. We will address these closer to public release as we do a full review of the codebase and reorganize things. We should keep notes below this table regarding the future.

| NDTensor     | NDArray              | Kernel(s)             | HostDevice              | CudaDevice                        |
| ------------ | -------------------- | --------------------- | ----------------------- | --------------------------------- |
| random       | random               | mem_random            | [x]                     | [x]                               |
| sequence     | sequence             | mem_sequence          | [x]                     | [x]                               |
| transpose    | transpose            | transpose             | [x] (naive)             | [x] (permute)                     |
| subset*      |                      | N/A                   | [ ]                     | [ ]                               |
| alias        | N/A                  | N/A                   |                         |                                   |
| clip*_norm   | _clip_norm           | clip_nrm2             | [x]                     | [x]                               |
| clamp        | clamp                | clamp_fwd/clamp_bwd   | [x][x]                  | [ ][ ]                            |
| add_scalar   | add_scalar           | add                   | [x]                     | [x]                               |
| sub_scalar   | add_scalar           | see above             | see above               | see above                         |
| add          | add                  | add                   | [x]                     | [x]                               |
| sub          | sub                  | sub                   | [x]                     | [x]                               |
| mul          | mul                  | mul                   | [x]                     | [x]                               |
| div          | div                  | div                   | [x]                     | [x]                               |
| max          | max                  | max_fwd/max_bwd       | [x] [x]                 | [x] [x]                           |
| exp          | exp                  | exp_fwd/exp_bwd       | [x] [x]                 | [x] [ ]                           |
| pow          | pow                  | pow_fwd/pow_bwd       | [x] [ ]                 | [x] [ ]                           |
| sqrt         | sqrt                 | sqrt_fwd/sqrt_bwd     | [x] [ ]                 | [x] [ ]                           |
| rsqrt        | rsqrt                | rsqrt_fwd/rsqrt_bwd   | [x] [ ]                 | [x] [ ]                           |
| bmm          | bmm                  | rsqrt_fwd/rsqrt_bwd   | see bmm_acc_            | see bmm_acc_                      |
| bmm_acc_     | bmm_acc_             | bmm_acc_ (gemm)       | [x]                     | [x]                               |
| dot          | dot                  | dot,axpy**            | [x] [x]                 | [x] [x]                           |
| matvec*      | matvec*              | matvec, outer (ger)** | [x] [x]                 | [x] [x]                           |
| gather       | gather,scatter_add** | mem_take,scatter_add  | [x] [x] (vadd)          | [ ] (has mem_take, no gather) [ ] |
| scatter_add* | scatter_add          | scatter_add           | [x] (vadd)              | [ ] (has mem_take, no gather)     |
| index_select | MISSING              | MISSING               | [ ]                     | [ ]                               |
| clone        | copy                 | mem_cache_dupe        | [x]                     | [x]                               |
|              | equal                | equal                 | [ ]                     | [x]                               |
|              | scale                | scale (scal)          | [x]                     | [x]                               |
|              | outer                | outer (ger)           | [x]                     | [x]                               |
|              | _axpy                | axpy                  | [x]                     | [x]                               |
|              | sum_along            | sum_along             | [x] (native reduce.zig) | [x] (reduce_one)                  |
|              | max_along            | max_along             | [x] (native inline)     | [x] (reduce_one)                  |
|              | take                 | mem_take              | [x]                     | [x]                               |
|              | l2_norm              | nrm2                  | [x]                     | [x]                               |
|              | broadcast            | ?                     | ?                       | ?                                 |
|              | unbroadcast          | ?                     | ?                       | ?                                 |

*Implemented with mem_cache_dupe/free
**For backward

## Notes

