#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import re
from . import fndecls as TARGET_FUNCTIONS

# TARGET_FUNCTIONS = {
#     "build_tracy",
#     # ----- ndtensor -----
#     # "init",
#     # "empty",
#     # "zeros",
#     # "get_shape"
#     # "get_size"
#     # "get_strides"
#     # "get_data"
#     "cast",
#     # _unsqueeze
#     # requires_grad
#     # setup_grad
#     # create_dependent
#     # "deinit",
#     # "teardown",
#     # acquire
#     # release
#     "from_zarray",
#     "to_deviceImpl",
#     "_to_device",
#     "to_device",
#     "to_device_bw_impl",
#     "clone",
#     # log_shape
#     # _reshape
#     # reshape
#     "reshape_bw_impl",
#     "transpose",
#     "transpose_bw_impl",
#     "transposeBwImpl",
#     # set_label
#     # fill
#     # get
#     # set
#     # pos_to_index
#     # flex_pos_to_index
#     # index_to_pos
#     "slice_ranges",
#     "set_slice",
#     # print
#     # print_to_writer
#     "clip_grad_norm_delta",
#     # set_children
#     "add",
#     "add_bw_impl",
#     "addBwImpl",
#     "sub",
#     "sub_bw_impl",
#     "subBwImpl",
#     "mul",
#     "mul_bw_impl",
#     "mulBwImpl",
#     "div",
#     "div_bw_impl",
#     "divBwImpl",
#     "max",
#     "max_bw_impl",
#     "maxBwImpl",
#     "exp",
#     "exp_bw_impl",
#     "expBwImpl",
#     "bmm",
#     "bmm_acc",
#     "bmmAcc",
#     "bw_bmm_acc",
#     "dot",
#     "dot_bw_impl",
#     "matvec",
#     "matvec_bw_impl",
#     "sum",
#     "sum_bw_impl",
#     "max_over_dim",
#     "max_over_dim_bw_impl",
#     "gather",
#     "gather_bw_impl",
#     "backward",
#     # set_backward
#     # print_arrows
#     # ----- ndarray -----
#     # "init",
#     # "deinit",
#     # "empty",
#     # "zeros",
#     # "copy",
#     # "cast",
#     # "fill",
#     # "log_shape",
#     "_reshape",
#     # "get",
#     # "set",
#     # "pos_to_offset",
#     # "offset_to_pos",
#     # "size",
#     # "print",
#     # "print_to_writer",
#     # "has_transfer",
#     # "print_to_writer_impl",
#     # "slice",
#     # "slice_unsafe_no_alloc",
#     # "slice_raw_no_alloc",
#     # "slice_ranges",
#     # "set_slice_ranges",
#     # "get_stride",
#     "add",
#     "_add",
#     "_add_scalar",
#     "_addScalar",
#     "sub",
#     "_sub",
#     "mul",
#     "_mul",
#     "div",
#     "_div",
#     "sum",
#     "sum_no_alloc",
#     "max",
#     "exp",
#     "_exp",
#     "_scale",
#     "bmm",
#     "_bmm_acc",
#     "_bmmAcc",
#     "expand_as",
#     "expandAs",
#     # "content_check",
#     "dot",
#     "outer",
#     "matvec",
#     "sum_along",
#     "sumAlong",
#     "max_over_dim",
#     "maxOverDim",
#     "gather",
#     "take",
#     "l2_norm",
#     "clip_norm",
#     "unbroadcast",
#     "transpose",
# }


def has_tracy_zone(lines: list[str], start_idx: int) -> bool:
    """Check if Tracy zone is already present in the function body."""
    # Look at the next few lines after the opening brace
    for line in lines[start_idx : start_idx + 4]:
        if "tracy.zone" in line:
            return True
    return False


def inject_tracy_zones(fname: str, content: str) -> tuple[str, bool]:
    """Injects tracy zones into the target functions if not already present.

    Returns
        modified_content, was_modified
    """
    lines = content.splitlines()
    modified_lines = []
    was_modified = False

    i = 0
    while i < len(lines):
        line = lines[i]
        modified_lines.append(line)

        # Check if this line starts a target function
        for func_name in TARGET_FUNCTIONS:
            pattern = rf"^\s*(pub )?(inline )?fn {func_name}\s*\("
            if re.match(pattern, line):
                # Find opening brace line
                j = i
                while j < len(lines) and "{" not in lines[j]:
                    j += 1

                if j < len(lines) and not has_tracy_zone(lines, j + 1):
                    match = re.match(r"^\s*", lines[j])
                    assert match is not None
                    indent = match[0]  # get indent level
                    # Insert tracy zone after opening brace
                    tracy_lines = [
                        f'{indent}    const zone = tracy.traceNamed(@src(), "{fname}:{func_name}");',
                        f"{indent}    defer zone.end();",
                    ]
                    modified_lines.extend(tracy_lines)
                    was_modified = True
                break
        i += 1

    if was_modified:
        modified_lines = ['const tracy = @import("tracy");'] + modified_lines
    # need an extra newline at the end of the file
    modified_lines.append("\n")
    return "\n".join(modified_lines), was_modified


def process_file(path: Path, remove: bool) -> bool:
    """Process a single file. Returns True if file was modified."""
    try:
        content = path.read_text()
        if remove:
            modified_content, was_modified = remove_tracy_zones(content)
        else:
            modified_content, was_modified = inject_tracy_zones(path.stem, content)

        if was_modified:
            path.write_text(modified_content)
            print(f"Modified: {path}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {path}: {e}", file=sys.stderr)
        return False


def remove_tracy_zones(content: str) -> tuple[str, bool]:
    """Removes tracy zones from the target functions.

    Returns
        modified_content, was_modified
    """
    lines = content.splitlines()
    modified_lines = []
    was_modified = False
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()
        if "tracy.traceNamed" in line:  # Skip tracy zone lines
            was_modified = True
            # Skip the defer line too
            # if i + 1 < len(lines) and "defer zone.deinit()" in lines[i + 1]:
            #     i += 2
            #     continue
            i += 1
            continue
        elif "defer zone.end()" in line:
            was_modified = True
            i += 1
            continue
        elif 'const tracy = @import("tracy");' in line:
            was_modified = True
            i += 1
            continue

        modified_lines.append(lines[i])
        i += 1

    return "\n".join(modified_lines), was_modified


def main():
    parser = argparse.ArgumentParser(description="Inject or remove Tracy zones in Zig functions")
    parser.add_argument("path", type=Path, help="File or directory to process")
    parser.add_argument("--ext", default=".zig", help="File extension to process (default: .zig)")
    parser.add_argument("--remove", action="store_true", help="Remove Tracy zones instead of adding them")
    args = parser.parse_args()

    assert args.path.exists()
    modified_count = 0
    paths = iter([args.path]) if args.path.is_file() else args.path.rglob(f"*{args.ext}")
    for path in paths:
        modified_count += process_file(path, args.remove)

    print(f"\nModified {modified_count} file(s)")


if __name__ == "__main__":
    main()
