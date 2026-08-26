const std = @import("std");
pub const schema = @import("schema.zig");

const max_command_depth = 8;

/// Render terminal help for one command.
pub fn write_help(writer: *std.Io.Writer, command: *const schema.Command) !void {
    try writer.writeAll("Usage: ");
    try write_command_path(writer, command);
    if (command == &schema.root) try writer.writeAll(" [global options]");
    if (command.subcommands.len != 0) try writer.writeAll(" <command>");
    for (command.positionals) |positional| {
        if (positional.optional) {
            try writer.print(" [{s}]", .{positional.name});
        } else {
            try writer.print(" {s}", .{positional.name});
        }
    }
    if (command.options.len != 0 and command != &schema.root)
        try writer.writeAll(" [options]");
    try writer.writeAll("\n\n");

    try writer.print("{s}\n", .{command.summary});
    if (command.description) |description|
        try writer.print("\n{s}\n", .{description});
    if (command.requirement) |requirement|
        try writer.print("\nRequires: {s}\n", .{requirement});

    if (command.subcommands.len != 0) {
        try writer.writeAll("\nCommands:\n");
        for (command.subcommands) |*child| {
            if (child.hidden) continue;
            try writer.print("  {s:<24} {s}", .{ child.name, child.summary });
            if (child.requirement) |requirement|
                try writer.print(" [requires {s}]", .{requirement});
            try writer.writeByte('\n');
        }
    }

    if (command.positionals.len != 0) {
        try writer.writeAll("\nArguments:\n");
        for (command.positionals) |positional|
            try writer.print("  {s:<24} {s}\n", .{ positional.name, positional.description });
    }

    if (command.options.len != 0) {
        try writer.writeAll(if (command == &schema.root) "\nGlobal options:\n" else "\nOptions:\n");
        for (command.options) |*option| try write_help_option(writer, option);
    }
    if (command != &schema.root and schema.find_option(command, "help") == null) {
        if (command.options.len == 0)
            try writer.writeAll("\nOptions:\n");
        try write_help_option(writer, &schema.global_options[schema.global_options.len - 1]);
    }

    if (command.subcommands.len != 0) {
        try writer.writeAll("\nRun '");
        try write_command_path(writer, command);
        try writer.writeAll(" <command> --help' for command help.\n");
    }
}

fn write_help_option(writer: *std.Io.Writer, option: *const schema.Option) !void {
    try writer.writeAll("  ");
    if (option.short_name) |short_name|
        try writer.print("-{c}, ", .{short_name});
    if (option.negatable)
        try writer.writeAll("--[no-]")
    else
        try writer.writeAll("--");
    try writer.writeAll(option.long_name);
    if (option.value_name) |value_name| {
        if (option.value_optional)
            try writer.print("[={s}]", .{value_name})
        else
            try writer.print(" {s}", .{value_name});
    }
    try writer.print("\n      {s}", .{option.description});
    if (option.choices.len != 0) {
        try writer.writeAll(" (");
        for (option.choices, 0..) |choice, index| {
            if (index != 0) try writer.writeAll(", ");
            try writer.writeAll(choice);
        }
        try writer.writeByte(')');
    }
    try writer.writeByte('\n');
}

/// Render a section 1 manual page for the complete command tree.
pub fn write_manpage(writer: *std.Io.Writer, version: []const u8) !void {
    try writer.writeAll(
        \\.TH ZIGRAD 1
        \\.SH NAME
        \\zigrad \- Zigrad core and integration development CLI
        \\.SH SYNOPSIS
        \\.B zigrad
        \\[global options] <group> <command> [options]
        \\.SH DESCRIPTION
        \\Integration commands remain visible when their integration is disabled.
        \\Invoking one reports the required build capability.
        \\.SH COMMANDS
        \\
    );
    try write_man_commands(writer, &schema.root);
    try writer.writeAll(
        \\.SH GLOBAL OPTIONS
        \\
    );
    for (schema.root.options) |*option| try write_man_option(writer, option);
    try writer.writeAll(
        \\.SH VERSION
        \\
    );
    try writer.print("{s}\n", .{version});
}

fn write_man_commands(writer: *std.Io.Writer, command: *const schema.Command) !void {
    for (command.subcommands) |*child| {
        if (child.hidden) continue;
        try writer.writeAll(".TP\n.B ");
        try write_command_path(writer, child);
        for (child.positionals) |positional|
            try writer.print(" {s}", .{positional.name});
        try writer.writeByte('\n');
        try write_roff_text(writer, child.summary);
        if (child.requirement) |requirement| {
            try writer.writeAll(" Requires ");
            try write_roff_text(writer, requirement);
            try writer.writeByte('.');
        }
        try writer.writeByte('\n');
        for (child.options) |*option| try write_man_option(writer, option);
        try write_man_commands(writer, child);
    }
}

fn write_man_option(writer: *std.Io.Writer, option: *const schema.Option) !void {
    try writer.writeAll(".RS\n.TP\n.B ");
    if (option.short_name) |short_name|
        try writer.print("\\-{c}, ", .{short_name});
    if (option.negatable)
        try writer.writeAll("\\-\\-[no\\-]")
    else
        try writer.writeAll("\\-\\-");
    try write_roff_text(writer, option.long_name);
    if (option.value_name) |value_name|
        try writer.print(" {s}", .{value_name});
    try writer.writeByte('\n');
    try write_roff_text(writer, option.description);
    try writer.writeAll("\n.RE\n");
}

fn write_roff_text(writer: *std.Io.Writer, text: []const u8) !void {
    for (text) |byte| {
        if (byte == '-') try writer.writeByte('\\');
        try writer.writeByte(byte);
    }
}

/// Render Bash completion backed by the static command schema.
pub fn write_bash_completion(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\_zigrad()
        \\{
        \\    local cur prev path word i candidates
        \\    cur="${COMP_WORDS[COMP_CWORD]}"
        \\    prev="${COMP_WORDS[COMP_CWORD-1]}"
        \\    path=""
        \\    for ((i=1; i<COMP_CWORD; i++)); do
        \\        word="${COMP_WORDS[i]}"
        \\        [[ "$word" == -* ]] && continue
        \\        case "$path|$word" in
        \\
    );
    try write_bash_transitions(writer, &schema.root);
    try writer.writeAll(
        \\        esac
        \\    done
        \\    case "$path|$prev" in
        \\
    );
    try write_bash_value_cases(writer, &schema.root);
    try writer.writeAll(
        \\    esac
        \\    if [[ -z "$candidates" ]]; then
        \\        case "$path" in
        \\
    );
    try write_bash_cases(writer, &schema.root);
    try writer.writeAll(
        \\        esac
        \\    fi
        \\    COMPREPLY=( $(compgen -W "$candidates" -- "$cur") )
        \\    [[ ${#COMPREPLY[@]} -eq 0 ]] && compopt -o default
        \\}
        \\complete -F _zigrad zigrad
        \\
    );
}

fn write_bash_transitions(writer: *std.Io.Writer, command: *const schema.Command) !void {
    for (command.subcommands) |*child| {
        try writer.writeAll("            \"");
        try write_command_path_without_root(writer, command);
        try writer.writeByte('|');
        try writer.writeAll(child.name);
        try writer.writeAll("\") path=\"");
        try write_command_path_without_root(writer, child);
        try writer.writeAll("\" ;;\n");
        try write_bash_transitions(writer, child);
    }
}

fn write_bash_value_cases(writer: *std.Io.Writer, command: *const schema.Command) !void {
    for (command.options) |option| {
        if (option.choices.len == 0) continue;
        try writer.writeAll("        \"");
        try write_command_path_without_root(writer, command);
        try writer.print("|--{s}\") candidates=\"", .{option.long_name});
        try write_choices(writer, option.choices);
        try writer.writeAll("\" ;;\n");
    }
    for (command.subcommands) |*child| try write_bash_value_cases(writer, child);
}

fn write_bash_cases(writer: *std.Io.Writer, command: *const schema.Command) !void {
    try writer.writeAll("        \"");
    try write_command_path_without_root(writer, command);
    try writer.writeAll("\") candidates=\"");
    try write_completion_candidates(writer, command);
    try writer.writeAll("\" ;;\n");
    for (command.subcommands) |*child| try write_bash_cases(writer, child);
}

/// Render Zsh completion backed by the static command schema.
pub fn write_zsh_completion(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\#compdef zigrad
        \\local path word prev candidates
        \\prev="${words[$CURRENT-1]}"
        \\path=""
        \\for word in ${words[2,$CURRENT-1]}; do
        \\  [[ "$word" == -* ]] && continue
        \\  case "$path|$word" in
        \\
    );
    try write_zsh_transitions(writer, &schema.root);
    try writer.writeAll(
        \\  esac
        \\done
        \\case "$path|$prev" in
        \\
    );
    try write_zsh_value_cases(writer, &schema.root);
    try writer.writeAll(
        \\esac
        \\if [[ -z "$candidates" ]]; then
        \\  case "$path" in
        \\
    );
    try write_zsh_cases(writer, &schema.root);
    try writer.writeAll(
        \\  esac
        \\fi
        \\compadd -- ${(z)candidates}
        \\
    );
}

fn write_zsh_transitions(writer: *std.Io.Writer, command: *const schema.Command) !void {
    for (command.subcommands) |*child| {
        try writer.writeAll("    \"");
        try write_command_path_without_root(writer, command);
        try writer.writeByte('|');
        try writer.writeAll(child.name);
        try writer.writeAll("\") path=\"");
        try write_command_path_without_root(writer, child);
        try writer.writeAll("\" ;;\n");
        try write_zsh_transitions(writer, child);
    }
}

fn write_zsh_value_cases(writer: *std.Io.Writer, command: *const schema.Command) !void {
    for (command.options) |option| {
        if (option.choices.len == 0) continue;
        try writer.writeAll("  \"");
        try write_command_path_without_root(writer, command);
        try writer.print("|--{s}\") candidates=\"", .{option.long_name});
        try write_choices(writer, option.choices);
        try writer.writeAll("\" ;;\n");
    }
    for (command.subcommands) |*child| try write_zsh_value_cases(writer, child);
}

fn write_zsh_cases(writer: *std.Io.Writer, command: *const schema.Command) !void {
    try writer.writeAll("  \"");
    try write_command_path_without_root(writer, command);
    try writer.writeAll("\") candidates=\"");
    try write_completion_candidates(writer, command);
    try writer.writeAll("\" ;;\n");
    for (command.subcommands) |*child| try write_zsh_cases(writer, child);
}

/// Render Fish completion backed by the static command schema.
pub fn write_fish_completion(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\function __zigrad_command_path
        \\    set -l path
        \\    set -l words (commandline -opc)
        \\    if test (count $words) -gt 0
        \\        set -e words[1]
        \\    end
        \\    for word in $words
        \\        string match -q -- '-*' $word; and continue
        \\        switch "$path|$word"
        \\
    );
    try write_fish_transitions(writer, &schema.root);
    try writer.writeAll(
        \\        end
        \\    end
        \\    echo $path
        \\end
        \\
        \\function __zigrad_at
        \\    set -l actual (__zigrad_command_path)
        \\    set -l expected (string join ' ' $argv)
        \\    test "$actual" = "$expected"
        \\end
        \\
        \\function __zigrad_has_option
        \\    for word in (commandline -opc)
        \\        string match -q -- '-*' $word; and return 0
        \\    end
        \\    return 1
        \\end
        \\
        \\complete -c zigrad -f
        \\
    );
    try write_fish_commands(writer, &schema.root);
}

fn write_fish_transitions(writer: *std.Io.Writer, command: *const schema.Command) !void {
    for (command.subcommands) |*child| {
        try writer.writeAll("            case '");
        try write_command_path_without_root(writer, command);
        try writer.writeByte('|');
        try writer.writeAll(child.name);
        try writer.writeAll("'\n                set path '");
        try write_command_path_without_root(writer, child);
        try writer.writeAll("'\n");
        try write_fish_transitions(writer, child);
    }
}

fn write_fish_commands(writer: *std.Io.Writer, command: *const schema.Command) !void {
    for (command.subcommands) |*child| {
        try writer.writeAll("complete -c zigrad -n '__zigrad_at");
        if (command != &schema.root) {
            try writer.writeByte(' ');
            try write_command_path_without_root(writer, command);
        }
        try writer.print("' -a '{s}' -d '", .{child.name});
        try write_single_quoted(writer, child.summary);
        try writer.writeAll("'\n");
        try write_fish_commands(writer, child);
    }

    for (command.options) |*option| {
        try writer.writeAll("complete -c zigrad -n '__zigrad_at");
        if (command != &schema.root) {
            try writer.writeByte(' ');
            try write_command_path_without_root(writer, command);
        }
        try writer.writeByte('\'');
        try writer.print(" -l {s}", .{option.long_name});
        if (option.short_name) |short_name|
            try writer.print(" -s {c}", .{short_name});
        if (option.value_name != null)
            try writer.writeAll(" -r");
        if (option.value_name) |value_name| {
            if (std.mem.eql(u8, value_name, "PATH")) {
                try writer.writeAll(" -F");
            } else {
                try writer.writeAll(" -f");
            }
        }
        if (option.choices.len != 0) {
            try writer.writeAll(" -a '");
            try write_choices(writer, option.choices);
            try writer.writeByte('\'');
        }
        try writer.writeAll(" -d '");
        try write_single_quoted(writer, option.description);
        try writer.writeAll("'\n");

        if (option.negatable) {
            try writer.writeAll("complete -c zigrad -n '__zigrad_at");
            if (command != &schema.root) {
                try writer.writeByte(' ');
                try write_command_path_without_root(writer, command);
            }
            try writer.print("' -l no-{s} -d 'Disable: ", .{option.long_name});
            try write_single_quoted(writer, option.description);
            try writer.writeAll("'\n");
        }
    }

    if (command != &schema.root) {
        try writer.writeAll("complete -c zigrad -n '__zigrad_at ");
        try write_command_path_without_root(writer, command);
        try writer.writeAll("' -l help -s h -d 'Show help for the selected command'\n");
    }
    if (command.positionals.len != 0) {
        try writer.writeAll("complete -c zigrad -n '__zigrad_at ");
        try write_command_path_without_root(writer, command);
        try writer.writeAll("; and not __zigrad_has_option' -F\n");
    }
}

/// Render PowerShell completion backed by the static command schema.
pub fn write_powershell_completion(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\Register-ArgumentCompleter -Native -CommandName zigrad -ScriptBlock {
        \\    param($wordToComplete, $commandAst, $cursorPosition)
        \\    $words = @($commandAst.CommandElements | Select-Object -Skip 1 | ForEach-Object { $_.Extent.Text })
        \\    if ($wordToComplete.Length -gt 0) {
        \\        $words = @($words | Select-Object -SkipLast 1)
        \\    }
        \\    $path = ''
        \\    foreach ($word in $words) {
        \\        if ($word.StartsWith('-')) { continue }
        \\        switch ("$path|$word") {
        \\
    );
    try write_powershell_transitions(writer, &schema.root);
    try writer.writeAll(
        \\        }
        \\    }
        \\    $previous = if ($words.Count -gt 0) { $words[-1] } else { '' }
        \\    $candidates = switch ("$path|$previous") {
        \\
    );
    try write_powershell_value_cases(writer, &schema.root);
    try writer.writeAll(
        \\        default {
        \\            switch ($path) {
        \\
    );
    try write_powershell_cases(writer, &schema.root);
    try writer.writeAll(
        \\                default { @() }
        \\            }
        \\        }
        \\    }
        \\    $candidates | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
        \\        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
        \\    }
        \\}
        \\
    );
}

fn write_powershell_value_cases(writer: *std.Io.Writer, command: *const schema.Command) !void {
    for (command.options) |option| {
        if (option.choices.len == 0) continue;
        try writer.writeAll("        '");
        try write_command_path_without_root(writer, command);
        try writer.print("|--{s}' {{ @(", .{option.long_name});
        for (option.choices, 0..) |choice, index| {
            if (index != 0) try writer.writeAll(", ");
            try writer.print("'{s}'", .{choice});
        }
        try writer.writeAll(") }\n");
    }
    for (command.subcommands) |*child|
        try write_powershell_value_cases(writer, child);
}

fn write_powershell_transitions(writer: *std.Io.Writer, command: *const schema.Command) !void {
    for (command.subcommands) |*child| {
        try writer.writeAll("            '");
        try write_command_path_without_root(writer, command);
        try writer.writeByte('|');
        try writer.writeAll(child.name);
        try writer.writeAll("' { $path = '");
        try write_command_path_without_root(writer, child);
        try writer.writeAll("' }\n");
        try write_powershell_transitions(writer, child);
    }
}

fn write_powershell_cases(writer: *std.Io.Writer, command: *const schema.Command) !void {
    try writer.writeAll("        '");
    try write_command_path_without_root(writer, command);
    try writer.writeAll("' { @(");
    var first = true;
    for (command.subcommands) |child| {
        if (child.hidden) continue;
        if (!first) try writer.writeAll(", ");
        first = false;
        try writer.print("'{s}'", .{child.name});
    }
    for (command.options) |option| {
        if (!first) try writer.writeAll(", ");
        first = false;
        try writer.print("'--{s}'", .{option.long_name});
        if (option.negatable)
            try writer.print(", '--no-{s}'", .{option.long_name});
    }
    if (command != &schema.root) {
        if (!first) try writer.writeAll(", ");
        try writer.writeAll("'--help', '-h'");
    }
    try writer.writeAll(") }\n");
    for (command.subcommands) |*child| try write_powershell_cases(writer, child);
}

fn write_completion_candidates(writer: *std.Io.Writer, command: *const schema.Command) !void {
    var first = true;
    for (command.subcommands) |child| {
        if (child.hidden) continue;
        if (!first) try writer.writeByte(' ');
        first = false;
        try writer.writeAll(child.name);
    }
    for (command.options) |option| {
        if (!first) try writer.writeByte(' ');
        first = false;
        try writer.print("--{s}", .{option.long_name});
        if (option.negatable)
            try writer.print(" --no-{s}", .{option.long_name});
    }
    if (command != &schema.root) {
        if (!first) try writer.writeByte(' ');
        try writer.writeAll("--help -h");
    }
}

fn write_choices(writer: *std.Io.Writer, choices: []const []const u8) !void {
    for (choices, 0..) |choice, index| {
        if (index != 0) try writer.writeByte(' ');
        try writer.writeAll(choice);
    }
}

/// Write a command's complete path from the `zigrad` root.
pub fn write_command_path(writer: *std.Io.Writer, target: *const schema.Command) !void {
    var path: [max_command_depth]*const schema.Command = undefined;
    const len = command_path(target.id, &path) orelse unreachable;
    for (path[0..len], 0..) |command, index| {
        if (index != 0) try writer.writeByte(' ');
        try writer.writeAll(command.name);
    }
}

fn write_command_path_without_root(writer: *std.Io.Writer, target: *const schema.Command) !void {
    var path: [max_command_depth]*const schema.Command = undefined;
    const len = command_path(target.id, &path) orelse unreachable;
    for (path[1..len], 0..) |command, index| {
        if (index != 0) try writer.writeByte(' ');
        try writer.writeAll(command.name);
    }
}

fn command_path(
    target: schema.CommandId,
    path: *[max_command_depth]*const schema.Command,
) ?usize {
    var len: usize = 0;
    if (!find_path(&schema.root, target, path, &len)) return null;
    return len;
}

fn find_path(
    command: *const schema.Command,
    target: schema.CommandId,
    path: *[max_command_depth]*const schema.Command,
    len: *usize,
) bool {
    if (len.* == path.len) return false;
    path[len.*] = command;
    len.* += 1;
    if (command.id == target) return true;
    for (command.subcommands) |*child| {
        if (find_path(child, target, path, len)) return true;
    }
    len.* -= 1;
    return false;
}

fn write_single_quoted(writer: *std.Io.Writer, text: []const u8) !void {
    for (text) |byte| {
        if (byte == '\'') try writer.writeAll("\\'");
        try writer.writeByte(byte);
    }
}

test "subcommand help labels the inherited help option" {
    const command = schema.find_by_id(.pr) orelse unreachable;
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();

    try write_help(&output.writer, command);
    try std.testing.expect(std.mem.indexOf(
        u8,
        output.written(),
        "\nOptions:\n  -h, --help",
    ) != null);
}

test "help and metadata render from the command schema" {
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();

    const command = schema.find_by_id(.demo_train).?;
    try write_help(&output.writer, command);
    try std.testing.expect(std.mem.indexOf(u8, output.written(), "zigrad demo train") != null);
    try std.testing.expect(std.mem.indexOf(u8, output.written(), "--steps") != null);
}
