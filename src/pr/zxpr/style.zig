const std = @import("std");

const Writer = std.Io.Writer;
const tty = std.Io.tty;

pub const Symbols = struct {
    region_start: []const u8,
    region_gutter: []const u8,
    region_end: []const u8,

    pub fn ascii() Symbols {
        return .{ .region_start = ">", .region_gutter = "|", .region_end = "<" };
    }

    pub fn unicode() Symbols {
        return .{ .region_start = "┌", .region_gutter = "│", .region_end = "└" };
    }
};

pub const ColorMode = enum {
    auto,
    always,
    never,
    truecolor,
};

pub const Palette = enum {
    default,
    alt,
    nord,
    gruvbox_material,
    flat_dark,
    catppuccin,
    tokyonight,
};

pub const Rgb = struct {
    r: u8,
    g: u8,
    b: u8,
};

pub const RoleColor = struct {
    const Self = @This();
    ansi: tty.Color,
    rgb: ?Rgb = null,

    fn init(ansi: tty.Color, value: ?Rgb) Self {
        return .{ .ansi = ansi, .rgb = value };
    }
};

pub const Theme = struct {
    keyword: RoleColor,
    section: RoleColor,
    var_name: RoleColor,
    type_name: RoleColor,
    op_name: RoleColor,
    comment: RoleColor,
    region: RoleColor,

    pub fn default() Theme {
        return .{
            .keyword = .{ .ansi = .bright_yellow, .rgb = rgb(0xFF8700) },
            .section = .{ .ansi = .bright_yellow, .rgb = rgb(0xD8A657) },
            .var_name = .{ .ansi = .white },
            .type_name = .{ .ansi = .magenta },
            .op_name = .{ .ansi = .green },
            .comment = .{ .ansi = .bright_black, .rgb = rgb(0x565F89) },
            .region = .{ .ansi = .blue },
        };
    }
};

pub const Config = struct {
    symbols: Symbols = Symbols.ascii(),
    color_mode: ColorMode = .never,
    tty_config: ?tty.Config = null,
    theme: Theme = Theme.default(),
    shape_format: ShapeFormat = .dtype_suffix,
    include_dtype_attrs: bool = false,
};

pub const ConfigMode = enum {
    plain,
    auto_stdout,
};

pub const ConfigOpts = struct {
    symbols: ?Symbols = null,
    color_mode: ?ColorMode = null,
    tty_config: ?tty.Config = null,
    palette: ?Palette = null,
    theme: ?Theme = null,
    shape_format: ?ShapeFormat = null,
    include_dtype_attrs: ?bool = null,
};

pub const ShapeFormat = enum {
    dtype_suffix,
};

pub fn config(mode: ConfigMode, opts: ConfigOpts) Config {
    var cfg: Config = switch (mode) {
        .plain => .{
            .symbols = Symbols.ascii(),
            .color_mode = .never,
            .tty_config = null,
            .theme = Theme.default(),
            .shape_format = .dtype_suffix,
            .include_dtype_attrs = false,
        },
        .auto_stdout => .{
            .symbols = Symbols.unicode(),
            .color_mode = .auto,
            .tty_config = tty.Config.detect(std.fs.File.stdout()),
            .theme = Theme.default(),
            .shape_format = .dtype_suffix,
            .include_dtype_attrs = false,
        },
    };

    if (opts.symbols) |symbols| cfg.symbols = symbols;
    if (opts.color_mode) |color_mode| cfg.color_mode = color_mode;
    if (opts.tty_config) |tty_config| cfg.tty_config = tty_config;
    if (opts.palette) |palette| cfg.theme = theme_for_palette(palette);
    if (opts.theme) |theme| cfg.theme = theme;
    if (opts.shape_format) |shape_format| cfg.shape_format = shape_format;
    if (opts.include_dtype_attrs) |include_dtype_attrs| cfg.include_dtype_attrs = include_dtype_attrs;

    return cfg;
}

/// I am bad with colors, PRs very welcome here
pub fn theme_for_palette(palette: Palette) Theme {
    const rc = RoleColor.init;
    return switch (palette) {
        .default => Theme.default(),
        .alt => .{
            // .keyword = rc(.bright_magenta, rgb(0xD3869B)),
            .keyword = rc(.bright_yellow, rgb(0xFF8700)),
            .section = rc(.bright_yellow, rgb(0xD8A657)),
            // .section = rc(.bright_yellow, rgb(0xFF9E64)),
            .var_name = rc(.white, rgb(0xECEFF1)),
            .type_name = rc(.bright_magenta, rgb(0xD3869B)),
            // .type_name = rc(.bright_yellow, rgb(0xFF9E64)),
            .op_name = rc(.green, null),
            .comment = rc(.bright_black, rgb(0x565F89)),
            // .region = rc(.bright_yellow, rgb(0xFF8700)),
            // .region = rc(.bright_magenta, rgb(0xD3869B)),
            .region = rc(.blue, null),
            // .region = rc(.bright_yellow, rgb(0xFF9E64)),
        },
        .nord => .{
            .keyword = rc(.cyan, rgb(0x88C0D0)),
            .section = rc(.bright_cyan, rgb(0x8FBCBB)),
            .var_name = rc(.white, rgb(0xE5E9F0)),
            .type_name = rc(.bright_magenta, rgb(0xB48EAD)),
            .op_name = rc(.bright_green, rgb(0xA3BE8C)),
            .comment = rc(.bright_black, rgb(0x4C566A)),
            .region = rc(.bright_blue, rgb(0x5E81AC)),
        },
        .gruvbox_material => .{
            .keyword = rc(.bright_yellow, rgb(0xD8A657)),
            .section = rc(.bright_red, rgb(0xE78A4E)),
            .var_name = rc(.white, rgb(0xEBDBB2)),
            .type_name = rc(.bright_blue, rgb(0x7DAEA3)),
            .op_name = rc(.bright_green, rgb(0xA9B665)),
            .comment = rc(.bright_black, rgb(0x928374)),
            .region = rc(.bright_magenta, rgb(0xD3869B)),
        },
        .flat_dark => .{
            .keyword = rc(.bright_cyan, rgb(0x4FC3F7)),
            .section = rc(.bright_yellow, rgb(0xFFD54F)),
            .var_name = rc(.white, rgb(0xECEFF1)),
            .type_name = rc(.bright_magenta, rgb(0xCE93D8)),
            .op_name = rc(.bright_green, rgb(0xAED581)),
            .comment = rc(.bright_black, rgb(0x757575)),
            .region = rc(.bright_blue, rgb(0x64B5F6)),
        },
        .catppuccin => .{
            .keyword = rc(.bright_magenta, rgb(0xCBA6F7)),
            .section = rc(.bright_blue, rgb(0x89B4FA)),
            .var_name = rc(.white, rgb(0xCDD6F4)),
            .type_name = rc(.bright_yellow, rgb(0xF9E2AF)),
            .op_name = rc(.bright_green, rgb(0xA6E3A1)),
            .comment = rc(.bright_black, rgb(0x6C7086)),
            .region = rc(.bright_cyan, rgb(0x94E2D5)),
        },
        .tokyonight => .{
            .keyword = rc(.bright_blue, rgb(0x7AA2F7)),
            .section = rc(.cyan, rgb(0x7DCFFF)),
            .var_name = rc(.white, rgb(0xC0CAF5)),
            .type_name = rc(.bright_magenta, rgb(0xBB9AF7)),
            .op_name = rc(.bright_green, rgb(0x9ECE6A)),
            .comment = rc(.bright_black, rgb(0x565F89)),
            .region = rc(.bright_yellow, rgb(0xFF9E64)),
        },
    };
}

fn rgb(hex: u32) Rgb {
    return .{
        .r = @intCast((hex >> 16) & 0xff),
        .g = @intCast((hex >> 8) & 0xff),
        .b = @intCast(hex & 0xff),
    };
}

pub const Styler = struct {
    writer: *Writer,
    cfg: Config,

    pub fn init(writer: *Writer, cfg: Config) Styler {
        return .{ .writer = writer, .cfg = cfg };
    }

    pub fn write_keyword(self: *Styler, text: []const u8) !void {
        try self.write_colored(self.cfg.theme.keyword, text);
    }

    pub fn write_section(self: *Styler, text: []const u8) !void {
        try self.write_colored(self.cfg.theme.section, text);
    }

    pub fn write_var_name(self: *Styler, text: []const u8) !void {
        try self.write_colored(self.cfg.theme.var_name, text);
    }

    pub fn write_type_name(self: *Styler, text: []const u8) !void {
        try self.write_colored(self.cfg.theme.type_name, text);
    }

    pub fn write_op_name(self: *Styler, text: []const u8) !void {
        try self.write_colored(self.cfg.theme.op_name, text);
    }

    pub fn write_comment(self: *Styler, text: []const u8) !void {
        try self.write_colored(self.cfg.theme.comment, text);
    }

    pub fn write_region(self: *Styler, text: []const u8) !void {
        try self.write_colored(self.cfg.theme.region, text);
    }

    fn supports_color(self: *Styler) bool {
        return switch (self.cfg.color_mode) {
            .never => false,
            .always => true,
            .truecolor => self.supports_truecolor(),
            .auto => switch (self.active_tty_config()) {
                .no_color => false,
                .escape_codes, .windows_api => true,
            },
        };
    }

    fn write_colored(self: *Styler, color: RoleColor, text: []const u8) !void {
        if (!self.supports_color()) {
            try self.writer.writeAll(text);
            return;
        }
        if (self.use_truecolor()) {
            if (color.rgb) |rgb_value| {
                try self.write_truecolor(rgb_value);
                try self.writer.writeAll(text);
                try self.reset_color();
                return;
            }
        }
        const conf = self.active_tty_config();
        try tty.Config.setColor(conf, self.writer, color.ansi);
        try self.writer.writeAll(text);
        try tty.Config.setColor(conf, self.writer, .reset);
    }

    fn write_truecolor(self: *Styler, rgb_value: Rgb) !void {
        try self.writer.print("\x1b[38;2;{d};{d};{d}m", .{ rgb_value.r, rgb_value.g, rgb_value.b });
    }

    fn reset_color(self: *Styler) !void {
        try self.writer.writeAll("\x1b[0m");
    }

    fn supports_truecolor(self: *Styler) bool {
        return switch (self.active_tty_config()) {
            .escape_codes => true,
            else => false,
        };
    }

    fn use_truecolor(self: *Styler) bool {
        if (!self.supports_truecolor()) return false;
        if (self.cfg.color_mode == .truecolor) return true;
        if (self.cfg.color_mode != .auto) return false;
        var buf: [128]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        return std.process.hasNonEmptyEnvVar(fba.allocator(), "ZG_TRUECOLOR") catch false;
    }

    fn active_tty_config(self: *Styler) tty.Config {
        const conf = self.cfg.tty_config orelse tty.Config.detect(std.fs.File.stdout());
        if (self.cfg.color_mode != .always) return conf;
        return switch (conf) {
            .no_color => .escape_codes,
            else => conf,
        };
    }
};
