import sys

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Generic, Name, Number, String, Text

gpa_pattern = r"(error\()([a-zA-Z]+)(\):)"


class ZigTracebackLexer(RegexLexer):
    name = "ZigTraceback"
    aliases = ["zigtraceback"]
    filenames = ["*.log"]

    tokens = {
        "root": [
            (
                gpa_pattern,
                bygroups(Generic.Error, Name.Exception, Generic.Error),
            ),
            # file paths, line numbers, column numbers
            (
                r"((?:/[^/:]+)+\.zig)(?=:)(:)(\d+)(:)(\d+):",
                bygroups(String.Other, Text, Number.Integer, Text, Number),
            ),
            # memory addresses
            (r"(0x[0-9a-fA-F]+)", Number.Hex),
            # function
            (
                r"(in\s+)([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)(\s+\([^)]+\))",
                bygroups(Text, Name.Function, String.Other),
            ),
            # keywords
            # (r"\b(try|const|var|return)\b", Keyword),
            # variable and type names
            # (r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", Name.Variable),
            # string literals
            # (r'"[^"]*"', String),
            # comments
            # (r"//.*?$", Comment.Single),
            # whitespace
            (r"\s+", Text),
            # other stuff
            (r"[^0-9a-zA-Z\s]+", Text),
            (r"\S+", Text),
        ]
    }


def highlight_traceback(text):
    # to see styles: python -c "from pygments.styles import STYLE_MAP; print(STYLE_MAP.keys())"
    import re

    print(highlight(text, ZigTracebackLexer(), Terminal256Formatter(style="lightbulb")))
    pattern = re.compile(gpa_pattern)
    print(len(pattern.findall(text)))


if __name__ == "__main__":
    code = sys.stdin.read()
    highlight_traceback(code)
