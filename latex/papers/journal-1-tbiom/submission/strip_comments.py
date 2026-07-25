#!/usr/bin/env python3
"""Strip all LaTeX comments from .tex files IN PLACE (for a double-blind .tex upload).

Removes every % comment (respecting \\% escaped percents), including trailing
line-continuation %; pure-comment lines are deleted entirely so they do NOT turn
into blank lines (which would be spurious paragraph breaks); originally-blank lines
are preserved. This is what makes the anonymized submission safe to upload as source:
the commented-out CAMERA-READY author block (real names/affiliation/e-mail/ORCID/repo)
is removed rather than merely commented.

Usage:  python strip_comments.py manuscript/main.tex supplement/supplement.tex
"""
import sys


def strip_comment(line: str) -> str:
    """Return the code part of a line, cutting at the first unescaped %."""
    out = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c == "\\":                 # escape: copy backslash + next char verbatim
            out.append(c)
            if i + 1 < n:
                out.append(line[i + 1])
                i += 2
            else:
                i += 1
        elif c == "%":                # unescaped % -> comment starts here
            break
        else:
            out.append(c)
            i += 1
    return "".join(out)


def strip_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    out = []
    for line in lines:
        code = strip_comment(line)
        if code.strip() == "":
            if line.strip() == "":
                out.append("")        # keep an intentional blank line
            # else: a pure-comment line -> drop entirely
        else:
            out.append(code.rstrip())
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    print(f"stripped comments: {path}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        strip_file(p)
