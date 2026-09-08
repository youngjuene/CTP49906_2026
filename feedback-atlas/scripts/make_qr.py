#!/usr/bin/env python
"""Render a tunnel URL as a QR code: to the terminal now, and to a PNG for a slide.

Separated from scripts/class.sh rather than inlined, because a Python heredoc
nested inside a shell heredoc is a delimiter collision waiting to happen -- it
already happened once while writing this.

segno is used over `qrcode` because on Python >= 3.10 it has no runtime
dependencies at all and writes PNG without Pillow.
"""

import sys

import segno


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: make_qr.py <url> <output.png>", file=sys.stderr)
        return 2
    url, png = argv[1], argv[2]
    # error="m" (~15% recovery) is the useful middle: it survives a projector's
    # glare and a phone held at an angle without inflating the module count,
    # which is what makes a code hard to scan from the back of a room.
    qr = segno.make(url, error="m")
    qr.terminal(compact=True)
    qr.save(png, scale=10, border=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
