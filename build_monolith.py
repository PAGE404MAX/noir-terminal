#!/usr/bin/env python3
"""
build_monolith.py — compress the entire noir-terminal suite into MONOLITH.py

Usage:
    python build_monolith.py

Reads every node below, compresses each with zlib-9 / bz2-9 / lzma
(whichever wins), base85-encodes it, and writes MONOLITH.py — a single
runnable artifact that can execute any node, print any source, extract
everything back to disk, and verify its own integrity.

Regenerate any time the sources change. the monolith forgets nothing.
"""

from __future__ import annotations

import base64
import bz2
import lzma
import zlib
from pathlib import Path

ROOT = Path(__file__).parent

# name -> (kind, origin path)
#   kind "run"     = executable node (has a launcher entry)
#   kind "archive" = inert payload (view / extract only)
NODES = {
    "daw":      ("run",     "Nill (Open Source DAW).py"),
    "paranoia": ("run",     "full code"),
    "scope":    ("run",     "Visual"),
    "serum":    ("run",     "testwavetable.py"),
    "daw-test": ("archive", "DAW TEST"),
    "drum-import": ("archive", "DRUM IMPORT CODE"),
    "drum-logic":  ("archive", "ONLY GET THE DRUM LOGIC"),
    "fixes":    ("archive", "FIXESNEEDED"),
    "readme":   ("archive", "README.md"),
    "license":  ("archive", "LICENSE"),
}

TEMPLATE = r'''#!/usr/bin/env python3
"""
MONOLITH \u2205 — noir-terminal, entire, compressed, one file.

every node of the suite lives inside this artifact as a compressed
payload. nothing was deleted. the monolith forgets nothing.

run a node:
    python MONOLITH.py               menu
    python MONOLITH.py daw           NILL \u2205 DAW          (PySide6)
    python MONOLITH.py paranoia      PARANOIA_UI         (tkinter)
    python MONOLITH.py scope         NILL_SCOPE          (pygame)
    python MONOLITH.py serum         SERUM_2 \u2205 GHOST     (tkinter)
    python MONOLITH.py --nill-osc    DAW self-respawn passthrough
    python MONOLITH.py --nill-visualizer

inspect:
    python MONOLITH.py list          payloads + compression stats
    python MONOLITH.py source <name> print original source
    python MONOLITH.py extract [dir] write everything back to disk
    python MONOLITH.py verify        decompress + compile check

compression: best of zlib-9 / bz2-9 / lzma per payload, base85-armored.
strictly greyscale. strictly one file. trust nothing.
"""

from __future__ import annotations

import base64
import bz2
import lzma
import sys
import zlib
from pathlib import Path

# ============================ PAYLOADS ============================

PAYLOADS = {
@@PAYLOAD_ENTRIES@@
}

STATS = {
    "raw_bytes": @@RAW_TOTAL@@,
    "embedded_bytes": @@PACKED_TOTAL@@,
    "ratio": @@RATIO@@,
}

_DECOMPRESS = {
    "zlib": zlib.decompress,
    "bz2": bz2.decompress,
    "lzma": lzma.decompress,
}


def raw(name: str) -> bytes:
    kind, origin, method, data, _n = PAYLOADS[name]
    return _DECOMPRESS[method](base64.b85decode(data))


# ============================ LAUNCHERS ============================

def _fresh_ns(name: str) -> dict:
    return {
        "__name__": "monolith_node",
        "__file__": str(Path(PAYLOADS[name][1]).name),
    }


def run_daw(extra: list) -> None:
    ns = _fresh_ns("daw")
    exec(compile(raw("daw"), "<daw>", "exec"), ns)
    ns["main"]()


def run_paranoia(extra: list) -> None:
    ns = _fresh_ns("paranoia")
    exec(compile(raw("paranoia"), "<paranoia>", "exec"), ns)
    root = ns["tk"].Tk()
    ns["ParanoidUI"](root)
    root.mainloop()


def run_scope(extra: list) -> None:
    ns = _fresh_ns("scope")
    try:
        exec(compile(raw("scope"), "<scope>", "exec"), ns)
    except SystemExit as exc:
        print(f"[scope] exited with code {exc.code} — no loopback device. it denies involvement.")


def run_serum(extra: list) -> None:
    ns = _fresh_ns("serum")
    exec(compile(raw("serum"), "<serum>", "exec"), ns)
    root = ns["tk"].Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    ns["SerumApp"](root)
    root.mainloop()


RUNNERS = {
    "daw": run_daw,
    "paranoia": run_paranoia,
    "scope": run_scope,
    "serum": run_serum,
}

PASSTHROUGH = {"--nill-osc": "daw", "--nill-visualizer": "daw"}


# ============================ COMMANDS ============================

def cmd_list() -> None:
    print("MONOLITH \u2205 — payload manifest")
    print(f"{'node':<13}{'kind':<9}{'method':<7}{'raw':>9}{'packed':>9}  origin")
    for name, (kind, origin, method, data, nbytes) in PAYLOADS.items():
        print(f"{name:<13}{kind:<9}{method:<7}{nbytes:>9}{len(data):>9}  {origin}")
    s = STATS
    print(f"\ntotal: {s['raw_bytes']:,} bytes raw -> {s['embedded_bytes']:,} bytes embedded "
          f"(x{s['ratio']:.2f}). color: none detected.")


def cmd_source(name: str) -> None:
    if name not in PAYLOADS:
        print(f"unknown node: {name} — try: {', '.join(PAYLOADS)}"); return
    sys.stdout.write(raw(name).decode("utf-8", errors="replace"))


def cmd_extract(dest: str) -> None:
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    for name, (kind, origin, *_rest) in PAYLOADS.items():
        target = out / Path(origin).name
        target.write_bytes(raw(name))
        print(f"[ok] {target}  ({PAYLOADS[name][4]:,} bytes)")
    print(f"\neverything written to {out} — identical to what was compressed. it remembers.")


def cmd_verify() -> None:
    failures = 0
    for name, (kind, *_rest) in PAYLOADS.items():
        try:
            data = raw(name)
            note = "decompress ok"
            if kind == "run":
                compile(data, f"<{name}>", "exec")
                note += ", compiles ok"
            print(f"[ok] {name:<12} {note} ({len(data):,} bytes)")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {name}: {exc}")
    print("\nintegrity: " + ("HOLDING \u2205" if not failures else f"{failures} BREACH(ES)"))
    sys.exit(1 if failures else 0)


def cmd_menu() -> None:
    print("""
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551  M O N O L I T H \u2205   —   every node, one file, compressed \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d""")
    print(f"  payloads: {len(PAYLOADS)}   raw: {STATS['raw_bytes']:,}B   "
          f"embedded: {STATS['embedded_bytes']:,}B   x{STATS['ratio']:.2f}\n")
    print("  1  NILL \u2205 DAW        (PySide6 DAW + embedded OSC + scope)")
    print("  2  PARANOIA_UI       (tkinter paranoia suite)")
    print("  3  NILL_SCOPE        (pygame signal ghost)")
    print("  4  SERUM_2 \u2205 GHOST   (tkinter wavetable synth)")
    print("  5  list payloads")
    print("  6  extract everything")
    print("  0  exit              (it will deny you left)\n")
    try:
        choice = input("node> ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    table = {"1": "daw", "2": "paranoia", "3": "scope", "4": "serum",
             "5": "list", "6": "extract"}
    if choice == "0" or not choice:
        return
    if choice == "5":
        cmd_list(); return
    if choice == "6":
        cmd_extract("noir_extracted"); return
    node = table.get(choice)
    if node is None:
        print("unrecognized. the monolith noted the attempt.")
        return
    run_node(node, [])


def run_node(node: str, extra: list) -> None:
    runner = RUNNERS.get(node)
    if runner is None:
        print(f"not a runnable node: {node}")
        return
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0]] + extra
    try:
        runner(extra)
    finally:
        sys.argv = saved_argv


def main() -> None:
    args = sys.argv[1:]
    if not args:
        cmd_menu(); return
    first = args[0]
    if first in PASSTHROUGH:                 # the DAW re-launches itself
        run_node(PASSTHROUGH[first], args); return
    if first in RUNNERS:
        run_node(first, args[1:]); return
    if first == "list":
        cmd_list(); return
    if first == "source":
        if len(args) < 2: print("usage: source <node>"); return
        cmd_source(args[1]); return
    if first == "extract":
        cmd_extract(args[1] if len(args) > 1 else "noir_extracted"); return
    if first == "verify":
        cmd_verify(); return
    print(f"unknown command: {first} — daw | paranoia | scope | serum | list | source | extract | verify")


if __name__ == "__main__":
    main()
'''


def best_compress(data: bytes):
    candidates = [
        ("zlib", zlib.compress(data, 9)),
        ("bz2", bz2.compress(data, 9)),
        ("lzma", lzma.compress(data, preset=9)),
    ]
    method, blob = min(candidates, key=lambda c: len(c[1]))
    return method, base64.b85encode(blob).decode("ascii")


def chunk(s: str, width: int = 110) -> str:
    parts = [s[i:i + width] for i in range(0, len(s), width)]
    return '"\n        "'.join(parts)


def main() -> None:
    entries, raw_total, packed_total = [], 0, 0
    for name, (kind, origin) in NODES.items():
        data = (ROOT / origin).read_bytes()
        method, b85 = best_compress(data)
        raw_total += len(data)
        packed_total += len(b85)
        print(f"{name:<13} {len(data):>8,}B -> {len(b85):>8,}B via {method}  ({origin})")
        entries.append(
            f'    "{name}": ("{kind}", "{origin}", "{method}",\n'
            f'        "{chunk(b85)}", {len(data)}),'
        )

    monolith = (TEMPLATE
                .replace("@@PAYLOAD_ENTRIES@@", "\n".join(entries))
                .replace("@@RAW_TOTAL@@", str(raw_total))
                .replace("@@PACKED_TOTAL@@", str(packed_total))
                .replace("@@RATIO@@", f"{raw_total / packed_total:.3f}"))
    out = ROOT / "MONOLITH.py"
    out.write_text(monolith, encoding="utf-8")
    print(f"\n[ok] {out}  ({out.stat().st_size:,} bytes total)")
    print(f"     {raw_total:,}B raw -> {packed_total:,}B embedded (x{raw_total / packed_total:.2f})")


if __name__ == "__main__":
    main()
