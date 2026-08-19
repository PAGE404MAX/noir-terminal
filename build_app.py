#!/usr/bin/env python3
"""
build_app.py — turn the noir-terminal suite into standalone .exe apps.

Usage:
    python build_app.py           # builds NILL.exe + PARANOIA_UI.exe
    python build_app.py --all     # also builds NILL_SCOPE.exe + SERUM_GHOST.exe
    python build_app.py --what-if # show the commands without running them

Requirements:
    pip install pyinstaller PySide6 numpy sounddevice pygame mido

Output lands in ./dist — each exe is self-contained (no Python install
needed to run it). NILL.exe re-launches ITSELF with --nill-osc /
--nill-visualizer flags for the synth and the scope, so one exe carries
the whole DAW suite.

Note: the embedded OSC is exec()'d from a string, so PyInstaller cannot
see its `import tkinter` — that's why tkinter is a hidden import below.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"

ICON = ROOT / "assets" / "icon.ico"

APPS = [
    {
        "name": "NILL",
        "script": "Nill (Open Source DAW).py",
        # tkinter + pygame live behind exec()/lazy imports — force them in
        "hidden": ["tkinter", "pygame", "sounddevice", "mido"],
        "always": True,
    },
    {
        "name": "PARANOIA_UI",
        "script": "full code",
        "hidden": [],
        "always": True,
    },
    {
        "name": "NILL_SCOPE",
        "script": "Visual",
        "hidden": ["sounddevice"],
        "always": False,          # scope is also inside NILL.exe
    },
    {
        "name": "SERUM_GHOST",
        "script": "testwavetable.py",
        "hidden": ["sounddevice"],
        "always": False,
    },
]


def build_all(all_apps: bool, what_if: bool) -> int:
    targets = [a for a in APPS if all_apps or a["always"]]
    DIST.mkdir(exist_ok=True)

    failures = 0
    for app in targets:
        script = ROOT / app["script"]
        if not script.exists():
            print(f"[SKIP] {app['name']}: {app['script']} not found")
            failures += 1
            continue

        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--noconfirm", "--clean",
            "--onefile", "--windowed",
            "--name", app["name"],
            "--distpath", str(DIST),
            "--workpath", str(ROOT / "build" / app["name"]),
            "--specpath", str(ROOT / "build" / app["name"]),
        ]
        if ICON.exists():
            cmd += ["--icon", str(ICON)]
        for hidden in app["hidden"]:
            cmd += ["--hidden-import", hidden]
        cmd.append(str(script))

        print(f"\n=== {app['name']}  <-  {app['script']} ===")
        print("    " + " ".join(cmd[2:]))
        if what_if:
            continue
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"[FAIL] {app['name']} exited with {result.returncode}")
            failures += 1
        else:
            out = DIST / f"{app['name']}.exe"
            size = out.stat().st_size / (1024 * 1024) if out.exists() else 0
            print(f"[ OK ] {out}  ({size:.1f} MB)")

    print()
    if what_if:
        print("what-if mode: nothing was built.")
        return 0
    if failures:
        print(f"{failures} build(s) failed. the terminal remembers.")
        return 1
    print(f"all apps built into {DIST} // color: none detected")
    return 0


if __name__ == "__main__":
    sys.exit(build_all(
        all_apps="--all" in sys.argv,
        what_if="--what-if" in sys.argv,
    ))
