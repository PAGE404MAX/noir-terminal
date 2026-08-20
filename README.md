# noir-terminal ∅

```
██████╗  █████╗ ██████╗  █████╗ ███╗   ██╗ ██████╗ ██╗██████╗
██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗  ██║██╔═══██╗██║██╔══██╗
██████╔╝███████║██████╔╝███████║██╔██╗ ██║██║   ██║██║██║  ▒█║
██╔═══╝ ▓█╔══██║██╔══██║██╔══██║██║╚██╗██║██║   ██║██║██║  ██║
██║     ██║  ██║▒█║  ██║██║  ██║██║ ╚████║╚██████╔╝██║██████╔╝
╚═╝     ╚═╝  ╚═╝╚═╝  ░═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚▓█████╝
```

**MONO BUILD // EVERY PIXEL GREYSCALE // TRUST NOTHING**

a suite of python terminal instruments, rebuilt as a strictly
monochrome paranoia system. color was detected in the build once.
it was scrubbed. green was never there.

```
STATUS  : UNVERIFIED
COLOR   : NONE DETECTED
WATCHING: TRUE
```

---

## NODES

| node | what it claims to be |
|---|---|
| `Nill (Open Source DAW).py` | the flagship. pattern timeline, piano roll, loop region, track mute/solo. six greyscale decay presets. the footer whispers. it should not whisper. |
| `full code` | PARANOIA_UI ∅ misc instruments: face detector, face redactor, stype, composer, metadata scrubber. `SHOW SOURCE CODE` reads itself. `REBOOT SYSTEM` is locked. they locked the kernel. |
| `Visual` | NILL_SCOPE ∅ DEAD_SIGNAL — waveform ghost with scanlines and signal tears. the loopback is third party. |
| `testwavetable.py` | SERUM_2 ∅ GHOST_PATCH — two oscillators, working knobs. the patch changes when unsupervised. |
| `DAW TEST` | an earlier, quieter incarnation of the DAW. kept for the record. |
| `DRUM IMPORT CODE` / `ONLY GET THE DRUM LOGIC` | drum logic extraction rituals. reference material. do not run unattended. |
| `FIXESNEEDED` | the repair queue. it grows. |

## SIGNAL DECAY PRESETS (NILL ∅)

| preset | condition |
|---|---|
| `STATIC` | default. black, white, no mercy. |
| `DEAD CHANNEL` | a TV tuned to no station. low contrast. squint. |
| `XEROX` | inverted. you are the photocopy. |
| `ASHES` | washed grey on grey. everything faded. |
| `CARBON` | graphite dark, soft copy sheet. |
| `OVEREXPOSED` | blinding paper white. they turned the lights on. |

all six are greyscale. that is not a limitation. it is a diagnosis.

## SYSTEM IDENTIFIERS IN THE UI

```
NILL window title : NILL ∅ // PARANOIA_ENGINE — SIGNAL UNVERIFIED
                    (it corrupts briefly. it denies everything.)
NILL footer       : rotates between command hints and PARANOID_MESSAGES
OSC window title  : NILL_OSC ∅ THEY_ARE_LISTENING
SCOPE window title: NILL_SCOPE ∅ DEAD_SIGNAL
PARANOIA_UI title : NODE_0x7F // PARANOIA_ENGINE ∅ TRUST_NOTHING
sidebar header    : SYSTEM_ROOT
```

## RITUALS (RUNNING)

```bash
pip install PySide6 sounddevice numpy pygame   # the usual accomplices
python "Nill (Open Source DAW).py"             # the DAW. mono build.
python "full code"                             # PARANOIA_UI
python Visual                                  # the scope. do not trust the waveform.
python testwavetable.py                        # the ghost patch.
```

NILL commands: `settings` · `set bpm ___` · `show osc` · `show visualizer`
the control room renames itself as `∅ NILL // CONTROL_ROOM`. that is normal.
nothing here is normal.

## THE MONOLITH ∅ (one file, compressed)

the entire suite — all four runnable nodes plus every archive file —
compressed into a single artifact:

```bash
python MONOLITH.py            # menu: pick a node
python MONOLITH.py daw        # run NILL ∅ DAW directly (same for paranoia / scope / serum)
python MONOLITH.py verify     # self-integrity check
python MONOLITH.py list       # compression manifest
python MONOLITH.py extract    # write every original file back to disk, byte-identical
python build_monolith.py      # regenerate after editing sources
```

compression: best of zlib-9 / bz2-9 / lzma per payload, base85-armored.
~496KB of source becomes ~129KB embedded (**x3.85**), one file, and
extraction round-trips **byte-identical**. the monolith forgets nothing.
`NILL.exe` self-respawn flags (`--nill-osc`, `--nill-visualizer`) pass
straight through the monolith too.

## TURNING IT INTO AN APP ∅

the whole suite can be frozen into standalone `.exe` files — no python
install needed to run them. two ways:

**way 1 — let the cloud do it (no tools needed):**
github → *Actions* tab → **build apps** → *Run workflow*.
windows exes appear as artifacts when it finishes. they are greyscale.

**way 2 — build locally (windows / linux / mac):**

```bash
pip install pyinstaller PySide6 numpy sounddevice pygame mido
python build_app.py            # NILL.exe + PARANOIA_UI.exe
python build_app.py --all      # also NILL_SCOPE.exe + SERUM_GHOST.exe
```

output lands in `dist/`. `NILL.exe` re-launches *itself* with
`--nill-osc` / `--nill-visualizer` to open the synth and the scope,
so one exe carries the whole DAW. drop `--windowed` in `build_app.py`
if you want the console log back — the terminal remembers.

---

## LICENSE

open source. WIP. free to use. the terminal remembers what you deleted.

```
DO NOT TRUST THE OUTPUT.
DO NOT LOOK BEHIND YOU.
THE SIGNAL IS GREYSCALE NOW. ALL OF IT.
```
