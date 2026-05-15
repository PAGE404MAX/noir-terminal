#!/usr/bin/env python3
"""
Nill | FL x BeepBox Open Source DAW

A deeper hybrid rewrite inspired by:
- FL Studio-style piano roll ideas: ghost notes, scale highlighting, stamp/chord tool,
  channel-style track list, visible piano keyboard, editable note blocks.
- BeepBox-style ideas: pattern-first composition, fast grid entry, colorful chiptune-friendly
  tracks, multiple rows/tracks playing simultaneously.

Requirements:
    pip install PySide6 sounddevice numpy

Notes:
- This is a single-file desktop prototype, not a full DAW.
- If sounddevice is unavailable, the editor still works visually.
"""

from __future__ import annotations

import json
import math
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Hashable

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from PySide6.QtCore import Qt, QTimer, QRectF, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ============================ DATA MODELS ============================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BLACK_PITCH_CLASSES = {1, 3, 6, 8, 10}
SCALE_INTERVALS = {
    "Chromatic": set(range(12)),
    "Major": {0, 2, 4, 5, 7, 9, 11},
    "Minor": {0, 2, 3, 5, 7, 8, 10},
    "Pentatonic": {0, 2, 4, 7, 9},
    "Harmonic Minor": {0, 2, 3, 5, 7, 8, 11},
}
STAMP_INTERVALS = {
    "Single": [0],
    "Power": [0, 7],
    "Major": [0, 4, 7],
    "Minor": [0, 3, 7],
    "Sus2": [0, 2, 7],
    "Sus4": [0, 5, 7],
    "Maj7": [0, 4, 7, 11],
    "Min7": [0, 3, 7, 10],
}
DEFAULT_TRACK_COLORS = [
    "#00E5FF",
    "#FF4FD8",
    "#A4FF4F",
    "#FFD54F",
    "#FF7F50",
    "#8A7DFF",
    "#FF5C5C",
    "#5CFFC8",
]


@dataclass
class Note:
    pitch: int
    start: float
    duration: float
    velocity: int = 100
    muted: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Note":
        return Note(
            pitch=int(data.get("pitch", 60)),
            start=float(data.get("start", 0.0)),
            duration=float(data.get("duration", 1.0)),
            velocity=int(data.get("velocity", 100)),
            muted=bool(data.get("muted", False)),
        )


@dataclass
class Track:
    name: str
    color: str
    waveform: str = "square"
    gain: float = 0.65
    patterns: Dict[int, List[Note]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for i in range(8):
            self.patterns.setdefault(i, [])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "color": self.color,
            "waveform": self.waveform,
            "gain": self.gain,
            "patterns": {
                str(index): [note.to_dict() for note in notes]
                for index, notes in self.patterns.items()
            },
        }

    @staticmethod
    def from_dict(data: dict) -> "Track":
        patterns = {}
        raw_patterns = data.get("patterns", {})
        for key, notes in raw_patterns.items():
            patterns[int(key)] = [Note.from_dict(n) for n in notes]
        return Track(
            name=str(data.get("name", "Track")),
            color=str(data.get("color", "#00E5FF")),
            waveform=str(data.get("waveform", "square")),
            gain=float(data.get("gain", 0.65)),
            patterns=patterns,
        )


# ============================ SYNTH ENGINE ============================

class ChiptuneSynth:
    """Lightweight chiptune-ish poly synth for a prototype DAW."""

    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512) -> None:
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self._voices: Dict[Hashable, dict] = {}
        self._lock = threading.Lock()

        self.attack = 0.003
        self.decay = 0.07
        self.sustain = 0.65
        self.release = 0.14
        self.master_gain = 0.22

    @staticmethod
    def midi_to_freq(pitch: int) -> float:
        return 440.0 * (2.0 ** ((pitch - 69) / 12.0))

    def note_on(
        self,
        key: Hashable,
        pitch: int,
        velocity: float = 1.0,
        waveform: str = "square",
        gain: float = 0.6,
    ) -> None:
        with self._lock:
            self._voices[key] = {
                "pitch": pitch,
                "freq": self.midi_to_freq(pitch),
                "phase": 0.0,
                "velocity": float(max(0.0, min(1.0, velocity))),
                "waveform": waveform,
                "gain": float(max(0.0, min(1.25, gain))),
                "env": 0.0,
                "state": "attack",
            }

    def note_off(self, key: Hashable) -> None:
        with self._lock:
            voice = self._voices.get(key)
            if voice and voice["state"] != "release":
                voice["state"] = "release"

    def all_notes_off(self) -> None:
        with self._lock:
            for voice in self._voices.values():
                if voice["state"] != "release":
                    voice["state"] = "release"

    def _wave(self, waveform: str, phases: np.ndarray) -> np.ndarray:
        if waveform == "square":
            return np.where(np.sin(phases) >= 0.0, 1.0, -1.0).astype(np.float32)
        if waveform == "triangle":
            return ((2.0 / np.pi) * np.arcsin(np.sin(phases))).astype(np.float32)
        if waveform == "saw":
            return (((phases / np.pi) % 2.0) - 1.0).astype(np.float32)
        if waveform == "noise":
            return np.random.uniform(-1.0, 1.0, len(phases)).astype(np.float32)
        return np.sin(phases).astype(np.float32)

    def generate_audio(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float32)
        sr = self.sample_rate
        dt = 1.0 / sr

        attack_step = 1.0 / max(1, int(self.attack * sr))
        decay_step = (1.0 - self.sustain) / max(1, int(self.decay * sr))
        release_step = 1.0 / max(1, int(self.release * sr))

        with self._lock:
            dead_keys: List[Hashable] = []
            for key, voice in list(self._voices.items()):
                phase = float(voice["phase"])
                freq = float(voice["freq"])
                phase_inc = 2.0 * np.pi * freq * dt
                idx = np.arange(frames)
                phases = (phase + phase_inc * idx) % (2.0 * np.pi)
                wave = self._wave(str(voice["waveform"]), phases)

                env = float(voice["env"])
                state = str(voice["state"])
                env_values = np.zeros(frames, dtype=np.float32)
                for i in range(frames):
                    if state == "attack":
                        env += attack_step
                        if env >= 1.0:
                            env = 1.0
                            state = "decay"
                    elif state == "decay":
                        env -= decay_step
                        if env <= self.sustain:
                            env = self.sustain
                            state = "sustain"
                    elif state == "sustain":
                        env = self.sustain
                    elif state == "release":
                        env -= release_step
                        if env <= 0.0:
                            env = 0.0
                            state = "dead"
                    env_values[i] = env

                voice["env"] = env
                voice["state"] = state
                voice["phase"] = float((phase + phase_inc * frames) % (2.0 * np.pi))

                out += wave * env_values * float(voice["velocity"]) * float(voice["gain"])
                if state == "dead" or env <= 0.0001:
                    dead_keys.append(key)

            for key in dead_keys:
                self._voices.pop(key, None)

        out = np.tanh(out * (self.master_gain * 2.4))
        return out.astype(np.float32)


# ============================ PIANO ROLL ============================

class PianoRoll(QWidget):
    note_changed = Signal()

    KEYBOARD_W = 92
    HEADER_H = 30
    FOOTER_H = 0
    ROW_H = 18
    BEAT_W = 44
    MIN_MIDI = 36
    MAX_MIDI = 96

    def __init__(self, daw: "NillFLBeepBox") -> None:
        super().__init__()
        self.daw = daw
        self.setMinimumSize(1100, 620)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.selected_note: Optional[Note] = None
        self.drag_mode: Optional[str] = None
        self.drag_note_start_beat = 0.0
        self.drag_note_start_pitch = 60
        self.drag_origin_beat = 0.0
        self.drag_origin_pitch = 60
        self.preview_key: Optional[str] = None

        self._hover_note: Optional[Note] = None

    @property
    def total_rows(self) -> int:
        return self.MAX_MIDI - self.MIN_MIDI + 1

    def is_black(self, pitch: int) -> bool:
        return pitch % 12 in BLACK_PITCH_CLASSES

    def note_name(self, pitch: int) -> str:
        octave = (pitch // 12) - 1
        return f"{NOTE_NAMES[pitch % 12]}{octave}"

    def beat_to_x(self, beat: float) -> float:
        return self.KEYBOARD_W + beat * self.BEAT_W

    def pitch_to_y(self, pitch: int) -> float:
        row = self.MAX_MIDI - pitch
        return self.HEADER_H + row * self.ROW_H

    def x_to_beat(self, x: float) -> float:
        return max(0.0, (x - self.KEYBOARD_W) / self.BEAT_W)

    def y_to_pitch(self, y: float) -> int:
        row = int((y - self.HEADER_H) / self.ROW_H)
        row = max(0, min(self.total_rows - 1, row))
        return self.MAX_MIDI - row

    def note_rect(self, note: Note) -> QRectF:
        x = self.beat_to_x(note.start)
        y = self.pitch_to_y(note.pitch) + 1
        w = max(8.0, note.duration * self.BEAT_W)
        h = self.ROW_H - 2
        return QRectF(x, y, w, h)

    def current_notes(self) -> List[Note]:
        return self.daw.current_notes()

    def ghost_notes(self) -> List[Tuple[Track, Note]]:
        return self.daw.ghost_notes_for_current_pattern()

    def snap_value(self, value: float) -> float:
        snap = self.daw.current_snap
        if snap <= 0.0:
            return value
        return round(value / snap) * snap

    def pitch_in_scale(self, pitch: int) -> bool:
        scale = SCALE_INTERVALS.get(self.daw.scale_mode, set(range(12)))
        root_pc = NOTE_NAMES.index(self.daw.scale_root)
        relative_pc = (pitch - root_pc) % 12
        return relative_pc in scale

    def find_note_at(self, pos_x: float, pos_y: float) -> Optional[Note]:
        for note in reversed(self.current_notes()):
            if self.note_rect(note).contains(pos_x, pos_y):
                return note
        return None

    def _remove_note(self, note: Note) -> None:
        notes = self.current_notes()
        if note in notes:
            notes.remove(note)
            if self.selected_note is note:
                self.selected_note = None
            self.note_changed.emit()
            self.update()

    def _insert_note_or_stamp(self, beat: float, pitch: int) -> Optional[Note]:
        notes = self.current_notes()
        duration = self.daw.default_note_length
        inserted_root: Optional[Note] = None

        for interval in STAMP_INTERVALS.get(self.daw.stamp_mode, [0]):
            new_pitch = pitch + interval
            if new_pitch < self.MIN_MIDI or new_pitch > self.MAX_MIDI:
                continue
            duplicate = next(
                (
                    n
                    for n in notes
                    if n.pitch == new_pitch and abs(n.start - beat) < 0.001 and abs(n.duration - duration) < 0.001
                ),
                None,
            )
            if duplicate:
                continue
            new_note = Note(pitch=new_pitch, start=beat, duration=duration, velocity=105)
            notes.append(new_note)
            if inserted_root is None:
                inserted_root = new_note

        notes.sort(key=lambda n: (n.start, n.pitch))
        self.note_changed.emit()
        self.update()
        return inserted_root

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#101114"))

        pattern_len = self.daw.pattern_length_beats
        grid_right = self.beat_to_x(pattern_len)

        # Header / ruler
        painter.fillRect(0, 0, self.width(), self.HEADER_H, QColor("#15181E"))
        painter.fillRect(0, 0, self.KEYBOARD_W, self.height(), QColor("#0B0D10"))

        # Row backgrounds + piano keys
        for pitch in range(self.MAX_MIDI, self.MIN_MIDI - 1, -1):
            y = self.pitch_to_y(pitch)
            row_rect = QRectF(self.KEYBOARD_W, y, max(0, grid_right - self.KEYBOARD_W), self.ROW_H)
            key_rect = QRectF(0, y, self.KEYBOARD_W, self.ROW_H)

            if self.is_black(pitch):
                key_color = QColor("#20242C")
                row_color = QColor("#14171C")
            else:
                key_color = QColor("#E6E8EC")
                row_color = QColor("#171A20")

            if self.daw.scale_mode != "Chromatic" and self.pitch_in_scale(pitch):
                row_color = QColor(row_color)
                row_color.setAlpha(255)
                row_color = row_color.lighter(116)

            painter.fillRect(row_rect, row_color)
            painter.fillRect(key_rect, key_color)

            painter.setPen(QPen(QColor("#2A3038"), 1))
            painter.drawLine(int(self.KEYBOARD_W), int(y), int(grid_right), int(y))
            painter.setPen(QPen(QColor("#0F1217"), 1))
            painter.drawRect(key_rect)

            label_color = QColor("#0F1114") if not self.is_black(pitch) else QColor("#E4E8F0")
            painter.setPen(label_color)
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(key_rect.adjusted(6, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.note_name(pitch))

        # Vertical grid
        snap = self.daw.current_snap
        if snap <= 0.0:
            snap = 0.25
        steps = int(pattern_len / snap) + 1
        for i in range(steps + 1):
            beat = i * snap
            x = self.beat_to_x(beat)
            if i % int(max(1, round(1.0 / snap))) == 0:
                color = QColor("#313845")
            else:
                color = QColor("#232934")
            if abs(beat % 4.0) < 0.001:
                color = QColor("#465066")
            painter.setPen(QPen(color, 1))
            painter.drawLine(int(x), self.HEADER_H, int(x), self.height())

        # Header beat labels
        painter.setPen(QPen(QColor("#AAB5C4"), 1))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        for beat in range(int(pattern_len) + 1):
            x = self.beat_to_x(beat)
            painter.drawLine(int(x), 0, int(x), self.HEADER_H)
            if beat < pattern_len:
                painter.drawText(int(x + 4), 20, f"{beat + 1}")

        painter.setPen(QPen(QColor("#596273"), 1))
        painter.drawText(10, 20, "PIANO")
        painter.drawText(self.KEYBOARD_W + 8, 20, f"Pattern {self.daw.current_pattern + 1} / {self.daw.pattern_count} | Track: {self.daw.current_track().name}")

        # Ghost notes
        if self.daw.show_ghosts:
            for track, note in self.ghost_notes():
                rect = self.note_rect(note)
                ghost_fill = QColor(track.color)
                ghost_fill.setAlpha(55)
                ghost_border = QColor(track.color)
                ghost_border.setAlpha(110)
                painter.fillRect(rect, ghost_fill)
                painter.setPen(QPen(ghost_border, 1, Qt.PenStyle.DashLine))
                painter.drawRect(rect)

        # Current track notes
        for note in self.current_notes():
            rect = self.note_rect(note)
            fill = QColor(self.daw.current_track().color)
            if note.muted:
                fill = QColor("#5B5F66")
            border = QColor("#F4F7FB")
            if self.selected_note is note:
                fill = fill.lighter(135)
                border = QColor("#FFD54F")

            painter.fillRect(rect, fill)
            painter.setPen(QPen(border, 1))
            painter.drawRect(rect)

            painter.setPen(QPen(QColor("#0A0B0D"), 1))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(rect.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.note_name(note.pitch))

        # Playhead
        if self.daw.playing:
            play_x = self.beat_to_x(self.daw.current_visible_local_beat())
            painter.setPen(QPen(QColor("#FFF15C"), 2))
            painter.drawLine(int(play_x), self.HEADER_H, int(play_x), self.height())

        # Border
        painter.setPen(QPen(QColor("#394250"), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        pos = event.position()
        x = pos.x()
        y = pos.y()

        if y < self.HEADER_H:
            return

        # Preview from piano keyboard area
        if x < self.KEYBOARD_W and event.button() == Qt.MouseButton.LeftButton:
            pitch = self.y_to_pitch(y)
            self.preview_key = f"preview-{time.time_ns()}"
            self.daw.preview_note_on(self.preview_key, pitch)
            return

        clicked_note = self.find_note_at(x, y)
        self.selected_note = clicked_note

        if event.button() == Qt.MouseButton.RightButton:
            if clicked_note is not None:
                self._remove_note(clicked_note)
            else:
                self.selected_note = None
                self.update()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        beat = self.snap_value(self.x_to_beat(x))
        beat = max(0.0, min(self.daw.pattern_length_beats - self.daw.current_snap, beat))
        pitch = self.y_to_pitch(y)

        if clicked_note is not None:
            rect = self.note_rect(clicked_note)
            self.drag_mode = "resize" if (rect.right() - x) <= 8 else "move"
            self.drag_origin_beat = beat
            self.drag_origin_pitch = pitch
            self.drag_note_start_beat = clicked_note.start
            self.drag_note_start_pitch = clicked_note.pitch
        else:
            inserted = self._insert_note_or_stamp(beat, pitch)
            self.selected_note = inserted
            self.drag_mode = None

        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        x = pos.x()
        y = pos.y()

        note_under_cursor = self.find_note_at(x, y) if x >= self.KEYBOARD_W else None
        self._hover_note = note_under_cursor

        if self.drag_mode and self.selected_note is not None:
            beat = self.snap_value(self.x_to_beat(x))
            beat = max(0.0, min(self.daw.pattern_length_beats, beat))
            pitch = self.y_to_pitch(y)

            if self.drag_mode == "move":
                delta_beats = beat - self.drag_origin_beat
                delta_pitch = pitch - self.drag_origin_pitch
                new_start = max(0.0, self.drag_note_start_beat + delta_beats)
                new_start = min(self.daw.pattern_length_beats - self.selected_note.duration, new_start)
                self.selected_note.start = self.snap_value(new_start)
                self.selected_note.pitch = max(self.MIN_MIDI, min(self.MAX_MIDI, self.drag_note_start_pitch + delta_pitch))
            elif self.drag_mode == "resize":
                new_duration = max(self.daw.current_snap, beat - self.selected_note.start)
                max_duration = self.daw.pattern_length_beats - self.selected_note.start
                self.selected_note.duration = min(max_duration, self.snap_value(new_duration))

            self.note_changed.emit()
            self.update()
        else:
            if note_under_cursor is not None:
                rect = self.note_rect(note_under_cursor)
                near_edge = (rect.right() - x) <= 8
                self.setCursor(Qt.CursorShape.SizeHorCursor if near_edge else Qt.CursorShape.OpenHandCursor)
            elif x < self.KEYBOARD_W:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseReleaseEvent(self, event) -> None:
        self.drag_mode = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if self.preview_key is not None:
            self.daw.preview_note_off(self.preview_key)
            self.preview_key = None

    def leaveEvent(self, event) -> None:
        if self.preview_key is not None:
            self.daw.preview_note_off(self.preview_key)
            self.preview_key = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def delete_selected(self) -> None:
        if self.selected_note is not None:
            self._remove_note(self.selected_note)


# ============================ MAIN WINDOW ============================

class NillFLBeepBox(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Nill | FL x BeepBox Open Source DAW")
        self.setGeometry(80, 60, 1520, 920)

        self.pattern_count = 8
        self.current_pattern = 0
        self.pattern_length_beats = 8
        self.bpm = 140
        self.scale_root = "C"
        self.scale_mode = "Minor"
        self.current_snap = 0.25
        self.default_note_length = 1.0
        self.stamp_mode = "Single"
        self.show_ghosts = True
        self.playing = False
        self.loop_mode = "Pattern"
        self.song_order = list(range(self.pattern_count))
        self.playhead_song_beat = 0.0
        self._last_playback_time = time.perf_counter()
        self._last_active_keys: set = set()
        self._preview_keys: set = set()

        self.synth = ChiptuneSynth()
        self.stream = None
        self._start_audio()

        self.tracks: List[Track] = []
        self.selected_track_index = 0
        self._build_default_tracks()

        self._pattern_buttons: List[QPushButton] = []

        self._build_ui()
        self._bind_shortcuts()
        self.refresh_track_list()
        self.refresh_track_controls()
        self.refresh_pattern_buttons()

        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(16)
        self.playback_timer.timeout.connect(self.update_playback)
        self.playback_timer.start()

    # -------------------------- Setup --------------------------

    def _build_default_tracks(self) -> None:
        defaults = [
            ("Lead", DEFAULT_TRACK_COLORS[0], "square"),
            ("Bass", DEFAULT_TRACK_COLORS[1], "triangle"),
            ("Chord", DEFAULT_TRACK_COLORS[2], "saw"),
        ]
        for name, color, waveform in defaults:
            self.tracks.append(Track(name=name, color=color, waveform=waveform, gain=0.70))

    def _start_audio(self) -> None:
        if sd is None:
            print("sounddevice not available. Visual editor still works.")
            return
        try:
            self.stream = sd.OutputStream(
                samplerate=self.synth.sample_rate,
                channels=1,
                blocksize=self.synth.buffer_size,
                callback=self.audio_callback,
            )
            self.stream.start()
            print("Audio engine started.")
        except Exception as exc:
            self.stream = None
            print("Audio startup failed:", exc)

    def audio_callback(self, outdata, frames, time_info, status) -> None:
        if status:
            print(status)
        audio = self.synth.generate_audio(frames)
        outdata[:, 0] = np.clip(audio, -0.97, 0.97)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(340)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(8)

        title = QLabel("NILL // FL x BEEPBOX")
        title.setFont(QFont("Consolas", 15, QFont.Weight.Bold))
        title.setStyleSheet("color:#00E5FF; padding: 6px 0;")
        sidebar_layout.addWidget(title)

        subtitle = QLabel("Pattern-based chiptune DAW prototype with an FL-style piano roll.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#B8C4D6;")
        sidebar_layout.addWidget(subtitle)

        transport_grid = QGridLayout()
        self.play_btn = QPushButton("Play / Pause [Space]")
        self.stop_btn = QPushButton("Stop")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.stop_btn.clicked.connect(self.stop_playback)
        transport_grid.addWidget(self.play_btn, 0, 0, 1, 2)
        transport_grid.addWidget(self.stop_btn, 1, 0, 1, 2)

        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(40, 260)
        self.bpm_spin.setValue(self.bpm)
        self.bpm_spin.valueChanged.connect(self.on_bpm_changed)
        transport_grid.addWidget(QLabel("BPM"), 2, 0)
        transport_grid.addWidget(self.bpm_spin, 2, 1)

        self.loop_mode_combo = QComboBox()
        self.loop_mode_combo.addItems(["Pattern", "Song"])
        self.loop_mode_combo.currentTextChanged.connect(self.on_loop_mode_changed)
        transport_grid.addWidget(QLabel("Loop"), 3, 0)
        transport_grid.addWidget(self.loop_mode_combo, 3, 1)

        self.pattern_len_spin = QSpinBox()
        self.pattern_len_spin.setRange(4, 32)
        self.pattern_len_spin.setSingleStep(4)
        self.pattern_len_spin.setValue(self.pattern_length_beats)
        self.pattern_len_spin.valueChanged.connect(self.on_pattern_length_changed)
        transport_grid.addWidget(QLabel("Pattern Beats"), 4, 0)
        transport_grid.addWidget(self.pattern_len_spin, 4, 1)

        self.snap_combo = QComboBox()
        self.snap_combo.addItem("1 Beat", 1.0)
        self.snap_combo.addItem("1/2 Beat", 0.5)
        self.snap_combo.addItem("1/4 Beat", 0.25)
        self.snap_combo.addItem("1/8 Beat", 0.125)
        self.snap_combo.setCurrentIndex(2)
        self.snap_combo.currentIndexChanged.connect(self.on_snap_changed)
        transport_grid.addWidget(QLabel("Snap"), 5, 0)
        transport_grid.addWidget(self.snap_combo, 5, 1)

        self.note_len_combo = QComboBox()
        self.note_len_combo.addItem("1/8", 0.125)
        self.note_len_combo.addItem("1/4", 0.25)
        self.note_len_combo.addItem("1/2", 0.5)
        self.note_len_combo.addItem("1", 1.0)
        self.note_len_combo.addItem("2", 2.0)
        self.note_len_combo.setCurrentIndex(3)
        self.note_len_combo.currentIndexChanged.connect(self.on_note_length_changed)
        transport_grid.addWidget(QLabel("Draw Length"), 6, 0)
        transport_grid.addWidget(self.note_len_combo, 6, 1)

        self.scale_root_combo = QComboBox()
        self.scale_root_combo.addItems(NOTE_NAMES)
        self.scale_root_combo.setCurrentText(self.scale_root)
        self.scale_root_combo.currentTextChanged.connect(self.on_scale_changed)
        transport_grid.addWidget(QLabel("Scale Root"), 7, 0)
        transport_grid.addWidget(self.scale_root_combo, 7, 1)

        self.scale_mode_combo = QComboBox()
        self.scale_mode_combo.addItems(list(SCALE_INTERVALS.keys()))
        self.scale_mode_combo.setCurrentText(self.scale_mode)
        self.scale_mode_combo.currentTextChanged.connect(self.on_scale_changed)
        transport_grid.addWidget(QLabel("Scale"), 8, 0)
        transport_grid.addWidget(self.scale_mode_combo, 8, 1)

        self.stamp_combo = QComboBox()
        self.stamp_combo.addItems(list(STAMP_INTERVALS.keys()))
        self.stamp_combo.currentTextChanged.connect(self.on_stamp_changed)
        transport_grid.addWidget(QLabel("Stamp / Chord"), 9, 0)
        transport_grid.addWidget(self.stamp_combo, 9, 1)

        self.ghosts_check = QCheckBox("Show Ghost Notes")
        self.ghosts_check.setChecked(self.show_ghosts)
        self.ghosts_check.toggled.connect(self.on_ghosts_toggled)
        transport_grid.addWidget(self.ghosts_check, 10, 0, 1, 2)

        sidebar_layout.addLayout(transport_grid)

        sidebar_layout.addWidget(QLabel("Tracks / Channel Rack"))
        self.track_list = QListWidget()
        self.track_list.currentRowChanged.connect(self.on_track_selected)
        sidebar_layout.addWidget(self.track_list, 1)

        track_btns = QHBoxLayout()
        add_track_btn = QPushButton("+ Add Track")
        remove_track_btn = QPushButton("- Remove")
        add_track_btn.clicked.connect(self.add_track)
        remove_track_btn.clicked.connect(self.remove_track)
        track_btns.addWidget(add_track_btn)
        track_btns.addWidget(remove_track_btn)
        sidebar_layout.addLayout(track_btns)

        sidebar_layout.addWidget(QLabel("Selected Track"))
        props = QGridLayout()

        self.track_name_edit = QLineEdit()
        self.track_name_edit.editingFinished.connect(self.on_track_name_edited)
        props.addWidget(QLabel("Name"), 0, 0)
        props.addWidget(self.track_name_edit, 0, 1)

        self.track_wave_combo = QComboBox()
        self.track_wave_combo.addItems(["square", "triangle", "saw", "sine", "noise"])
        self.track_wave_combo.currentTextChanged.connect(self.on_track_wave_changed)
        props.addWidget(QLabel("Waveform"), 1, 0)
        props.addWidget(self.track_wave_combo, 1, 1)

        self.track_gain_spin = QDoubleSpinBox()
        self.track_gain_spin.setRange(0.05, 1.25)
        self.track_gain_spin.setSingleStep(0.05)
        self.track_gain_spin.setValue(0.70)
        self.track_gain_spin.valueChanged.connect(self.on_track_gain_changed)
        props.addWidget(QLabel("Gain"), 2, 0)
        props.addWidget(self.track_gain_spin, 2, 1)

        sidebar_layout.addLayout(props)

        save_load = QHBoxLayout()
        save_btn = QPushButton("Save Project")
        load_btn = QPushButton("Load Project")
        save_btn.clicked.connect(self.save_project)
        load_btn.clicked.connect(self.load_project)
        save_load.addWidget(save_btn)
        save_load.addWidget(load_btn)
        sidebar_layout.addLayout(save_load)

        help_text = QLabel(
            "Left click: draw/select note\n"
            "Drag note: move\n"
            "Drag note edge: resize\n"
            "Right click note: delete\n"
            "Click piano keys: preview notes\n"
            "Ghost notes show other tracks in the same pattern"
        )
        help_text.setStyleSheet("color:#8DA1BC;")
        sidebar_layout.addWidget(help_text)

        outer.addWidget(sidebar)

        # Main editor area
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        pattern_bar = QWidget()
        pattern_layout = QHBoxLayout(pattern_bar)
        pattern_layout.setContentsMargins(0, 0, 0, 0)
        pattern_layout.setSpacing(6)
        pattern_layout.addWidget(QLabel("Patterns"))
        for i in range(self.pattern_count):
            btn = QPushButton(f"P{i + 1}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked=False, idx=i: self.set_current_pattern(idx))
            self._pattern_buttons.append(btn)
            pattern_layout.addWidget(btn)
        pattern_layout.addStretch(1)
        right_layout.addWidget(pattern_bar)

        self.piano_roll = PianoRoll(self)
        self.piano_roll.note_changed.connect(self.on_notes_changed)
        right_layout.addWidget(self.piano_roll, 1)

        footer = QLabel("Hybrid workflow: FL-style piano roll + BeepBox-style pattern editing.")
        footer.setStyleSheet("color:#92A0B8; padding:4px;")
        right_layout.addWidget(footer)

        outer.addWidget(right, 1)

        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0E1015;
                color: #E6EAF2;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 12px;
            }
            QPushButton {
                background: #1A1F28;
                border: 1px solid #364155;
                padding: 8px;
                border-radius: 6px;
            }
            QPushButton:hover { background: #212836; }
            QPushButton:checked { background: #00B8D9; color: #091016; border: 1px solid #6CF1FF; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {
                background: #141922;
                border: 1px solid #334055;
                padding: 5px;
                border-radius: 4px;
            }
            QLabel { color: #E6EAF2; }
            QListWidget::item { padding: 5px; }
            """
        )

    def _bind_shortcuts(self) -> None:
        save_action = QAction(self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        self.addAction(save_action)

        load_action = QAction(self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_project)
        self.addAction(load_action)

        delete_action = QAction(self)
        delete_action.setShortcut("Delete")
        delete_action.triggered.connect(self.piano_roll.delete_selected)
        self.addAction(delete_action)

    # -------------------------- Data access --------------------------

    def current_track(self) -> Track:
        return self.tracks[self.selected_track_index]

    def current_notes(self) -> List[Note]:
        return self.current_track().patterns.setdefault(self.current_pattern, [])

    def ghost_notes_for_current_pattern(self) -> List[Tuple[Track, Note]]:
        items: List[Tuple[Track, Note]] = []
        for index, track in enumerate(self.tracks):
            if index == self.selected_track_index:
                continue
            for note in track.patterns.get(self.current_pattern, []):
                items.append((track, note))
        return items

    def current_visible_local_beat(self) -> float:
        if self.loop_mode == "Pattern":
            return self.playhead_song_beat % self.pattern_length_beats
        return self.playhead_song_beat % self.pattern_length_beats

    # -------------------------- UI refresh --------------------------

    def refresh_pattern_buttons(self) -> None:
        for i, btn in enumerate(self._pattern_buttons):
            btn.blockSignals(True)
            btn.setChecked(i == self.current_pattern)
            btn.blockSignals(False)

    def refresh_track_list(self) -> None:
        self.track_list.blockSignals(True)
        self.track_list.clear()
        for track in self.tracks:
            item = QListWidgetItem(f"{track.name}  [{track.waveform}]")
            item.setBackground(QColor(track.color))
            item.setForeground(QColor("#090B0E"))
            self.track_list.addItem(item)
        self.track_list.setCurrentRow(self.selected_track_index)
        self.track_list.blockSignals(False)

    def refresh_track_controls(self) -> None:
        if not self.tracks:
            return
        track = self.current_track()
        self.track_name_edit.blockSignals(True)
        self.track_wave_combo.blockSignals(True)
        self.track_gain_spin.blockSignals(True)
        self.track_name_edit.setText(track.name)
        self.track_wave_combo.setCurrentText(track.waveform)
        self.track_gain_spin.setValue(track.gain)
        self.track_name_edit.blockSignals(False)
        self.track_wave_combo.blockSignals(False)
        self.track_gain_spin.blockSignals(False)
        self.piano_roll.update()

    # -------------------------- Controls --------------------------

    def on_bpm_changed(self, value: int) -> None:
        self.bpm = value

    def on_loop_mode_changed(self, text: str) -> None:
        self.loop_mode = text
        self.stop_playback()

    def on_pattern_length_changed(self, value: int) -> None:
        self.pattern_length_beats = value
        self.piano_roll.update()

    def on_snap_changed(self, index: int) -> None:
        self.current_snap = float(self.snap_combo.currentData())
        self.piano_roll.update()

    def on_note_length_changed(self, index: int) -> None:
        self.default_note_length = float(self.note_len_combo.currentData())

    def on_scale_changed(self, *args) -> None:
        self.scale_root = self.scale_root_combo.currentText()
        self.scale_mode = self.scale_mode_combo.currentText()
        self.piano_roll.update()

    def on_stamp_changed(self, text: str) -> None:
        self.stamp_mode = text

    def on_ghosts_toggled(self, checked: bool) -> None:
        self.show_ghosts = checked
        self.piano_roll.update()

    def on_track_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.tracks):
            return
        self.selected_track_index = row
        self.refresh_track_controls()
        self.piano_roll.selected_note = None
        self.piano_roll.update()

    def on_track_name_edited(self) -> None:
        self.current_track().name = self.track_name_edit.text().strip() or "Track"
        self.refresh_track_list()
        self.piano_roll.update()

    def on_track_wave_changed(self, text: str) -> None:
        self.current_track().waveform = text
        self.refresh_track_list()

    def on_track_gain_changed(self, value: float) -> None:
        self.current_track().gain = value

    def on_notes_changed(self) -> None:
        self.piano_roll.update()

    def set_current_pattern(self, index: int) -> None:
        self.current_pattern = index
        self.refresh_pattern_buttons()
        self.piano_roll.selected_note = None
        self.piano_roll.update()

    def add_track(self) -> None:
        color = DEFAULT_TRACK_COLORS[len(self.tracks) % len(DEFAULT_TRACK_COLORS)]
        name = f"Track {len(self.tracks) + 1}"
        waveform_cycle = ["square", "triangle", "saw", "noise"]
        waveform = waveform_cycle[len(self.tracks) % len(waveform_cycle)]
        self.tracks.append(Track(name=name, color=color, waveform=waveform, gain=0.70))
        self.selected_track_index = len(self.tracks) - 1
        self.refresh_track_list()
        self.refresh_track_controls()

    def remove_track(self) -> None:
        if len(self.tracks) <= 1:
            QMessageBox.information(self, "Nill", "At least one track must remain.")
            return
        self.tracks.pop(self.selected_track_index)
        self.selected_track_index = max(0, min(self.selected_track_index, len(self.tracks) - 1))
        self.refresh_track_list()
        self.refresh_track_controls()
        self.piano_roll.update()

    # -------------------------- Playback --------------------------

    def toggle_playback(self) -> None:
        self.playing = not self.playing
        self._last_playback_time = time.perf_counter()
        if not self.playing:
            self._release_all_active_playback_notes()
        self.piano_roll.update()

    def stop_playback(self) -> None:
        self.playing = False
        self.playhead_song_beat = 0.0
        self._last_playback_time = time.perf_counter()
        self._release_all_active_playback_notes()
        self.piano_roll.update()

    def _release_all_active_playback_notes(self) -> None:
        for key in list(self._last_active_keys):
            self.synth.note_off(key)
        self._last_active_keys.clear()

    def _collect_active_notes(self) -> Dict[str, Tuple[Track, Note]]:
        active: Dict[str, Tuple[Track, Note]] = {}

        if self.loop_mode == "Pattern":
            pattern_index = self.current_pattern
            local_beat = self.playhead_song_beat % self.pattern_length_beats
        else:
            total_beats = self.pattern_count * self.pattern_length_beats
            if total_beats <= 0:
                return active
            song_beat = self.playhead_song_beat % total_beats
            slot = int(song_beat // self.pattern_length_beats)
            pattern_index = self.song_order[slot]
            local_beat = song_beat - slot * self.pattern_length_beats

        for track_index, track in enumerate(self.tracks):
            pattern_notes = track.patterns.get(pattern_index, [])
            for note_index, note in enumerate(pattern_notes):
                if note.muted:
                    continue
                if note.start <= local_beat < (note.start + note.duration):
                    key = f"{track_index}:{pattern_index}:{note_index}"
                    active[key] = (track, note)
        return active

    def update_playback(self) -> None:
        now = time.perf_counter()
        if not self.playing:
            self._last_playback_time = now
            return

        delta = now - self._last_playback_time
        self._last_playback_time = now
        self.playhead_song_beat += delta * (self.bpm / 60.0)

        if self.loop_mode == "Pattern":
            total_beats = self.pattern_length_beats
        else:
            total_beats = self.pattern_length_beats * self.pattern_count

        if total_beats > 0 and self.playhead_song_beat >= total_beats:
            self.playhead_song_beat %= total_beats

        active = self._collect_active_notes()
        active_keys = set(active.keys())

        for key in self._last_active_keys - active_keys:
            self.synth.note_off(key)

        for key in active_keys - self._last_active_keys:
            track, note = active[key]
            self.synth.note_on(
                key=key,
                pitch=note.pitch,
                velocity=max(0.0, min(1.0, note.velocity / 127.0)),
                waveform=track.waveform,
                gain=track.gain,
            )

        self._last_active_keys = active_keys
        self.piano_roll.update()

    # -------------------------- Preview --------------------------

    def preview_note_on(self, key: str, pitch: int) -> None:
        track = self.current_track()
        self._preview_keys.add(key)
        self.synth.note_on(key, pitch, velocity=0.9, waveform=track.waveform, gain=min(1.0, track.gain))

    def preview_note_off(self, key: str) -> None:
        if key in self._preview_keys:
            self.synth.note_off(key)
            self._preview_keys.discard(key)

    # -------------------------- Save / Load --------------------------

    def project_data(self) -> dict:
        return {
            "app": "Nill FL x BeepBox",
            "version": 2,
            "bpm": self.bpm,
            "pattern_count": self.pattern_count,
            "current_pattern": self.current_pattern,
            "pattern_length_beats": self.pattern_length_beats,
            "scale_root": self.scale_root,
            "scale_mode": self.scale_mode,
            "snap": self.current_snap,
            "draw_length": self.default_note_length,
            "stamp_mode": self.stamp_mode,
            "show_ghosts": self.show_ghosts,
            "tracks": [track.to_dict() for track in self.tracks],
        }

    def save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Nill Project",
            str(Path.home() / "nill_project.json"),
            "Nill Project (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.project_data(), fh, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Nill Project",
            str(Path.home()),
            "Nill Project (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            return

        self.stop_playback()
        self.bpm = int(data.get("bpm", 140))
        self.pattern_count = min(len(self._pattern_buttons), int(data.get("pattern_count", 8)))
        self.current_pattern = int(data.get("current_pattern", 0))
        self.pattern_length_beats = int(data.get("pattern_length_beats", 8))
        self.scale_root = str(data.get("scale_root", "C"))
        self.scale_mode = str(data.get("scale_mode", "Minor"))
        self.current_snap = float(data.get("snap", 0.25))
        self.default_note_length = float(data.get("draw_length", 1.0))
        self.stamp_mode = str(data.get("stamp_mode", "Single"))
        self.show_ghosts = bool(data.get("show_ghosts", True))
        self.song_order = list(range(self.pattern_count))

        loaded_tracks = [Track.from_dict(t) for t in data.get("tracks", [])]
        self.tracks = loaded_tracks if loaded_tracks else [Track(name="Track 1", color="#00E5FF")]
        self.selected_track_index = 0

        self.bpm_spin.setValue(self.bpm)
        self.pattern_len_spin.setValue(self.pattern_length_beats)
        self.scale_root_combo.setCurrentText(self.scale_root)
        self.scale_mode_combo.setCurrentText(self.scale_mode)
        self.ghosts_check.setChecked(self.show_ghosts)
        self.stamp_combo.setCurrentText(self.stamp_mode)

        snap_index = max(0, self.snap_combo.findData(self.current_snap))
        self.snap_combo.setCurrentIndex(snap_index)
        note_len_index = max(0, self.note_len_combo.findData(self.default_note_length))
        self.note_len_combo.setCurrentIndex(note_len_index)

        self.current_pattern = max(0, min(self.current_pattern, self.pattern_count - 1))
        self.refresh_track_list()
        self.refresh_track_controls()
        self.refresh_pattern_buttons()
        self.piano_roll.selected_note = None
        self.piano_roll.update()

    # -------------------------- Events --------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self.toggle_playback()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Delete:
            self.piano_roll.delete_selected()
            event.accept()
            return

        note = self.piano_roll.selected_note
        if note is not None:
            if event.key() == Qt.Key.Key_Up:
                note.pitch = min(PianoRoll.MAX_MIDI, note.pitch + 1)
                self.piano_roll.update()
                return
            if event.key() == Qt.Key.Key_Down:
                note.pitch = max(PianoRoll.MIN_MIDI, note.pitch - 1)
                self.piano_roll.update()
                return
            if event.key() == Qt.Key.Key_Left:
                note.start = max(0.0, note.start - self.current_snap)
                self.piano_roll.update()
                return
            if event.key() == Qt.Key.Key_Right:
                note.start = min(self.pattern_length_beats - note.duration, note.start + self.current_snap)
                self.piano_roll.update()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.stop_playback()
        self.synth.all_notes_off()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        super().closeEvent(event)


def main() -> None:
    print("Nill | FL x BeepBox Open Source DAW")
    print("Features: pattern editor, FL-style piano roll, visible black/white keyboard, ghost notes, scale highlighting, track rack.")
    app = QApplication(sys.argv)
    window = NillFLBeepBox()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
