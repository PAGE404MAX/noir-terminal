#!/usr/bin/env python3
"""
FIXESNEEDED | monochrome terminal DAW

FIXESNEEDED code with the Nill left-sidebar/UI design recolored to a black-and-white terminal feel.

Inspired by:
- FL Studio-style piano roll ideas: ghost notes, scale highlighting, stamp/chord tool,
  channel-style track list, visible piano keyboard, editable note blocks.
- BeepBox-style ideas: pattern-first composition, fast grid entry, high-contrast monochrome chiptune-friendly
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
import subprocess
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

from PySide6.QtCore import Qt, QTimer, QRectF, Signal, QEvent
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPushButton,
    QScrollArea,
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
    "#E6E6E6",
    "#CFCFCF",
    "#B8B8B8",
    "#FFFFFF",
    "#8A8A8A",
    "#737373",
    "#5C5C5C",
    "#444444",
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
            color=str(data.get("color", "#E6E6E6")),
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
    # ROW_H and BEAT_W are responsive properties below so the piano roll fills the right side.
    MIN_MIDI = 36
    MAX_MIDI = 96

    def __init__(self, daw: "Nill") -> None:
        super().__init__()
        self.daw = daw
        self.setMinimumSize(800, 500)
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

    @property
    def BEAT_W(self) -> float:
        # Horizontal piano-roll zoom only: expand beat spacing inside the scroll area.
        if self.daw.pattern_length_beats <= 0:
            return 40.0
        viewport_width = self.width()
        scroll = getattr(self.daw, "piano_scroll", None)
        if scroll is not None:
            viewport_width = scroll.viewport().width()
        available = max(100.0, float(viewport_width - self.KEYBOARD_W))
        fit_width = max(12.0, available / float(self.daw.pattern_length_beats))
        return fit_width * float(getattr(self.daw, "zoom_x", 1.0))

    @property
    def ROW_H(self) -> float:
        # Vertical piano-roll zoom only: makes rows and note blocks taller inside the scroll area.
        viewport_height = self.height()
        scroll = getattr(self.daw, "piano_scroll", None)
        if scroll is not None:
            viewport_height = scroll.viewport().height()
        available = max(200.0, float(viewport_height - self.HEADER_H - self.FOOTER_H))
        fit_height = max(10.0, available / float(self.total_rows))
        return fit_height * float(getattr(self.daw, "zoom_y", 1.0))

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
        painter.fillRect(self.rect(), QColor("#0B0B0B"))

        pattern_len = self.daw.pattern_length_beats
        grid_right = self.beat_to_x(pattern_len)

        # Header / ruler
        painter.fillRect(0, 0, self.width(), self.HEADER_H, QColor("#151515"))
        painter.fillRect(0, 0, self.KEYBOARD_W, self.height(), QColor("#050505"))

        # Row backgrounds + piano keys
        for pitch in range(self.MAX_MIDI, self.MIN_MIDI - 1, -1):
            y = self.pitch_to_y(pitch)
            row_rect = QRectF(self.KEYBOARD_W, y, max(0, grid_right - self.KEYBOARD_W), self.ROW_H)
            key_rect = QRectF(0, y, self.KEYBOARD_W, self.ROW_H)

            if self.is_black(pitch):
                key_color = QColor("#202020")
                row_color = QColor("#141414")
            else:
                key_color = QColor("#E6E6E6")
                row_color = QColor("#181818")

            if self.daw.scale_mode != "Chromatic" and self.pitch_in_scale(pitch):
                row_color = QColor(row_color)
                row_color.setAlpha(255)
                row_color = row_color.lighter(116)

            painter.fillRect(row_rect, row_color)
            painter.fillRect(key_rect, key_color)

            painter.setPen(QPen(QColor("#2A2A2A"), 1))
            painter.drawLine(int(self.KEYBOARD_W), int(y), int(grid_right), int(y))
            painter.setPen(QPen(QColor("#0F0F0F"), 1))
            painter.drawRect(key_rect)

            label_color = QColor("#111111") if not self.is_black(pitch) else QColor("#E4E4E4")
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
                color = QColor("#3A3A3A")
            else:
                color = QColor("#262626")
            if abs(beat % 4.0) < 0.001:
                color = QColor("#555555")
            painter.setPen(QPen(color, 1))
            painter.drawLine(int(x), self.HEADER_H, int(x), self.height())

        # Header beat labels
        painter.setPen(QPen(QColor("#C8C8C8"), 1))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        for beat in range(int(pattern_len) + 1):
            x = self.beat_to_x(beat)
            painter.drawLine(int(x), 0, int(x), self.HEADER_H)
            if beat < pattern_len:
                painter.drawText(int(x + 4), 20, f"{beat + 1}")

        painter.setPen(QPen(QColor("#777777"), 1))
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
                fill = QColor("#5F5F5F")
            border = QColor("#F4F4F4")
            if self.selected_note is note:
                fill = fill.lighter(135)
                border = QColor("#FFFFFF")

            painter.fillRect(rect, fill)
            painter.setPen(QPen(border, 1))
            painter.drawRect(rect)

            painter.setPen(QPen(QColor("#0A0A0A"), 1))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(rect.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.note_name(note.pitch))

        # Playhead
        if self.daw.playing:
            play_x = self.beat_to_x(self.daw.current_visible_local_beat())
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.drawLine(int(play_x), self.HEADER_H, int(play_x), self.height())

        # Border
        painter.setPen(QPen(QColor("#4A4A4A"), 1))
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
                self.daw.show_grid_menu(event.globalPosition().toPoint())
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

class Nill(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FIXESNEEDED | Black & White Terminal UI")
        self.resize(1400, 850)
        self.setMinimumSize(1000, 600)

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
        # Piano-roll zoom only: affects the note grid/note blocks, not sidebar/buttons/UI widgets.
        self.zoom_x = 1.0
        self.zoom_y = 1.0
        self.loop_mode = "Pattern"
        self.song_order = list(range(self.pattern_count))
        self.playhead_song_beat = 0.0
        self._last_playback_time = time.perf_counter()
        self._last_active_keys: set = set()
        self._preview_keys: set = set()
        self.visualizer_process = None
        self.osc_process = None

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

        title = QLabel("NILL // TERMINAL")
        title.setFont(QFont("Consolas", 15, QFont.Weight.Bold))
        title.setStyleSheet("color:#FFFFFF; padding: 6px 0; letter-spacing: 1px;")
        sidebar_layout.addWidget(title)

        subtitle = QLabel("Black-and-white terminal style pattern DAW with an FL-style piano roll.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#BDBDBD;")
        sidebar_layout.addWidget(subtitle)

        transport_grid = QGridLayout()
        self.play_btn = QPushButton("○")
        self.play_btn.setCheckable(True)
        self.play_btn.setObjectName("transportCircle")
        self.play_btn.setFixedSize(48, 48)
        self.play_btn.setToolTip("Play / Stop sequence (Space)")
        self.play_btn.clicked.connect(self.toggle_playback)
        transport_grid.addWidget(self.play_btn, 0, 0, 2, 1, Qt.AlignmentFlag.AlignLeft)
        space_hint = QLabel("SPACE\nPLAY / STOP")
        space_hint.setStyleSheet("color:#BDBDBD;")
        transport_grid.addWidget(space_hint, 0, 1, 2, 1)

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
        self.snap_combo.addItem("1/2", 0.5)
        self.snap_combo.addItem("1/4", 0.25)
        self.snap_combo.addItem("1/8", 0.125)
        self.snap_combo.addItem("1/16", 0.0625)
        self.snap_combo.addItem("1/32", 0.03125)
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
        help_text.setStyleSheet("color:#A0A0A0;")
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

        command_prompt = QLabel(">")
        command_prompt.setStyleSheet("color:#FFFFFF; font-weight:bold;")
        pattern_layout.addWidget(command_prompt)
        self.command_line = QLineEdit()
        self.command_line.setPlaceholderText("set bpm ___ | show osc | show visualizer")
        self.command_line.setMinimumWidth(260)
        self.command_line.setToolTip("Commands: set bpm ___, show osc, show visualizer")
        self.command_line.returnPressed.connect(self.run_command_line)
        command_completer = QCompleter(["set bpm ", "show osc", "show visualizer"], self.command_line)
        command_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.command_line.setCompleter(command_completer)
        pattern_layout.addWidget(self.command_line, 1)

        pattern_layout.addWidget(QLabel("Zoom"))
        self.zoom_out_btn = QPushButton("-")
        self.zoom_in_btn = QPushButton("+")
        self.zoom_reset_btn = QPushButton("100%")
        self.zoom_out_btn.setToolTip("Zoom piano roll out")
        self.zoom_in_btn.setToolTip("Zoom piano roll in")
        self.zoom_reset_btn.setToolTip("Reset piano roll zoom")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_reset_btn.clicked.connect(self.reset_zoom)
        pattern_layout.addWidget(self.zoom_out_btn)
        pattern_layout.addWidget(self.zoom_reset_btn)
        pattern_layout.addWidget(self.zoom_in_btn)
        right_layout.addWidget(pattern_bar)

        self.piano_roll = PianoRoll(self)
        self.piano_roll.note_changed.connect(self.on_notes_changed)
        self.piano_scroll = QScrollArea()
        self.piano_scroll.setWidgetResizable(True)
        self.piano_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.piano_scroll.setWidget(self.piano_roll)
        right_layout.addWidget(self.piano_scroll, 1)
        QTimer.singleShot(0, self.apply_zoom)

        footer = QLabel("Hybrid workflow: FL-style piano roll + BeepBox-style pattern editing.")
        footer.setStyleSheet("color:#9A9A9A; padding:4px;")
        right_layout.addWidget(footer)

        outer.addWidget(right, 1)

        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0A0A0A;
                color: #E6E6E6;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 12px;
            }
            QPushButton {
                background: #1A1A1A;
                border: 1px solid #444444;
                padding: 8px;
                border-radius: 6px;
            }
            QPushButton:hover { background: #242424; }
            QPushButton:checked { background: #D9D9D9; color: #080808; border: 1px solid #FFFFFF; }
            QPushButton#transportCircle {
                border-radius: 24px;
                font-size: 28px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton#transportCircle:checked {
                background: #E6E6E6;
                color: #080808;
                border: 2px solid #FFFFFF;
            }
            QScrollArea { border: 1px solid #4A4A4A; background: #050505; }
            QScrollBar:horizontal, QScrollBar:vertical { background: #0A0A0A; border: 1px solid #2A2A2A; }
            QScrollBar::handle:horizontal, QScrollBar::handle:vertical { background: #777777; border: 1px solid #BDBDBD; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {
                background: #141414;
                border: 1px solid #3F3F3F;
                padding: 5px;
                border-radius: 4px;
            }
            QLabel { color: #E6E6E6; }
            QListWidget::item { padding: 5px; }
            """
        )

        # Make Space a transport-only hotkey instead of accidentally pressing focused buttons.
        for button in self.findChildren(QPushButton):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        QApplication.instance().installEventFilter(self)

    def _bind_shortcuts(self) -> None:
        zoom_in_action = QAction(self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        zoom_in_action.triggered.connect(self.zoom_in)
        self.addAction(zoom_in_action)

        zoom_out_action = QAction(self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        zoom_out_action.triggered.connect(self.zoom_out)
        self.addAction(zoom_out_action)

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
            item.setForeground(QColor("#090909"))
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


    def run_command_line(self) -> None:
        raw = self.command_line.text().strip()
        command = " ".join(raw.lower().split())

        if not command:
            return

        if command.startswith("set bpm "):
            value_text = command.removeprefix("set bpm ").strip()
            try:
                bpm = int(value_text)
            except ValueError:
                QMessageBox.warning(self, "Command", "Use: set bpm ___")
                return
            if bpm < 40 or bpm > 260:
                QMessageBox.warning(self, "Command", "BPM must be between 40 and 260.")
                return
            self.bpm_spin.setValue(bpm)
            self.command_line.clear()
            return

        if command == "show osc":
            self.command_line.clear()
            self.show_osc()
            return

        if command == "show visualizer":
            self.command_line.clear()
            self.show_visualizer()
            return

        QMessageBox.warning(self, "Command", "Allowed commands:\nset bpm ___\nshow osc\nshow visualizer")

    def show_osc(self) -> None:
        """Launch Nill OSC from this Nill file."""
        if self.osc_process is not None and self.osc_process.poll() is None:
            QMessageBox.information(self, "OSC", "OSC window is already running.")
            return

        try:
            self.osc_process = subprocess.Popen([sys.executable, str(Path(__file__)), "--nill-osc"])
        except Exception as exc:
            QMessageBox.critical(self, "OSC Failed", str(exc))

    def show_visualizer(self) -> None:
        """Launch the embedded pygame audio visualizer from this Nill file."""
        if self.visualizer_process is not None and self.visualizer_process.poll() is None:
            QMessageBox.information(self, "Visualizer", "Visualizer is already running.")
            return

        try:
            self.visualizer_process = subprocess.Popen([sys.executable, str(Path(__file__)), "--nill-visualizer"])
        except Exception as exc:
            QMessageBox.critical(self, "Visualizer Failed", str(exc))

    def update_transport_button(self) -> None:
        if not hasattr(self, "play_btn"):
            return
        self.play_btn.blockSignals(True)
        self.play_btn.setChecked(self.playing)
        self.play_btn.setText("●" if self.playing else "○")
        self.play_btn.setToolTip("Stop sequence (Space)" if self.playing else "Play sequence (Space)")
        self.play_btn.blockSignals(False)

    def apply_zoom(self) -> None:
        """Apply FL-style piano-roll zoom without scaling the surrounding UI."""
        if not hasattr(self, "piano_roll"):
            return

        viewport_width = self.piano_roll.width()
        viewport_height = self.piano_roll.height()
        if hasattr(self, "piano_scroll"):
            viewport_width = max(1, self.piano_scroll.viewport().width())
            viewport_height = max(1, self.piano_scroll.viewport().height())

        base_beat = max(12.0, max(100.0, float(viewport_width - PianoRoll.KEYBOARD_W)) / float(max(1, self.pattern_length_beats)))
        base_row = max(10.0, max(200.0, float(viewport_height - PianoRoll.HEADER_H - PianoRoll.FOOTER_H)) / float(PianoRoll.MAX_MIDI - PianoRoll.MIN_MIDI + 1))

        zoomed_width = int(PianoRoll.KEYBOARD_W + self.pattern_length_beats * base_beat * self.zoom_x) + 4
        zoomed_height = int(PianoRoll.HEADER_H + (PianoRoll.MAX_MIDI - PianoRoll.MIN_MIDI + 1) * base_row * self.zoom_y) + 4

        # Only the piano-roll canvas changes. The sidebar, buttons, pattern bar,
        # and other UI widgets stay the same size. Bigger zoom = bigger note blocks.
        self.piano_roll.setMinimumWidth(max(800, zoomed_width))
        self.piano_roll.setMinimumHeight(max(500, zoomed_height))

        if hasattr(self, "zoom_reset_btn"):
            self.zoom_reset_btn.setText(f"{int(self.zoom_x * 100)}%")
        self.piano_roll.updateGeometry()
        self.piano_roll.update()

    def zoom_in(self) -> None:
        self.zoom_x = min(4.0, round(self.zoom_x + 0.25, 2))
        self.zoom_y = min(3.0, round(self.zoom_y + 0.20, 2))
        self.apply_zoom()

    def zoom_out(self) -> None:
        self.zoom_x = max(1.0, round(self.zoom_x - 0.25, 2))
        self.zoom_y = max(1.0, round(self.zoom_y - 0.20, 2))
        self.apply_zoom()

    def reset_zoom(self) -> None:
        self.zoom_x = 1.0
        self.zoom_y = 1.0
        self.apply_zoom()

    # -------------------------- Controls --------------------------

    def on_bpm_changed(self, value: int) -> None:
        self.bpm = value

    def on_loop_mode_changed(self, text: str) -> None:
        self.loop_mode = text
        self.stop_playback()

    def on_pattern_length_changed(self, value: int) -> None:
        self.pattern_length_beats = value
        self.apply_zoom()

    def on_snap_changed(self, index: int) -> None:
        self.current_snap = float(self.snap_combo.currentData())
        self.piano_roll.update()

    def set_grid_snap(self, value: float) -> None:
        self.current_snap = value
        snap_index = self.snap_combo.findData(value)
        if snap_index >= 0:
            self.snap_combo.blockSignals(True)
            self.snap_combo.setCurrentIndex(snap_index)
            self.snap_combo.blockSignals(False)
        self.piano_roll.update()

    def show_grid_menu(self, global_pos) -> None:
        menu = QMenu(self)
        menu.setTitle("Grid Size")
        grid_options = [("1/2", 0.5), ("1/4", 0.25), ("1/8", 0.125), ("1/16", 0.0625), ("1/32", 0.03125)]
        for label, value in grid_options:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(abs(self.current_snap - value) < 0.000001)
            action.triggered.connect(lambda checked=False, snap=value: self.set_grid_snap(snap))
        menu.exec(global_pos)

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
        # Single transport control: Space / circle button starts from the current position,
        # and stops the sequence by resetting the playhead to the beginning.
        if self.playing:
            self.stop_playback()
            return
        self.playing = True
        self._last_playback_time = time.perf_counter()
        self.update_transport_button()
        self.piano_roll.update()

    def stop_playback(self) -> None:
        self.playing = False
        self.playhead_song_beat = 0.0
        self._last_playback_time = time.perf_counter()
        self._release_all_active_playback_notes()
        self.update_transport_button()
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
            "app": "Nill Terminal Monochrome",
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
        self.tracks = loaded_tracks if loaded_tracks else [Track(name="Track 1", color="#E6E6E6")]
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

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            # Let the command line type spaces for commands like "set bpm 140".
            if hasattr(self, "command_line") and self.command_line.hasFocus():
                return False
            self.toggle_playback()
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:
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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "piano_roll"):
            QTimer.singleShot(0, self.apply_zoom)

    def closeEvent(self, event) -> None:
        self.stop_playback()
        self.synth.all_notes_off()
        if self.visualizer_process is not None and self.visualizer_process.poll() is None:
            try:
                self.visualizer_process.terminate()
            except Exception:
                pass
        if self.osc_process is not None and self.osc_process.poll() is None:
            try:
                self.osc_process.terminate()
            except Exception:
                pass
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        super().closeEvent(event)





EMBEDDED_OSC_CODE = '"""\nNill OSC\nFOCUS: LEFT PANEL (OSC 1 & OSC 2) - FULLY FUNCTIONAL\n\nFEATURES (THIS VERSION):\n1. OSC 1-2 - Waveform, Detune, Level, Phase (All Working)\n2. Working Scrollbars (Left & Right Panels)\n3. Double-Click Knob Reset\n4. Visual Piano Keyboard\n5. M/N Waveform Cycling (Global Override)\n6. Thread-safe Audio Engine\n"""\n\nimport tkinter as tk\nfrom tkinter import Canvas\nimport math\nimport numpy as np\nfrom typing import Dict, List, Optional, Callable\nimport threading\nimport time\nimport sounddevice as sd\n\n# --- Terminal Greyscale Palette ---\nBG_BLACK = \'#000000\'\nBG_DARK = \'#111111\'\nBG_PANEL = \'#0a0a0a\'\nFG_WHITE = \'#FFFFFF\'\nFG_DIM = \'#666666\'\nFG_BRIGHT = \'#CCCCCC\'\nBORDER = \'#333333\'\nKNOB_BODY = \'#505050\'\nKNOB_SURFACE = \'#1a1a1a\'\nFONT_MAIN = \'Consolas\'\n\nSR = 44100\nBUFFER_SIZE = 2048\nAMP = 0.08\n\nWAVEFORMS = [\'sine\', \'triangle\', \'saw\', \'square\']\nWHITE_KEYS = {\'a\': 60, \'s\': 62, \'d\': 64, \'f\': 65, \'g\': 67, \'h\': 69, \'j\': 71, \'k\': 72, \'l\': 74, \';\': 76, "\'": 77}\nBLACK_KEYS = {\'w\': 61, \'e\': 63, \'t\': 66, \'y\': 68, \'u\': 70, \'o\': 73, \'p\': 75}\nALL_KEYS = {**WHITE_KEYS, **BLACK_KEYS}\nOCTAVE_LABELS = {60: \'C3\', 72: \'C4\', 84: \'C5\'}\n\n\nclass PolyphonicSynth:\n    def __init__(self) -> None:\n        self.active_notes: Dict[int, Dict] = {}\n        self.sample_rate = SR\n        self.buffer_size = BUFFER_SIZE\n        self._lock = threading.Lock()\n        self.last_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)\n        \n        # Global waveform override (M/N cycling)\n        self.waveform_type = \'sine\'\n        self.use_global_waveform = False\n        \n        # OSC 1 Parameters\n        self.osc1_wave = \'sine\'\n        self.osc1_detune = 0.0\n        self.osc1_level = 1.0\n        self.osc1_phase = 0.0\n        \n        # OSC 2 Parameters\n        self.osc2_wave = \'saw\'\n        self.osc2_detune = 0.0\n        self.osc2_level = 0.0\n        self.osc2_phase = 0.0\n        \n        # FX Parameters\n        self.fx_distortion = 0.0\n        self.fx_delay_time = 0.3\n        self.fx_delay_feedback = 0.4\n        self.fx_delay_mix = 0.0\n        self.fx_reverb_mix = 0.0\n        self.delay_buffer_size = int(SR * 2.0)\n        self.delay_buffer = np.zeros(self.delay_buffer_size, dtype=np.float32)\n        self.delay_write_idx = 0\n\n        # ADSR Envelope parameters\n        self.attack_time = 0.01   # seconds\n        self.decay_time = 0.1     # seconds\n        self.sustain_level = 0.8  # 0-1\n        self.release_time = 0.3   # seconds\n\n    def set_waveform(self, waveform: str) -> None:\n        with self._lock:\n            self.waveform_type = waveform\n\n    def set_osc_wave(self, osc: int, wave: str) -> None:\n        with self._lock:\n            if osc == 1:\n                self.osc1_wave = wave\n            elif osc == 2:\n                self.osc2_wave = wave\n\n    def set_osc_detune(self, osc: int, value: float) -> None:\n        with self._lock:\n            if osc == 1:\n                self.osc1_detune = value\n            elif osc == 2:\n                self.osc2_detune = value\n\n    def set_osc_level(self, osc: int, value: float) -> None:\n        with self._lock:\n            if osc == 1:\n                self.osc1_level = value\n            elif osc == 2:\n                self.osc2_level = value\n\n    def set_osc_phase(self, osc: int, value: float) -> None:\n        with self._lock:\n            if osc == 1:\n                self.osc1_phase = value\n            elif osc == 2:\n                self.osc2_phase = value\n\n    def note_on(self, midi_note: int) -> None:\n        with self._lock:\n            freq = 440 * (2 ** ((midi_note - 69) / 12))\n            self.active_notes[midi_note] = {\n                \'freq\': freq, \n                \'phase1\': self.osc1_phase * np.pi * 2, \n                \'phase2\': self.osc2_phase * np.pi * 2, \n                \'vel\': 0.8,\n                \'start_time\': time.time(),\n                \'envelope_stage\': \'attack\',\n                \'envelope_value\': 0.0,\n                \'release_start\': None\n            }\n\n    def note_off(self, midi_note: int) -> None:\n        with self._lock:\n            if midi_note in self.active_notes:\n                note = self.active_notes[midi_note]\n                if note[\'envelope_stage\'] != \'release\':\n                    note[\'envelope_stage\'] = \'release\'\n                    note[\'release_start\'] = time.time()\n\n    def _get_envelope_value(self, note: Dict, current_time: float) -> float:\n        """Calculate current envelope value based on ADSR stage"""\n        elapsed = current_time - note[\'start_time\']\n\n        if note[\'envelope_stage\'] == \'attack\':\n            if elapsed < self.attack_time:\n                return elapsed / self.attack_time\n            else:\n                note[\'envelope_stage\'] = \'decay\'\n                elapsed -= self.attack_time\n\n        if note[\'envelope_stage\'] == \'decay\':\n            if elapsed < self.decay_time:\n                decay_amount = 1.0 - self.sustain_level\n                return 1.0 - decay_amount * (elapsed / self.decay_time)\n            else:\n                note[\'envelope_stage\'] = \'sustain\'\n                return self.sustain_level\n\n        if note[\'envelope_stage\'] == \'sustain\':\n            return self.sustain_level\n\n        if note[\'envelope_stage\'] == \'release\':\n            if note[\'release_start\'] is None:\n                return 0.0\n            release_elapsed = current_time - note[\'release_start\']\n            if release_elapsed >= self.release_time:\n                return 0.0\n            else:\n                sustain_value = note.get(\'envelope_value\', self.sustain_level)\n                return sustain_value * (1.0 - release_elapsed / self.release_time)\n\n        return 0.0\n\n    def set_effect_param(self, name: str, value: float) -> None:\n        with self._lock:\n            if name == \'distortion\': self.fx_distortion = value\n            elif name == \'delay_time\': self.fx_delay_time = value\n            elif name == \'delay_feedback\': self.fx_delay_feedback = value\n            elif name == \'delay_mix\': self.fx_delay_mix = value\n            elif name == \'reverb_mix\': self.fx_reverb_mix = value\n\n    def _apply_effects(self, buffer: np.ndarray) -> np.ndarray:\n        if self.fx_distortion > 0.01:\n            drive = 1.0 + (self.fx_distortion * 10.0)\n            buffer = np.tanh(buffer * drive)\n        if self.fx_delay_mix > 0.01 or self.fx_reverb_mix > 0.01:\n            delay_samples = int(self.fx_delay_time * self.sample_rate)\n            delay_samples = max(1, min(delay_samples, self.delay_buffer_size - 1))\n            wet_mix = max(self.fx_delay_mix, self.fx_reverb_mix)\n            feedback = self.fx_delay_feedback\n            if self.fx_reverb_mix > 0.5: feedback = 0.7 + (self.fx_reverb_mix * 0.25)\n            output = np.zeros_like(buffer)\n            for i in range(len(buffer)):\n                read_idx = int(self.delay_write_idx - delay_samples)\n                if read_idx < 0: read_idx += self.delay_buffer_size\n                delayed_sample = self.delay_buffer[read_idx]\n                self.delay_buffer[self.delay_write_idx] = buffer[i] + (delayed_sample * feedback)\n                output[i] = (buffer[i] * (1.0 - wet_mix)) + (delayed_sample * wet_mix)\n                self.delay_write_idx += 1\n                if self.delay_write_idx >= self.delay_buffer_size: self.delay_write_idx = 0\n            buffer = output\n        return buffer\n\n    def _generate_wave(self, phases: np.ndarray, wave_type: str, detune_cents: float) -> np.ndarray:\n        if detune_cents != 0.0:\n            detune_ratio = 2 ** (detune_cents / 1200.0)\n            phases = (phases * detune_ratio) % 1.0\n        \n        if wave_type == \'sine\':\n            return np.sin(2 * np.pi * phases)\n        elif wave_type == \'square\':\n            return np.where(phases < 0.5, 1.0, -1.0)\n        elif wave_type == \'saw\':\n            return 2 * (phases - 0.5)\n        elif wave_type == \'triangle\':\n            return 2 * np.abs(2 * (phases - 0.5)) - 1\n        return np.sin(2 * np.pi * phases)\n\n    def generate_audio(self, num_samples: int) -> np.ndarray:\n        buffer = np.zeros(num_samples, dtype=np.float32)\n        current_time = time.time()\n        \n        with self._lock:\n            active_count = len(self.active_notes)\n            \n            if active_count == 0:\n                buffer = self._apply_effects(buffer)\n                self.last_buffer = buffer\n                return buffer\n            \n            notes_to_remove = []\n            \n            for midi_note, note_data in list(self.active_notes.items()):\n                envelope = self._get_envelope_value(note_data, current_time)\n                note_data[\'envelope_value\'] = envelope\n\n                if envelope <= 0.001:  # Very quiet, remove note\n                    notes_to_remove.append(midi_note)\n                    continue\n\n                freq = note_data[\'freq\']\n                phase1 = note_data[\'phase1\']\n                phase2 = note_data[\'phase2\']\n                vel = note_data[\'vel\']\n                \n                t = np.arange(num_samples)\n                \n                phases1 = (phase1 / (np.pi * 2) + (freq / self.sample_rate) * t) % 1.0\n                phases2 = (phase2 / (np.pi * 2) + (freq / self.sample_rate) * t) % 1.0\n                \n                wave1 = self.waveform_type if self.use_global_waveform else self.osc1_wave\n                wave2 = self.waveform_type if self.use_global_waveform else self.osc2_wave\n                \n                osc1 = self._generate_wave(phases1, wave1, self.osc1_detune)\n                osc2 = self._generate_wave(phases2, wave2, self.osc2_detune)\n                \n                mix = (osc1 * self.osc1_level) + (osc2 * self.osc2_level)\n                buffer += mix * vel * envelope  # Apply envelope\n                \n                # Update phases for next buffer\n                note_data[\'phase1\'] = phases1[-1] * 2 * np.pi\n                note_data[\'phase2\'] = phases2[-1] * 2 * np.pi\n            \n            # Remove finished notes\n            for midi_note in notes_to_remove:\n                del self.active_notes[midi_note]\n            \n            if active_count > 1:\n                max_val = np.max(np.abs(buffer))\n                if max_val > 0: buffer = buffer / max_val * 0.9\n            buffer *= AMP\n        \n        buffer = self._apply_effects(buffer)\n        buffer = np.clip(buffer, -1.0, 1.0)\n        self.last_buffer = buffer\n        return buffer\n\n\nclass RotaryKnob(Canvas):\n    def __init__(self, parent, value=0.0, min_val=0.0, max_val=1.0, \n                 label="", format_fn=None, on_change=None, size=52, **kw):\n        super().__init__(parent, width=size + 40, height=size + 60, \n                        bg=BG_BLACK, highlightthickness=0, **kw)\n        \n        self.value = value\n        self.default_value = value\n        self.min_val = min_val\n        self.max_val = max_val\n        self.label = label\n        self.format_fn = format_fn or (lambda v: f\'{v:.0f}\')\n        self.on_change = on_change\n        self.size = size\n        self.radius = size / 2\n        self.center_x = self.radius + 20\n        self.center_y = self.radius + 30\n        self.min_angle = -135\n        self.max_angle = 135\n        self.is_dragging = False\n        self.drag_start_y = 0\n        self.drag_start_value = 0\n        \n        self.bind(\'<Button-1>\', self._on_mouse_down)\n        self.bind(\'<B1-Motion>\', self._on_mouse_drag)\n        self.bind(\'<ButtonRelease-1>\', self._on_mouse_up)\n        self.bind(\'<Double-Button-1>\', self._on_double_click)\n        self.bind(\'<MouseWheel>\', self._on_mouse_wheel)\n        self.bind(\'<Button-4>\', self._on_mouse_wheel)\n        self.bind(\'<Button-5>\', self._on_mouse_wheel)\n        \n        self.draw()\n    \n    def _get_angle(self):\n        if self.max_val == self.min_val:\n            return self.min_angle\n        normalized = (self.value - self.min_val) / (self.max_val - self.min_val)\n        return self.min_angle + normalized * (self.max_angle - self.min_angle)\n    \n    def set_value(self, value):\n        old_value = self.value\n        self.value = max(self.min_val, min(self.max_val, value))\n        if abs(self.value - old_value) > 0.001:\n            self.draw()\n            if self.on_change:\n                self.on_change(self.value)\n    \n    def reset_to_default(self):\n        if abs(self.value - self.default_value) > 0.001:\n            self.set_value(self.default_value)\n    \n    def _on_mouse_down(self, event):\n        dx = event.x - self.center_x\n        dy = event.y - self.center_y\n        distance = math.sqrt(dx**2 + dy**2)\n        if distance < self.radius * 0.8:\n            self.is_dragging = True\n            self.drag_start_y = event.y_root\n            self.drag_start_value = self.value\n            self.config(cursor=\'fleur\')\n    \n    def _on_mouse_drag(self, event):\n        if not self.is_dragging:\n            return\n        delta_y = self.drag_start_y - event.y_root\n        sensitivity = (self.max_val - self.min_val) / 200\n        self.set_value(self.drag_start_value + delta_y * sensitivity)\n    \n    def _on_mouse_up(self, event):\n        self.is_dragging = False\n        self.config(cursor=\'hand2\')\n    \n    def _on_double_click(self, event):\n        dx = event.x - self.center_x\n        dy = event.y - self.center_y\n        distance = math.sqrt(dx**2 + dy**2)\n        if distance < self.radius * 0.8:\n            self.reset_to_default()\n    \n    def _on_mouse_wheel(self, event):\n        delta = 0.02 if (event.num == 4 or event.delta > 0) else -0.02\n        self.set_value(self.value + delta * (self.max_val - self.min_val))\n        return \'break\'\n    \n    def draw(self):\n        self.delete(\'all\')\n        angle = self._get_angle()\n        angle_rad = math.radians(angle)\n        \n        self.create_oval(self.center_x - self.radius, self.center_y - self.radius,\n                        self.center_x + self.radius, self.center_y + self.radius,\n                        fill=KNOB_BODY, outline=BORDER, width=1)\n        \n        inner_r = self.radius * 0.75\n        self.create_oval(self.center_x - inner_r, self.center_y - inner_r,\n                        self.center_x + inner_r, self.center_y + inner_r,\n                        fill=KNOB_SURFACE, outline=BORDER, width=1)\n        \n        arc_radius = self.radius * 0.90\n        if self.value > self.min_val:\n            self.create_arc(self.center_x - arc_radius, self.center_y - arc_radius,\n                          self.center_x + arc_radius, self.center_y + arc_radius,\n                          start=self.min_angle, extent=angle - self.min_angle,\n                          style=\'arc\', outline=FG_BRIGHT, width=3)\n        \n        line_start_r = self.radius * 0.2\n        line_end_r = self.radius * 0.6\n        start_x = self.center_x + line_start_r * math.cos(angle_rad - math.pi/2)\n        start_y = self.center_y + line_start_r * math.sin(angle_rad - math.pi/2)\n        end_x = self.center_x + line_end_r * math.cos(angle_rad - math.pi/2)\n        end_y = self.center_y + line_end_r * math.sin(angle_rad - math.pi/2)\n        self.create_line(start_x, start_y, end_x, end_y, fill=FG_BRIGHT, width=2)\n        \n        self.create_oval(self.center_x - self.radius * 0.15, self.center_y - self.radius * 0.15,\n                        self.center_x + self.radius * 0.15, self.center_y + self.radius * 0.15,\n                        fill=BORDER, outline=KNOB_SURFACE, width=1)\n        \n        self.create_text(self.center_x, self.center_y + self.radius + 25,\n                        text=self.label.upper(), font=(FONT_MAIN, 7, \'bold\'), fill=FG_DIM)\n        self.create_text(self.center_x, 12, text=self.format_fn(self.value),\n                        font=(FONT_MAIN, 9, \'bold\'), fill=FG_WHITE)\n\n\nclass PianoKeyboard(Canvas):\n    def __init__(self, parent, start_note=60, num_octaves=2, **kw):\n        super().__init__(parent, bg=BG_BLACK, highlightthickness=0, **kw)\n        self.start_note = start_note\n        self.num_octaves = num_octaves\n        self.white_key_width = 40\n        self.black_key_width = 24\n        self.white_key_height = 120\n        self.black_key_height = 75\n        self.active_keys = set()\n        self.key_rects = {}\n        self._draw_keyboard()\n    \n    def _draw_keyboard(self):\n        self.delete(\'all\')\n        self.key_rects = {}\n        x = 10\n        white_notes = [0, 2, 4, 5, 7, 9, 11]\n        \n        for octave in range(self.num_octaves):\n            for offset in white_notes:\n                midi_note = self.start_note + (octave * 12) + offset\n                key_letter = None\n                for k, v in WHITE_KEYS.items():\n                    if v == midi_note:\n                        key_letter = k.upper()\n                        break\n                \n                rect = self.create_rectangle(x, 0, x + self.white_key_width, self.white_key_height,\n                                           fill=BG_DARK, outline=BORDER, width=1)\n                self.key_rects[midi_note] = rect\n                \n                if key_letter:\n                    self.create_text(x + self.white_key_width / 2, self.white_key_height - 20,\n                                   text=key_letter, fill=FG_DIM, font=(FONT_MAIN, 10))\n                if midi_note in OCTAVE_LABELS:\n                    self.create_text(x + self.white_key_width / 2, 15,\n                                   text=OCTAVE_LABELS[midi_note], fill=FG_BRIGHT, font=(FONT_MAIN, 9, \'bold\'))\n                x += self.white_key_width\n        \n        x = 10\n        black_notes = [(1, 0.7), (3, 1.7), (6, 3.7), (8, 4.7), (10, 5.7)]\n        for octave in range(self.num_octaves):\n            for offset, white_key_offset in black_notes:\n                midi_note = self.start_note + (octave * 12) + offset\n                if midi_note in self.key_rects:\n                    continue\n                key_letter = None\n                for k, v in BLACK_KEYS.items():\n                    if v == midi_note:\n                        key_letter = k.upper()\n                        break\n                black_x = x + (white_key_offset * self.white_key_width) - (self.black_key_width / 2)\n                rect = self.create_rectangle(black_x, 0, black_x + self.black_key_width, self.black_key_height,\n                                           fill=\'#1a1a1a\', outline=BORDER, width=1)\n                self.key_rects[midi_note] = rect\n                if key_letter:\n                    self.create_text(black_x + self.black_key_width / 2, self.black_key_height - 15,\n                                   text=key_letter, fill=FG_DIM, font=(FONT_MAIN, 9))\n            x += 7 * self.white_key_width\n        \n        self.config(width=x + 10, height=self.white_key_height + 10)\n    \n    def highlight_key(self, midi_note, active=True):\n        if midi_note not in self.key_rects:\n            return\n        rect_id = self.key_rects[midi_note]\n        if active:\n            self.itemconfig(rect_id, fill=FG_BRIGHT, outline=FG_WHITE)\n            self.active_keys.add(midi_note)\n        else:\n            is_black = midi_note in [61, 63, 66, 68, 70, 73, 75, 78, 80, 82, 85, 87]\n            self.itemconfig(rect_id, fill=\'#1a1a1a\' if is_black else BG_DARK, outline=BORDER)\n            self.active_keys.discard(midi_note)\n\n\ndef sep(parent, vertical=False):\n    return tk.Frame(parent, bg=BORDER, width=1 if vertical else 100, height=1 if not vertical else 100)\n\ndef label(parent, text, size=9, color=FG_DIM, bold=False):\n    return tk.Label(parent, text=text, bg=BG_BLACK, fg=color, font=(FONT_MAIN, size, \'bold\' if bold else \'normal\'))\n\n\nclass SerumApp:\n    def __init__(self, root):\n        self.root = root\n        root.title(\'Nill OSC\')\n        root.configure(bg=BG_BLACK)\n        root.geometry(\'1200x900\')\n        root.minsize(1000, 800)\n        \n        self.synth = PolyphonicSynth()\n        self._wt_canvases = []\n        self._playing_notes = {}\n        self._waveform_var = tk.StringVar(value=\'sine\')\n        self._waveform_btns = {}\n        self._waveform_index = 0\n        self._osc1_wave_var = tk.StringVar(value=\'sine\')\n        self._osc2_wave_var = tk.StringVar(value=\'saw\')\n        \n        self.stream = None\n        self.display_canvas = None\n        self.animation_id = None\n        self.piano_keyboard = None\n        \n        self._build()\n        root.after(100, self._init_mini_waves)\n        root.after(100, self._start_audio)\n        root.bind(\'<KeyPress>\', self._on_key_press)\n        root.bind(\'<KeyRelease>\', self._on_key_release)\n        root.protocol(\'WM_DELETE_WINDOW\', self._on_closing)\n\n    def _start_audio(self):\n        try:\n            self.stream = sd.OutputStream(channels=1, samplerate=SR, blocksize=BUFFER_SIZE, callback=self._audio_callback)\n            self.stream.start()\n            print("[SYSTEM] Audio stream started")\n        except Exception as e:\n            print(f"[ERROR] Audio: {e}")\n\n    def _audio_callback(self, outdata, frames, time_info, status):\n        if status:\n            print(f"[STATUS] {status}")\n        outdata[:, 0] = self.synth.generate_audio(frames)\n\n    def _on_closing(self):\n        if self.stream:\n            self.stream.stop()\n            self.stream.close()\n        if self.animation_id:\n            self.root.after_cancel(self.animation_id)\n        self.root.destroy()\n\n    def _build(self):\n        self._build_titlebar()\n        body = tk.Frame(self.root, bg=BG_BLACK)\n        body.pack(fill=\'both\', expand=True)\n        self._build_left(body)\n        self._build_right(body)\n        self._build_center(body)\n        self._build_keyboard()\n        self._build_bottombar()\n\n    def _build_titlebar(self):\n        bar = tk.Frame(self.root, bg=BG_BLACK, height=35)\n        bar.pack(fill=\'x\')\n        bar.pack_propagate(False)\n        sep(bar).pack(side=\'bottom\', fill=\'x\')\n        label(bar, \'Nill OSC\', size=11, color=FG_WHITE, bold=True).pack(side=\'left\', padx=15)\n        right = tk.Frame(bar, bg=BG_BLACK)\n        right.pack(side=\'right\', padx=15)\n        label(right, \'OSC 1 + OSC 2\', size=9).pack(side=\'left\', padx=15)\n\n    def _build_bottombar(self):\n        bar = tk.Frame(self.root, bg=BG_BLACK, height=28)\n        bar.pack(fill=\'x\', side=\'bottom\')\n        bar.pack_propagate(False)\n        sep(bar).pack(side=\'top\', fill=\'x\')\n        label(bar, \'PRESET: DEFAULT_PATCH\', size=8).pack(side=\'left\', padx=15)\n        self.cpu_label = label(bar, \'VOICES: 0/16\', size=8)\n        self.cpu_label.pack(side=\'right\', padx=15)\n\n    def _build_keyboard(self):\n        kb_frame = tk.Frame(self.root, bg=BG_BLACK, height=140)\n        kb_frame.pack(fill=\'x\', side=\'bottom\')\n        sep(kb_frame).pack(side=\'top\', fill=\'x\')\n        self.piano_keyboard = PianoKeyboard(kb_frame, start_note=60, num_octaves=2)\n        self.piano_keyboard.pack(pady=10)\n\n    def _build_left(self, parent):\n        """Left panel with OSC 1 & OSC 2 - WORKING SCROLLBAR"""\n        outer = tk.Frame(parent, bg=BG_BLACK, width=320)\n        outer.pack(side=\'left\', fill=\'y\')\n        outer.pack_propagate(False)\n        sep(outer, vertical=True).pack(side=\'right\', fill=\'y\')\n        \n        # Canvas with scrollbar\n        cv = tk.Canvas(outer, bg=BG_BLACK, highlightthickness=0)\n        sb = tk.Scrollbar(outer, orient=\'vertical\', command=cv.yview, \n                         bg=BG_BLACK, troughcolor=BG_DARK, activebackground=FG_DIM)\n        inner = tk.Frame(cv, bg=BG_BLACK)\n        \n        inner.bind(\'<Configure>\', lambda e: cv.configure(scrollregion=cv.bbox(\'all\')))\n        win_id = cv.create_window((0, 0), window=inner, anchor=\'nw\')\n        cv.configure(yscrollcommand=sb.set)\n        cv.bind(\'<Configure>\', lambda e: cv.itemconfig(win_id, width=e.width))\n        \n        # Mouse wheel scrolling\n        def _on_mousewheel(event):\n            cv.yview_scroll(int(-1*(event.delta/120)), \'units\')\n        cv.bind(\'<MouseWheel>\', _on_mousewheel)\n        cv.bind(\'<Button-4>\', lambda e: cv.yview_scroll(-1, \'units\'))\n        cv.bind(\'<Button-5>\', lambda e: cv.yview_scroll(1, \'units\'))\n        \n        cv.pack(side=\'left\', fill=\'both\', expand=True)\n        sb.pack(side=\'right\', fill=\'y\')\n        \n        # Build sections\n        self._build_waveform_selector(inner)\n        self._build_osc(inner, 1, \'OSC 1\', \'sine\', 0, 100, 0)\n        self._build_osc(inner, 2, \'OSC 2\', \'saw\', 0, 0, 0)\n\n    def _build_waveform_selector(self, parent):\n        sep(parent).pack(fill=\'x\')\n        sec = tk.Frame(parent, bg=BG_BLACK)\n        sec.pack(fill=\'x\', padx=12, pady=10)\n        \n        label(sec, \'[GLOBAL WAVEFORM]\', size=9, color=FG_WHITE, bold=True).pack(anchor=\'w\', pady=(0, 6))\n        label(sec, \'M=Next N=Previous (overrides OSC)\', size=7, color=FG_DIM).pack(anchor=\'w\', pady=(0, 8))\n        \n        btn_frame = tk.Frame(sec, bg=BG_BLACK)\n        btn_frame.pack(fill=\'x\')\n        \n        waveforms = [(\'sine\', \'SINE\'), (\'triangle\', \'TRI\'), (\'saw\', \'SAW\'), (\'square\', \'SQR\')]\n        for i, (wave, text) in enumerate(waveforms):\n            row = i // 2\n            col = i % 2\n            is_active = (wave == \'sine\')\n            btn = tk.Button(btn_frame, text=f\'[{text}]\', bg=FG_BRIGHT if is_active else BORDER,\n                          fg=BG_BLACK if is_active else FG_DIM, font=(FONT_MAIN, 8, \'bold\'),\n                          relief=\'flat\', width=6, height=2, cursor=\'hand2\',\n                          command=lambda w=wave: self._set_global_waveform(w))\n            btn.grid(row=row, column=col, padx=4, pady=4, sticky=\'ew\')\n            self._waveform_btns[wave] = btn\n        \n        btn_frame.columnconfigure(0, weight=1)\n        btn_frame.columnconfigure(1, weight=1)\n\n    def _set_global_waveform(self, waveform: str):\n        if waveform not in WAVEFORMS:\n            return\n        self.synth.use_global_waveform = True\n        self.synth.set_waveform(waveform)\n        self._waveform_index = WAVEFORMS.index(waveform)\n        for wave, btn in self._waveform_btns.items():\n            btn.config(bg=FG_BRIGHT if wave == waveform else BORDER, \n                      fg=BG_BLACK if wave == waveform else FG_DIM)\n\n    def _cycle_waveform(self, direction: int):\n        self._waveform_index = (self._waveform_index + direction) % len(WAVEFORMS)\n        self._set_global_waveform(WAVEFORMS[self._waveform_index])\n\n    def _build_osc(self, parent, osc_num, name, wave_type, detune, level, phase):\n        """Build OSC section with waveform buttons + 4 knobs"""\n        sep(parent).pack(fill=\'x\')\n        sec = tk.Frame(parent, bg=BG_BLACK)\n        sec.pack(fill=\'x\', padx=12, pady=10)\n        \n        # Header\n        hdr = tk.Frame(sec, bg=BG_BLACK)\n        hdr.pack(fill=\'x\', pady=(0, 8))\n        label(hdr, f\'[{name}]\', size=9, color=FG_WHITE, bold=True).pack(side=\'left\')\n        \n        on_var = tk.BooleanVar(value=True)\n        tog_btn = tk.Button(hdr, text=\'[ON]\', bg=FG_BRIGHT, fg=BG_BLACK, font=(FONT_MAIN, 8),\n                           relief=\'flat\', width=3, cursor=\'hand2\')\n        tog_btn.pack(side=\'right\')\n        \n        # Wavetable display\n        wt_canvas = tk.Canvas(sec, width=280, height=50, bg=BG_BLACK, \n                             highlightthickness=1, highlightbackground=BORDER)\n        wt_canvas.pack(pady=(0, 8))\n        self._wt_canvases.append((wt_canvas, wave_type, 2.0))\n        \n        # Waveform buttons\n        wave_frame = tk.Frame(sec, bg=BG_BLACK)\n        wave_frame.pack(fill=\'x\', pady=(0, 8))\n        label(wave_frame, \'WAVEFORM:\', size=7, color=FG_DIM).pack(anchor=\'w\')\n        \n        wave_btns = tk.Frame(wave_frame, bg=BG_BLACK)\n        wave_btns.pack(fill=\'x\')\n        for wave, text in [(\'sine\', \'SIN\'), (\'triangle\', \'TRI\'), (\'saw\', \'SAW\'), (\'square\', \'SQR\')]:\n            btn = tk.Button(wave_btns, text=text, bg=FG_BRIGHT if wave == wave_type else BORDER,\n                          fg=BG_BLACK if wave == wave_type else FG_DIM, font=(FONT_MAIN, 7),\n                          relief=\'flat\', width=5, cursor=\'hand2\',\n                          command=lambda w=wave: self._set_osc_wave(osc_num, w))\n            btn.pack(side=\'left\', padx=2)\n        \n        # 4 Knobs in a row: Detune, Level, Phase\n        knob_frame = tk.Frame(sec, bg=BG_BLACK)\n        knob_frame.pack(fill=\'x\')\n        for i, (lbl, init, min_v, max_v, fmt, param) in enumerate([\n            (\'DET\', 0.5, 0.0, 1.0, lambda v: f\'{int((v-0.5)*100):+d}\', \'detune\'),\n            (\'LEV\', 1.0 if osc_num == 1 else 0.0, 0.0, 1.0, lambda v: f\'{int(v*100)}%\', \'level\'),\n            (\'PHS\', 0.0, 0.0, 1.0, lambda v: f\'{int(v*360)}°\', \'phase\')\n        ]):\n            ctrl = tk.Frame(knob_frame, bg=BG_BLACK)\n            ctrl.pack(side=\'left\', expand=True, fill=\'x\', padx=3)\n            \n            val_lbl = label(ctrl, fmt(init), size=9, color=FG_WHITE, bold=True)\n            val_lbl.pack()\n            \n            knob = RotaryKnob(ctrl, value=init, min_val=min_v, max_val=max_v,\n                            label=lbl, size=52, format_fn=fmt,\n                            on_change=lambda v, o=osc_num, p=param, l=val_lbl: self._on_osc_change(o, p, v, l))\n            knob.pack()\n\n    def _set_osc_wave(self, osc_num, waveform):\n        self.synth.use_global_waveform = False\n        self.synth.set_osc_wave(osc_num, waveform)\n    \n    def _on_osc_change(self, osc_num, param, value, label):\n        if param == \'detune\':\n            label.config(text=f\'{int((value-0.5)*100):+d}\')\n            self.synth.set_osc_detune(osc_num, int((value - 0.5) * 100))\n        elif param == \'level\':\n            label.config(text=f\'{int(value*100)}%\')\n            self.synth.set_osc_level(osc_num, value)\n        elif param == \'phase\':\n            label.config(text=f\'{int(value*360)}°\')\n            self.synth.set_osc_phase(osc_num, value)\n\n    def _build_right(self, parent):\n        """Right panel - Master, Filter, FX (as before)"""\n        outer = tk.Frame(parent, bg=BG_BLACK, width=320)\n        outer.pack(side=\'right\', fill=\'y\')\n        outer.pack_propagate(False)\n        sep(outer, vertical=True).pack(side=\'left\', fill=\'y\')\n        \n        cv = tk.Canvas(outer, bg=BG_BLACK, highlightthickness=0)\n        sb = tk.Scrollbar(outer, orient=\'vertical\', command=cv.yview, \n                         bg=BG_BLACK, troughcolor=BG_DARK, activebackground=FG_DIM)\n        inner = tk.Frame(cv, bg=BG_BLACK)\n        \n        inner.bind(\'<Configure>\', lambda e: cv.configure(scrollregion=cv.bbox(\'all\')))\n        win_id = cv.create_window((0, 0), window=inner, anchor=\'nw\')\n        cv.configure(yscrollcommand=sb.set)\n        cv.bind(\'<Configure>\', lambda e: cv.itemconfig(win_id, width=e.width))\n        cv.bind(\'<MouseWheel>\', lambda e: cv.yview_scroll(int(-1*(e.delta/120)), \'units\'))\n        cv.bind(\'<Button-4>\', lambda e: cv.yview_scroll(-1, \'units\'))\n        cv.bind(\'<Button-5>\', lambda e: cv.yview_scroll(1, \'units\'))\n        \n        cv.pack(side=\'left\', fill=\'both\', expand=True)\n        sb.pack(side=\'right\', fill=\'y\')\n        \n        self._build_master(inner)\n        self._build_filter(inner)\n        self._build_effects(inner)\n\n    def _build_master(self, parent):\n        sep(parent).pack(fill=\'x\')\n        sec = tk.Frame(parent, bg=BG_BLACK)\n        sec.pack(fill=\'x\', padx=12, pady=10)\n        label(sec, \'[MASTER]\', size=9, color=FG_WHITE, bold=True).pack(anchor=\'w\', pady=(0, 8))\n        \n        row = tk.Frame(sec, bg=BG_BLACK)\n        row.pack(fill=\'x\')\n        for k in [\n            {\'key\': \'master_pitch\', \'label\': \'PITCH\', \'initial\': 0.5, \'format\': lambda v: f\'{(v-0.5)*24:+.1f}\'},\n            {\'key\': \'master_glide\', \'label\': \'GLIDE\', \'initial\': 0.0, \'format\': lambda v: f\'{int(v*100)}\'},\n            {\'key\': \'master_level\', \'label\': \'LEVEL\', \'initial\': 1.0, \'format\': lambda v: f\'{int(v*127)}\'}\n        ]:\n            self._create_knob_column(row, k)\n\n    def _create_knob_column(self, parent, knob_def):\n        f = tk.Frame(parent, bg=BG_BLACK)\n        f.pack(side=\'left\', expand=True, fill=\'x\')\n        disp = label(f, knob_def[\'format\'](knob_def[\'initial\']), size=9, color=FG_WHITE, bold=True)\n        disp.pack(pady=(0, 4))\n        knob = RotaryKnob(f, value=knob_def[\'initial\'], min_val=0.0, max_val=1.0,\n                         label=knob_def[\'label\'], size=52, format_fn=knob_def[\'format\'])\n        knob.pack()\n        label(f, knob_def[\'label\'], size=7, color=FG_DIM).pack(pady=(4, 0))\n\n    def _build_filter(self, parent):\n        sep(parent).pack(fill=\'x\')\n        sec = tk.Frame(parent, bg=BG_BLACK)\n        sec.pack(fill=\'x\', padx=12, pady=10)\n        label(sec, \'[FILTER]\', size=9, color=FG_WHITE, bold=True).pack(anchor=\'w\', pady=(0, 6))\n        \n        btn_row = tk.Frame(sec, bg=BG_BLACK)\n        btn_row.pack(fill=\'x\', pady=(0, 8))\n        for ft in [\'LP\', \'HP\', \'BP\', \'NT\']:\n            b = tk.Button(btn_row, text=ft, font=(FONT_MAIN, 8), relief=\'flat\', \n                         bg=BORDER, fg=FG_DIM, activebackground=BG_BLACK, \n                         activeforeground=FG_WHITE, cursor=\'hand2\', width=3)\n            b.pack(side=\'left\', padx=3)\n        \n        row = tk.Frame(sec, bg=BG_BLACK)\n        row.pack(fill=\'x\')\n        for k in [\n            {\'key\': \'filter_cutoff\', \'label\': \'CUTOFF\', \'initial\': 0.5, \'format\': lambda v: f\'{int(v*100)}\'},\n            {\'key\': \'filter_res\', \'label\': \'RES\', \'initial\': 0.3, \'format\': lambda v: f\'{int(v*100)}\'},\n            {\'key\': \'filter_drive\', \'label\': \'DRV\', \'initial\': 0.0, \'format\': lambda v: f\'{int(v*100)}\'}\n        ]:\n            self._create_knob_column(row, k)\n\n    def _build_effects(self, parent):\n        sep(parent).pack(fill=\'x\')\n        sec = tk.Frame(parent, bg=BG_BLACK)\n        sec.pack(fill=\'x\', padx=12, pady=10)\n        label(sec, \'[FX]\', size=9, color=FG_WHITE, bold=True).pack(anchor=\'w\', pady=(0, 6))\n        \n        for fx_name, params in [\n            (\'DISTORTION\', [{\'key\': \'fx_dist\', \'label\': \'AMT\', \'initial\': 0.0, \'format\': lambda v: f\'{int(v*100)}%\'}]),\n            (\'DELAY\', [\n                {\'key\': \'fx_delay_time\', \'label\': \'TIME\', \'initial\': 0.3, \'format\': lambda v: f\'{v:.2f}s\'},\n                {\'key\': \'fx_delay_fdbk\', \'label\': \'FDBK\', \'initial\': 0.4, \'format\': lambda v: f\'{int(v*100)}%\'},\n                {\'key\': \'fx_delay_mix\', \'label\': \'MIX\', \'initial\': 0.0, \'format\': lambda v: f\'{int(v*100)}%\'}\n            ]),\n            (\'REVERB\', [{\'key\': \'fx_reverb\', \'label\': \'AMT\', \'initial\': 0.0, \'format\': lambda v: f\'{int(v*100)}%\'}])\n        ]:\n            box = tk.Frame(sec, bg=BG_PANEL, bd=1, relief=\'solid\')\n            box.pack(fill=\'x\', pady=2)\n            label(tk.Frame(box, bg=BG_PANEL), fx_name, size=7).pack(side=\'left\', padx=5, pady=3)\n            row = tk.Frame(box, bg=BG_PANEL)\n            row.pack(fill=\'x\', padx=5, pady=(0, 4))\n            for k in params:\n                self._create_knob_column(row, k)\n\n    def _build_center(self, parent):\n        frame = tk.Frame(parent, bg=BG_BLACK)\n        frame.pack(side=\'left\', fill=\'both\', expand=True)\n        \n        hdr = tk.Frame(frame, bg=BG_BLACK, height=35)\n        hdr.pack(fill=\'x\')\n        hdr.pack_propagate(False)\n        sep(hdr).pack(side=\'bottom\', fill=\'x\')\n        label(hdr, \'VISUALIZER\', size=9, color=FG_WHITE, bold=True).pack(side=\'left\', padx=12, pady=8)\n        \n        content = tk.Frame(frame, bg=BG_BLACK)\n        content.pack(fill=\'both\', expand=True, padx=12, pady=12)\n        \n        self.display_canvas = tk.Canvas(content, bg=BG_BLACK, highlightthickness=1, highlightbackground=BORDER)\n        self.display_canvas.pack(fill=\'both\', expand=True)\n        \n        self._animate_wavetable()\n\n    def _animate_wavetable(self):\n        if not self.display_canvas:\n            return\n        w = self.display_canvas.winfo_width()\n        h = self.display_canvas.winfo_height()\n        \n        if w > 10 and h > 10:\n            self.display_canvas.delete(\'all\')\n            for x in range(0, w + 40, 40):\n                self.display_canvas.create_line(x, 0, x, h, fill=BORDER)\n            for y in range(0, h + 40, 40):\n                self.display_canvas.create_line(0, y, w, y, fill=BORDER)\n            self.display_canvas.create_line(0, h // 2, w, h // 2, fill=FG_DIM, width=1)\n            \n            audio = self.synth.last_buffer\n            pts = []\n            downsample = max(1, len(audio) // w)\n            for i in range(0, len(audio), downsample):\n                x = int((i / len(audio)) * w)\n                y = h // 2 - int(audio[i] * (h / 2.5))\n                pts.extend([x, y])\n            if len(pts) >= 4:\n                self.display_canvas.create_line(pts, fill=FG_WHITE, width=1, smooth=True)\n        \n        self.animation_id = self.root.after(33, self._animate_wavetable)\n\n    def _on_key_press(self, e):\n        key = e.keysym.lower()\n        if key == \'m\':\n            self._cycle_waveform(1)\n            return\n        elif key == \'n\':\n            self._cycle_waveform(-1)\n            return\n        if key in ALL_KEYS and key not in self._playing_notes:\n            midi_note = ALL_KEYS[key]\n            self._playing_notes[key] = midi_note\n            self.synth.note_on(midi_note)\n            if self.piano_keyboard:\n                self.piano_keyboard.highlight_key(midi_note, True)\n\n    def _on_key_release(self, e):\n        key = e.keysym.lower()\n        if key in self._playing_notes:\n            midi_note = self._playing_notes.pop(key)\n            self.synth.note_off(midi_note)\n            if self.piano_keyboard:\n                self.piano_keyboard.highlight_key(midi_note, False)\n\n    def _init_mini_waves(self):\n        for canvas, wtype, freq in self._wt_canvases:\n            canvas.delete(\'all\')\n            canvas.configure(bg=BG_BLACK)\n            w = canvas.winfo_width() or 280\n            h = canvas.winfo_height() or 50\n            for i in range(5):\n                canvas.create_line(0, int(h / 4 * i), w, int(h / 4 * i), fill=BORDER, width=1)\n            pts = []\n            for x in range(w):\n                t = (x / w) * math.pi * 2 * freq\n                if wtype == \'sine\':\n                    y = math.sin(t)\n                elif wtype == \'saw\':\n                    y = 2 * ((t % (math.pi * 2)) / (math.pi * 2)) - 1\n                elif wtype == \'square\':\n                    y = 1 if t % (math.pi * 2) < math.pi else -1\n                else:\n                    y = 2 * abs(2 * ((t % (math.pi * 2)) / (math.pi * 2)) - 1) - 1\n                pts.extend([x, h / 2 - y * h / 2.5])\n            if len(pts) >= 4:\n                canvas.create_line(pts, fill=FG_WHITE, width=1, smooth=True)\n\n\nif __name__ == \'__main__\':\n    root = tk.Tk()\n    try:\n        from ctypes import windll\n        windll.shcore.SetProcessDpiAwareness(1)\n    except Exception:\n        pass\n    app = SerumApp(root)\n    root.mainloop()\n'
def run_embedded_osc() -> None:
    """Embedded version of PAGE404MAX/noir-terminal testwavetable.py."""
    namespace = {"__name__": "__main__", "__file__": str(Path(__file__))}
    exec(EMBEDDED_OSC_CODE, namespace)

def run_embedded_visualizer() -> None:
    """Embedded version of PAGE404MAX/noir-terminal Visual."""
    try:
        import pygame
    except ImportError:
        try:
            import mygame as pygame
        except ImportError:
            print("✗ Missing required module: pygame")
            print("Install it with 'pip install pygame' and try again.")
            return

    if sd is None:
        print("✗ Missing required module or audio backend: sounddevice")
        print("Install it with 'pip install sounddevice' and try again.")
        return

    # =========================
    # CONFIG
    # =========================
    NATIVE_W, NATIVE_H = 600, 400
    SAMPLE_RATE = 44100
    BLOCK_SIZE = 2048
    GHOST_TRAIL_COUNT = 6
    GHOST_FADE = 0.12
    FRAME_RADIUS = 20

    # AUDIO DEVICE DETECTION
    print("Scanning for audio devices...")
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"✗ Could not query audio devices: {exc}")
        return

    AUDIO_DEVICE = None
    stereo_mix_keywords = [
        "stereo mix", "what u hear", "loopback", "wave out",
        "out mix", "realtek stereo", "virtual audio",
    ]
    for i, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            name_lower = str(device.get("name", "")).lower()
            if any(keyword in name_lower for keyword in stereo_mix_keywords):
                AUDIO_DEVICE = i
                print(f"✓ Found: [{i}] {device['name']}")
                break

    if AUDIO_DEVICE is None:
        print("✗ No Stereo Mix detected!")
        print("Using default input device...")

    pygame.init()
    screen = pygame.display.set_mode((NATIVE_W, NATIVE_H))
    pygame.display.set_caption("Ear Candy")
    clock = pygame.time.Clock()

    print(f"Initializing audio device [{AUDIO_DEVICE}]...")
    audio_block = np.zeros((BLOCK_SIZE, 2), dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_block
        audio_block = indata.copy()

    try:
        stream = sd.InputStream(
            device=AUDIO_DEVICE,
            callback=audio_callback,
            blocksize=BLOCK_SIZE,
            channels=2,
            samplerate=SAMPLE_RATE,
        )
        stream.start()
        print("✓ Audio initialized successfully")
    except Exception as exc:
        print(f"✗ Audio error: {exc}")
        pygame.quit()
        return

    def catmull_rom(points, samples=8):
        if len(points) < 4:
            return points
        smoothed = []
        for i in range(1, len(points) - 2):
            p0, p1, p2, p3 = points[i - 1], points[i], points[i + 1], points[i + 2]
            for t in np.linspace(0, 1, samples, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                x = 0.5 * (
                    (2 * p1[0])
                    + (-p0[0] + p2[0]) * t
                    + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                    + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    (2 * p1[1])
                    + (-p0[1] + p2[1]) * t
                    + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                    + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                )
                smoothed.append((x, y))
        smoothed.append(points[-2])
        smoothed.append(points[-1])
        return smoothed

    visual_surface = pygame.Surface((NATIVE_W, NATIVE_H), pygame.SRCALPHA)
    blur_surface = pygame.Surface((NATIVE_W, NATIVE_H), pygame.SRCALPHA)
    ghost_trails = []

    frame_rect = pygame.Rect(15, 15, NATIVE_W - 30, NATIVE_H - 30)
    frame_surface = pygame.Surface((NATIVE_W, NATIVE_H), pygame.SRCALPHA)
    pygame.draw.rect(frame_surface, (255, 255, 255), frame_rect, width=3, border_radius=FRAME_RADIUS)

    clip_surface = pygame.Surface((NATIVE_W, NATIVE_H), pygame.SRCALPHA)
    pygame.draw.rect(clip_surface, (255, 255, 255), frame_rect, border_radius=FRAME_RADIUS)

    PANEL_HEIGHT = 80

    def draw_glass_panel(surface):
        glass = pygame.Surface((NATIVE_W, PANEL_HEIGHT), pygame.SRCALPHA)
        glass.fill((30, 30, 30, 160))
        surface.blit(glass, (0, 0))

        reflection = pygame.Surface((NATIVE_W, PANEL_HEIGHT // 2), pygame.SRCALPHA)
        for y in range(reflection.get_height()):
            alpha = int(80 * (1 - y / reflection.get_height()))
            pygame.draw.line(reflection, (255, 255, 255, alpha), (0, y), (NATIVE_W, y))
        surface.blit(reflection, (0, 0))

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            blur_surface.fill((0, 0, 0, 40))
            visual_surface.blit(blur_surface, (0, 0))

            mono = np.mean(audio_block, axis=1)
            audio_level = np.mean(np.abs(mono))
            audio_level_boosted = audio_level * 10

            if audio_level_boosted > 0.001:
                max_val = np.max(np.abs(mono))
                if max_val > 0:
                    mono = mono / max_val
            else:
                t = pygame.time.get_ticks() / 1000.0
                mono = np.sin(np.linspace(0, 2 * np.pi, len(mono)) + t * 0.5) * 0.3

            mono = np.interp(
                np.linspace(0, len(mono), frame_rect.width),
                np.arange(len(mono)),
                mono,
            )

            points = []
            mid_y = frame_rect.y + frame_rect.height / 2
            amp = frame_rect.height * 0.45
            for i, sample in enumerate(mono):
                x = frame_rect.x + i
                y = int(mid_y + sample * amp)
                points.append((x, y))

            if len(points) > 4:
                points = catmull_rom(points, samples=4)

            ghost_trails.insert(0, points)
            if len(ghost_trails) > GHOST_TRAIL_COUNT:
                ghost_trails.pop()
            if audio_level_boosted <= 0.001:
                ghost_trails.clear()

            wave_surface = pygame.Surface((NATIVE_W, NATIVE_H), pygame.SRCALPHA)
            for idx, trail in enumerate(ghost_trails):
                alpha = int(255 * max(0, 1 - idx * GHOST_FADE))
                if len(trail) > 1:
                    pygame.draw.lines(wave_surface, (255, 255, 255, alpha), False, trail, 2)

            if len(points) > 1:
                pygame.draw.lines(wave_surface, (255, 255, 255, 255), False, points, 5)

            mask = pygame.mask.from_surface(clip_surface)
            mask_surf = mask.to_surface(setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0))
            wave_surface.blit(mask_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

            inner_bg = pygame.Surface((frame_rect.width, frame_rect.height), pygame.SRCALPHA)
            inner_bg.fill((10, 10, 10))
            visual_surface.blit(inner_bg, frame_rect.topleft)
            visual_surface.blit(wave_surface, (0, 0))
            visual_surface.blit(frame_surface, (0, 0))
            draw_glass_panel(visual_surface)

            screen.fill((0, 0, 0))
            screen.blit(visual_surface, (0, 0))
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--nill-visualizer":
        run_embedded_visualizer()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--nill-osc":
        run_embedded_osc()
        return

    print("Nill | Black & White Terminal Open Source DAW")
    print("Features: pattern editor, FL-style piano roll, black/white terminal UI, ghost notes, scale highlighting, track rack.")
    app = QApplication(sys.argv)
    window = Nill()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
