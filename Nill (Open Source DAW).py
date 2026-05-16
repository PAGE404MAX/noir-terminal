#!/usr/bin/env python3
"""
NILL | Open Source DAW

- Horizontal timeline with pattern blocks/clips on tracks
- Drag clips to move, resize edges, double-click to edit in piano roll
- Reusable patterns: edit once, all clips update
- Track mute/solo
- Smooth playback with optimized clip lookup
- Per-pattern piano roll
- Note preview minimap inside playlist clip blocks
- FL Studio loop-paint: left/right-click drag stamps/extends clips
- Song loop region: drag handles on the ruler bar, ⟲ button to enable

Requirements:
    pip install PySide6 sounddevice numpy
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

from PySide6.QtCore import Qt, QTimer, QRectF, Signal, QEvent, QSize, QPoint
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QAction, QPolygon
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QMenu, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

# ============================ CONSTANTS ============================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PATTERN_COLORS = [
    "#E6E6E6", "#CFCFCF", "#B8B8B8", "#FFFFFF",
    "#8A8A8A", "#737373", "#5C5C5C", "#444444",
    "#AAAAAA", "#666666", "#999999", "#CCCCCC",
]

# ============================ DATA MODELS ============================

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
class Pattern:
    name: str
    color: str
    length_beats: float = 4.0
    waveform: str = "square"
    gain: float = 0.70
    notes: List[Note] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "color": self.color,
            "length_beats": self.length_beats, "waveform": self.waveform,
            "gain": self.gain, "notes": [n.to_dict() for n in self.notes],
        }

    @staticmethod
    def from_dict(data: dict) -> "Pattern":
        return Pattern(
            name=str(data.get("name", "Pattern")),
            color=str(data.get("color", "#E6E6E6")),
            length_beats=float(data.get("length_beats", 4.0)),
            waveform=str(data.get("waveform", "square")),
            gain=float(data.get("gain", 0.70)),
            notes=[Note.from_dict(n) for n in data.get("notes", [])],
        )

@dataclass
class PlaylistClip:
    pattern_index: int
    track_index: int
    start_beat: float
    duration_beats: float
    muted: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "PlaylistClip":
        return PlaylistClip(
            pattern_index=int(data.get("pattern_index", 0)),
            track_index=int(data.get("track_index", 0)),
            start_beat=float(data.get("start_beat", 0.0)),
            duration_beats=float(data.get("duration_beats", 4.0)),
            muted=bool(data.get("muted", False)),
        )

@dataclass
class PlaylistTrack:
    name: str
    color: str
    muted: bool = False
    solo: bool = False
    clips: List[PlaylistClip] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "color": self.color,
            "muted": self.muted, "solo": self.solo,
            "clips": [c.to_dict() for c in self.clips],
        }

    @staticmethod
    def from_dict(data: dict) -> "PlaylistTrack":
        return PlaylistTrack(
            name=str(data.get("name", "Track")),
            color=str(data.get("color", "#E6E6E6")),
            muted=bool(data.get("muted", False)),
            solo=bool(data.get("solo", False)),
            clips=[PlaylistClip.from_dict(c) for c in data.get("clips", [])],
        )

# ============================ SYNTH ENGINE ============================

class ChiptuneSynth:
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512) -> None:
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self._voices: Dict[Hashable, dict] = {}
        self._lock = threading.Lock()
        self.master_gain = 0.22

    @staticmethod
    def midi_to_freq(pitch: int) -> float:
        return 440.0 * (2.0 ** ((pitch - 69) / 12.0))

    def note_on(self, key: Hashable, pitch: int, velocity: float = 1.0,
                waveform: str = "square", gain: float = 0.6) -> None:
        with self._lock:
            self._voices[key] = {
                "pitch": pitch, "freq": self.midi_to_freq(pitch), "phase": 0.0,
                "velocity": float(max(0.0, min(1.0, velocity))),
                "waveform": waveform, "gain": float(max(0.0, min(1.25, gain))),
                "env": 0.0, "state": "attack",
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
        attack_step  = 1.0 / max(1, int(0.003 * sr))
        decay_step   = (1.0 - 0.65) / max(1, int(0.07 * sr))
        release_step = 1.0 / max(1, int(0.14 * sr))

        with self._lock:
            dead_keys: List[Hashable] = []
            for key, voice in list(self._voices.items()):
                phase     = float(voice["phase"])
                freq      = float(voice["freq"])
                phase_inc = 2.0 * np.pi * freq * dt
                idx       = np.arange(frames)
                phases    = (phase + phase_inc * idx) % (2.0 * np.pi)
                wave      = self._wave(str(voice["waveform"]), phases)
                env       = float(voice["env"])
                state     = str(voice["state"])
                env_values = np.zeros(frames, dtype=np.float32)
                for i in range(frames):
                    if state == "attack":
                        env += attack_step
                        if env >= 1.0: env = 1.0; state = "decay"
                    elif state == "decay":
                        env -= decay_step
                        if env <= 0.65: env = 0.65; state = "sustain"
                    elif state == "sustain":
                        env = 0.65
                    elif state == "release":
                        env -= release_step
                        if env <= 0.0: env = 0.0; state = "dead"
                    env_values[i] = env
                voice["env"]   = env
                voice["state"] = state
                voice["phase"] = float((phase + phase_inc * frames) % (2.0 * np.pi))
                out += wave * env_values * float(voice["velocity"]) * float(voice["gain"])
                if state == "dead" or env <= 0.0001:
                    dead_keys.append(key)
            for key in dead_keys:
                self._voices.pop(key, None)

        out = np.tanh(out * (self.master_gain * 2.4))
        return out.astype(np.float32)

# ============================ PLAYLIST VIEW ============================

class PlaylistView(QWidget):
    pattern_double_clicked = Signal(int)

    TRACK_HEIGHT   = 42
    TRACK_HEADER_W = 110
    BEAT_W_BASE    = 36.0
    HEADER_H       = 26   # ruler height — loop handles live here
    MIN_BEATS      = 128

    # Pixel tolerance for grabbing loop handles on the ruler
    HANDLE_HIT = 7

    def __init__(self, daw: "Nill") -> None:
        super().__init__()
        self.daw = daw
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # clip interaction state
        self.selected_clip: Optional[Tuple[int, int]] = None
        self.drag_mode:     Optional[str]             = None
        self.drag_start_x          = 0.0
        self.drag_clip_start_beat  = 0.0
        self.drag_clip_duration    = 4.0
        self.drag_clip_track       = 0
        self.hover_clip: Optional[Tuple[int, int]] = None
        self.zoom_x = 1.0

        # clip cache
        self._clip_cache: List[Tuple[int, int, PlaylistClip]] = []
        self._cache_valid = False

        # paint-drag state (left-click on empty)
        self._loop_paint_clip:   Optional[PlaylistClip] = None
        self._loop_paint_track:  Optional[int]          = None
        self._loop_paint_start_x: float                 = 0.0
        self._paint_drag_track:       Optional[int]  = None
        self._paint_drag_pat_idx:     Optional[int]  = None
        self._paint_drag_painted_beats: set           = set()

        # loop-region drag state
        self._loop_drag: Optional[str] = None   # "start" | "end" | None

        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAutoFillBackground(False)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @property
    def BEAT_W(self) -> float:
        return self.BEAT_W_BASE * self.zoom_x

    def beat_to_x(self, beat: float) -> float:
        return self.TRACK_HEADER_W + beat * self.BEAT_W

    def x_to_beat(self, x: float) -> float:
        return max(0.0, (x - self.TRACK_HEADER_W) / self.BEAT_W)

    def total_beats(self) -> float:
        max_beat = self.MIN_BEATS
        for track in self.daw.playlist_tracks:
            for clip in track.clips:
                max_beat = max(max_beat, clip.start_beat + clip.duration_beats + 16)
        return max_beat

    # ------------------------------------------------------------------
    # Clip cache
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._cache_valid = False

    def _build_cache(self) -> None:
        if self._cache_valid:
            return
        self._clip_cache = []
        for t_idx, track in enumerate(self.daw.playlist_tracks):
            for c_idx, clip in enumerate(track.clips):
                self._clip_cache.append((t_idx, c_idx, clip))
        self._clip_cache.sort(key=lambda x: x[2].start_beat)
        self._cache_valid = True

    def find_clip_at(self, x: float, y: float) -> Optional[Tuple[int, int, PlaylistClip]]:
        if y < self.HEADER_H or x < self.TRACK_HEADER_W:
            return None
        track_idx = int((y - self.HEADER_H) / self.TRACK_HEIGHT)
        if track_idx < 0 or track_idx >= len(self.daw.playlist_tracks):
            return None
        beat = self.x_to_beat(x)
        for c_idx, clip in enumerate(self.daw.playlist_tracks[track_idx].clips):
            if clip.start_beat <= beat < clip.start_beat + clip.duration_beats:
                return (track_idx, c_idx, clip)
        return None

    def clip_rect(self, track_idx: int, clip: PlaylistClip) -> QRectF:
        x = self.beat_to_x(clip.start_beat)
        y = self.HEADER_H + track_idx * self.TRACK_HEIGHT + 2
        w = max(4.0, clip.duration_beats * self.BEAT_W)
        h = self.TRACK_HEIGHT - 4
        return QRectF(x, y, w, h)

    def sizeHint(self) -> QSize:
        w = int(self.TRACK_HEADER_W + self.total_beats() * self.BEAT_W) + 100
        h = self.HEADER_H + max(8, len(self.daw.playlist_tracks)) * self.TRACK_HEIGHT + 20
        return QSize(w, h)

    # ------------------------------------------------------------------
    # Loop-handle hit testing (ruler bar only)
    # ------------------------------------------------------------------

    def _loop_handle_hit(self, x: float, y: float) -> Optional[str]:
        """Return 'start', 'end', or None depending on which loop handle x/y hits."""
        if y > self.HEADER_H:
            return None
        lx_start = self.beat_to_x(self.daw.loop_start)
        lx_end   = self.beat_to_x(self.daw.loop_end)
        if abs(x - lx_end) <= self.HANDLE_HIT:
            return "end"
        if abs(x - lx_start) <= self.HANDLE_HIT:
            return "start"
        return None

    # ------------------------------------------------------------------
    # Mini note-preview inside clip block
    # ------------------------------------------------------------------

    def _draw_note_preview(self, painter: QPainter, rect: QRectF,
                           clip: PlaylistClip) -> None:
        pattern = self.daw.patterns[clip.pattern_index]
        notes   = [n for n in pattern.notes if not n.muted]
        if not notes:
            return

        pat_len  = max(pattern.length_beats, 0.001)
        clip_dur = max(clip.duration_beats, 0.001)

        label_h = 12
        inner   = rect.adjusted(2, label_h, -2, -2)
        if inner.width() < 4 or inner.height() < 4:
            return

        pitches = [n.pitch for n in notes]
        p_min   = min(pitches)
        p_max   = max(pitches)
        p_range = max(p_max - p_min, 1)

        painter.fillRect(inner, QColor(0, 0, 0, 60))

        num_loops       = int(clip_dur / pat_len)
        note_color      = QColor(0, 0, 0, 180)
        loop_line_color = QColor(0, 0, 0, 90)

        # Loop boundary dashes
        for loop_i in range(num_loops + 1):
            lx_beat = loop_i * pat_len
            if lx_beat >= clip_dur:
                break
            lx = inner.left() + (lx_beat / clip_dur) * inner.width()
            if loop_i > 0:
                painter.setPen(QPen(loop_line_color, 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(lx), int(inner.top()), int(lx), int(inner.bottom()))

        # Note bars
        painter.setPen(Qt.PenStyle.NoPen)
        note_h = max(1.5, inner.height() / max(p_range + 1, 8))
        for loop_i in range(num_loops + 1):
            loop_offset = loop_i * pat_len
            for note in notes:
                ns = loop_offset + note.start
                ne = loop_offset + note.start + note.duration
                if ns >= clip_dur:
                    break
                ne = min(ne, clip_dur)
                nx = inner.left() + (ns / clip_dur) * inner.width()
                nw = max(1.5, (ne - ns) / clip_dur * inner.width())
                ny = inner.bottom() - ((note.pitch - p_min) / p_range) * inner.height() - note_h
                painter.setBrush(QBrush(note_color))
                painter.drawRect(QRectF(nx, ny, nw, note_h))

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#0B0B0B"))

        total_beats = self.total_beats()
        num_tracks  = len(self.daw.playlist_tracks)

        # ---- Ruler background ----
        painter.fillRect(0, 0, self.width(), self.HEADER_H, QColor("#151515"))
        painter.fillRect(0, 0, self.TRACK_HEADER_W, self.height(), QColor("#0A0A0A"))

        # ---- Loop region shading on ruler ----
        if self.daw.loop_enabled:
            lx1 = self.beat_to_x(self.daw.loop_start)
            lx2 = self.beat_to_x(self.daw.loop_end)
            loop_fill = QColor(80, 160, 255, 38)
            painter.fillRect(QRectF(lx1, 0, lx2 - lx1, self.HEADER_H), loop_fill)

        # ---- Bar grid + numbers ----
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        for beat in range(int(total_beats) + 1):
            x      = self.beat_to_x(beat)
            is_bar = beat % 4 == 0
            color  = QColor("#666") if is_bar else QColor("#333")
            painter.setPen(QPen(color, 1))
            painter.drawLine(int(x), self.HEADER_H, int(x), self.height())
            if is_bar and beat < total_beats:
                painter.setPen(QColor("#AAA"))
                painter.drawText(int(x + 4), 18, str(beat // 4 + 1))

        # ---- Loop handles on ruler ----
        # Draw a thin line + triangle handle for start and end
        for handle_beat, color_hex, side in [
            (self.daw.loop_start, "#4AF", "start"),
            (self.daw.loop_end,   "#F84", "end"),
        ]:
            hx    = int(self.beat_to_x(handle_beat))
            hcol  = QColor(color_hex)
            hcol2 = QColor(color_hex)
            hcol2.setAlpha(180)
            painter.setPen(QPen(hcol, 2))
            painter.drawLine(hx, 0, hx, self.HEADER_H)
            # Small filled triangle at top of ruler
            painter.setBrush(QBrush(hcol))
            painter.setPen(Qt.PenStyle.NoPen)
            if side == "start":
                tri = QPolygon([QPoint(hx, 4), QPoint(hx + 10, 4), QPoint(hx, self.HEADER_H - 2)])
            else:
                tri = QPolygon([QPoint(hx, 4), QPoint(hx - 10, 4), QPoint(hx, self.HEADER_H - 2)])
            painter.drawPolygon(tri)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        # ---- Track rows ----
        for t_idx in range(num_tracks):
            y     = self.HEADER_H + t_idx * self.TRACK_HEIGHT
            track = self.daw.playlist_tracks[t_idx]
            bg    = QColor("#111") if t_idx % 2 == 0 else QColor("#151515")
            if track.muted:
                bg = QColor("#080808")
            painter.fillRect(self.TRACK_HEADER_W, y,
                             int(self.beat_to_x(total_beats) - self.TRACK_HEADER_W),
                             self.TRACK_HEIGHT, bg)

            header_bg = QColor(track.color) if not track.muted else QColor("#222")
            painter.fillRect(0, y, self.TRACK_HEADER_W, self.TRACK_HEIGHT, header_bg.darker(170))
            painter.setPen(QPen(QColor(track.color if not track.muted else "#444"), 1))
            painter.drawRect(0, y, self.TRACK_HEADER_W - 1, self.TRACK_HEIGHT - 1)
            painter.setPen(QColor("#FFF" if not track.muted else "#555"))
            painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            painter.drawText(6, y + 16, track.name[:12])
            painter.setFont(QFont("Consolas", 7))
            if track.muted:
                painter.setPen(QColor("#F44"))
                painter.drawText(6, y + 30, "MUTED")
            elif track.solo:
                painter.setPen(QColor("#FC0"))
                painter.drawText(6, y + 30, "SOLO")

        # ---- Clips ----
        for t_idx, track in enumerate(self.daw.playlist_tracks):
            for c_idx, clip in enumerate(track.clips):
                if clip.muted:
                    continue
                rect    = self.clip_rect(t_idx, clip)
                if not rect.intersects(event.rect()):
                    continue
                pattern = self.daw.patterns[clip.pattern_index]
                fill    = QColor(pattern.color)
                if track.muted:
                    fill = QColor("#222")
                border = QColor("#DDD")
                if self.selected_clip == (t_idx, c_idx):
                    fill   = fill.lighter(125)
                    border = QColor("#FFF")
                painter.fillRect(rect, fill)

                # Note minimap
                self._draw_note_preview(painter, rect, clip)

                # Loop-repeat separator lines
                pat_len  = max(pattern.length_beats, 0.001)
                clip_dur = max(clip.duration_beats, 0.001)
                if clip_dur > pat_len:
                    num_loops = int(clip_dur / pat_len)
                    for li in range(1, num_loops + 1):
                        sep_beat = clip.start_beat + li * pat_len
                        if sep_beat >= clip.start_beat + clip_dur:
                            break
                        sx = self.beat_to_x(sep_beat)
                        painter.setPen(QPen(QColor(0, 0, 0, 100), 1))
                        painter.drawLine(int(sx), int(rect.top() + 2),
                                         int(sx), int(rect.bottom() - 2))

                painter.setPen(QPen(border, 1))
                painter.drawRect(rect.adjusted(0, 0, -1, -1))
                painter.setPen(QColor("#0A0A0A"))
                painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
                txt = pattern.name[:max(3, int(rect.width() / 6))]
                painter.drawText(rect.adjusted(4, 3, -4, 0),
                                 Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, txt)
                if rect.width() > 16:
                    painter.setPen(QPen(QColor("#000"), 1, Qt.PenStyle.DotLine))
                    painter.drawLine(int(rect.left() + 3), int(rect.top() + 5),
                                     int(rect.left() + 3), int(rect.bottom() - 5))
                    painter.drawLine(int(rect.right() - 4), int(rect.top() + 5),
                                     int(rect.right() - 4), int(rect.bottom() - 5))

        # ---- Playhead ----
        px = self.beat_to_x(self.daw.playhead_song_beat)
        painter.setPen(QPen(QColor("#FFF"), 2))
        painter.drawLine(int(px), self.HEADER_H, int(px), self.height())
        painter.setBrush(QBrush(QColor("#FFF")))
        tri = QPolygon([QPoint(int(px), self.HEADER_H),
                        QPoint(int(px - 5), self.HEADER_H - 7),
                        QPoint(int(px + 5), self.HEADER_H - 7)])
        painter.drawPolygon(tri)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#333"), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        x = event.position().x()
        y = event.position().y()

        # ---- Track-header area ----
        if x < self.TRACK_HEADER_W and y >= self.HEADER_H:
            track_idx = int((y - self.HEADER_H) / self.TRACK_HEIGHT)
            if 0 <= track_idx < len(self.daw.playlist_tracks):
                if event.button() == Qt.MouseButton.RightButton:
                    self.daw.show_track_context_menu(track_idx, event.globalPosition().toPoint())
                else:
                    self.daw.select_track(track_idx)
            return

        # ---- Ruler (header) area — loop handles + playhead seek ----
        if y < self.HEADER_H:
            if event.button() == Qt.MouseButton.LeftButton:
                hit = self._loop_handle_hit(x, y)
                if hit:
                    # Grab a loop handle
                    self._loop_drag = hit
                    self.drag_mode  = "loop_handle"
                else:
                    # Seek playhead
                    self.daw.playhead_song_beat = max(0.0, self.x_to_beat(x))
                    self.update()
            return

        result = self.find_clip_at(x, y)

        # ---- Right-click: delete clip OR paint + loop-extend ----
        if event.button() == Qt.MouseButton.RightButton:
            if result:
                t_idx, c_idx, _ = result
                self.daw.playlist_tracks[t_idx].clips.pop(c_idx)
                self.selected_clip = None
                self._invalidate_cache()
                self.update()
            else:
                track_idx = int((y - self.HEADER_H) / self.TRACK_HEIGHT)
                if 0 <= track_idx < len(self.daw.playlist_tracks):
                    beat    = round(self.x_to_beat(x) * 4) / 4
                    pat_idx = self.daw.selected_pattern_index
                    pat     = self.daw.patterns[pat_idx]
                    new_clip = PlaylistClip(
                        pattern_index=pat_idx, track_index=track_idx,
                        start_beat=beat, duration_beats=pat.length_beats,
                    )
                    self.daw.playlist_tracks[track_idx].clips.append(new_clip)
                    self.daw.playlist_tracks[track_idx].clips.sort(key=lambda c: c.start_beat)
                    self._invalidate_cache()
                    self._loop_paint_clip    = new_clip
                    self._loop_paint_track   = track_idx
                    self._loop_paint_start_x = x
                    self.update()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        # ---- Left-click on existing clip: move / resize ----
        if result:
            t_idx, c_idx, clip = result
            self.selected_clip = (t_idx, c_idx)
            rect = self.clip_rect(t_idx, clip)
            if abs(x - rect.left()) <= 6:
                self.drag_mode = "resize_left"
            elif abs(x - rect.right()) <= 6:
                self.drag_mode = "resize_right"
            else:
                self.drag_mode = "move"
            self.drag_start_x         = x
            self.drag_clip_start_beat = clip.start_beat
            self.drag_clip_duration   = clip.duration_beats
            self.drag_clip_track      = t_idx
        else:
            # ---- Left-click on empty space: paint-drag ----
            track_idx = int((y - self.HEADER_H) / self.TRACK_HEIGHT)
            if 0 <= track_idx < len(self.daw.playlist_tracks):
                pat_idx = self.daw.selected_pattern_index
                pat     = self.daw.patterns[pat_idx]
                pat_len = max(pat.length_beats, 0.001)
                snapped_beat = max(0.0, math.floor(self.x_to_beat(x) / pat_len) * pat_len)
                new_clip = PlaylistClip(
                    pattern_index=pat_idx, track_index=track_idx,
                    start_beat=snapped_beat, duration_beats=pat_len,
                )
                self.daw.playlist_tracks[track_idx].clips.append(new_clip)
                self.daw.playlist_tracks[track_idx].clips.sort(key=lambda c: c.start_beat)
                self._invalidate_cache()
                self._loop_paint_clip          = new_clip
                self._loop_paint_track         = track_idx
                self._loop_paint_start_x       = x
                self._paint_drag_track         = track_idx
                self._paint_drag_pat_idx       = pat_idx
                self._paint_drag_painted_beats = {snapped_beat}
                self.drag_mode                 = "paint"
                self.selected_clip             = None
                self.update()
            else:
                self.daw.playhead_song_beat = max(0.0, self.x_to_beat(x))
                self.selected_clip = None
                self.drag_mode     = None
                self.update()

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        y = event.position().y()

        # ---- Loop-handle drag ----
        if self.drag_mode == "loop_handle" and self._loop_drag:
            beat = max(0.0, self.x_to_beat(x))
            if self._loop_drag == "start":
                self.daw.loop_start = min(beat, self.daw.loop_end - 0.25)
            else:
                self.daw.loop_end = max(beat, self.daw.loop_start + 0.25)
            self.update()
            return

        # ---- Cursor hinting when idle ----
        if not self.drag_mode and self._loop_paint_clip is None:
            # Check loop handles first
            hit = self._loop_handle_hit(x, y)
            if hit:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                return
            result = self.find_clip_at(x, y)
            if result and x >= self.TRACK_HEADER_W:
                t_idx, c_idx, clip = result
                rect = self.clip_rect(t_idx, clip)
                if abs(x - rect.left()) <= 6 or abs(x - rect.right()) <= 6:
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                else:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                self.hover_clip = (t_idx, c_idx)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.hover_clip = None

        # ---- Right-click loop-extend drag ----
        if self._loop_paint_clip is not None and self.drag_mode != "paint":
            clip    = self._loop_paint_clip
            pat     = self.daw.patterns[clip.pattern_index]
            pat_len = max(pat.length_beats, 0.001)
            dragged = self.x_to_beat(x) - clip.start_beat
            loops   = max(1, math.ceil(dragged / pat_len))
            new_dur = loops * pat_len
            if new_dur != clip.duration_beats:
                clip.duration_beats = new_dur
                self._invalidate_cache()
                self.update()
            return

        # ---- Left-click paint-drag ----
        if self.drag_mode == "paint":
            self.setCursor(Qt.CursorShape.CrossCursor)
            track_idx = self._paint_drag_track
            pat_idx   = self._paint_drag_pat_idx
            painted   = self._paint_drag_painted_beats
            if track_idx is None or pat_idx is None:
                return
            if track_idx >= len(self.daw.playlist_tracks):
                return

            pat       = self.daw.patterns[pat_idx]
            pat_len   = max(pat.length_beats, 0.001)
            beat_cur  = self.x_to_beat(x)
            slot_beat = max(0.0, math.floor(beat_cur / pat_len) * pat_len)
            track     = self.daw.playlist_tracks[track_idx]

            if slot_beat not in painted:
                occupied = any(
                    c.start_beat <= slot_beat < c.start_beat + c.duration_beats
                    for c in track.clips
                    if c not in self._paint_drag_painted_clips()
                )
                if not occupied:
                    new_clip = PlaylistClip(
                        pattern_index=pat_idx, track_index=track_idx,
                        start_beat=slot_beat, duration_beats=pat_len,
                    )
                    track.clips.append(new_clip)
                    track.clips.sort(key=lambda c: c.start_beat)
                    painted.add(slot_beat)
                    self._paint_drag_painted_beats = painted
                    self._loop_paint_clip          = new_clip
                    self._invalidate_cache()
                    self.update()
            else:
                # Loop-extend the clip already at this slot
                target = next(
                    (c for c in track.clips
                     if abs(c.start_beat - slot_beat) < 0.001 and c.pattern_index == pat_idx),
                    None,
                )
                if target is not None:
                    dragged = beat_cur - target.start_beat
                    loops   = max(1, math.ceil(dragged / pat_len))
                    new_dur = loops * pat_len
                    if new_dur != target.duration_beats:
                        target.duration_beats  = new_dur
                        self._loop_paint_clip  = target
                        self._invalidate_cache()
                        self.update()
            return

        # ---- Normal move / resize drag ----
        if self.drag_mode and self.drag_mode not in ("paint", "loop_handle") and self.selected_clip:
            t_idx, c_idx = self.selected_clip
            clip         = self.daw.playlist_tracks[t_idx].clips[c_idx]
            delta_beats  = (x - self.drag_start_x) / self.BEAT_W
            snap         = self.daw.current_snap

            if self.drag_mode == "move":
                new_start = round((self.drag_clip_start_beat + delta_beats) / snap) * snap
                clip.start_beat = max(0.0, new_start)
                new_track = int((event.position().y() - self.HEADER_H) / self.TRACK_HEIGHT)
                if 0 <= new_track < len(self.daw.playlist_tracks) and new_track != t_idx:
                    clip.track_index = new_track
                    self.daw.playlist_tracks[t_idx].clips.pop(c_idx)
                    self.daw.playlist_tracks[new_track].clips.append(clip)
                    self.daw.playlist_tracks[new_track].clips.sort(key=lambda c: c.start_beat)
                    self.selected_clip    = (new_track, self.daw.playlist_tracks[new_track].clips.index(clip))
                    self.drag_clip_track  = new_track
            elif self.drag_mode == "resize_right":
                new_dur = max(snap, round((self.drag_clip_duration + delta_beats) / snap) * snap)
                clip.duration_beats = new_dur
            elif self.drag_mode == "resize_left":
                new_start = round((self.drag_clip_start_beat + delta_beats) / snap) * snap
                end       = self.drag_clip_start_beat + self.drag_clip_duration
                if new_start < end - snap:
                    clip.start_beat     = max(0.0, new_start)
                    clip.duration_beats = end - clip.start_beat

            self._invalidate_cache()
            self.update()

    def _paint_drag_painted_clips(self) -> list:
        painted   = getattr(self, "_paint_drag_painted_beats", set())
        track_idx = getattr(self, "_paint_drag_track", None)
        pat_idx   = getattr(self, "_paint_drag_pat_idx", None)
        if track_idx is None or track_idx >= len(self.daw.playlist_tracks):
            return []
        track = self.daw.playlist_tracks[track_idx]
        return [c for c in track.clips if c.start_beat in painted and c.pattern_index == pat_idx]

    def mouseReleaseEvent(self, event) -> None:
        self.drag_mode                 = None
        self._loop_drag                = None
        self._loop_paint_clip          = None
        self._loop_paint_track         = None
        self._paint_drag_track         = None
        self._paint_drag_pat_idx       = None
        self._paint_drag_painted_beats = set()
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event) -> None:
        x = event.position().x()
        y = event.position().y()
        if x < self.TRACK_HEADER_W or y < self.HEADER_H:
            return
        result = self.find_clip_at(x, y)
        if result:
            _, _, clip = result
            self.pattern_double_clicked.emit(clip.pattern_index)

# ============================ PIANO ROLL ============================

class PianoRoll(QWidget):
    note_changed = Signal()

    KEYBOARD_W = 92
    HEADER_H   = 30
    MIN_MIDI   = 36
    MAX_MIDI   = 96

    def __init__(self, daw: "Nill") -> None:
        super().__init__()
        self.daw = daw
        self.setMinimumSize(600, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAutoFillBackground(False)
        self.selected_note:       Optional[Note] = None
        self.drag_mode:           Optional[str]  = None
        self.drag_note_start_beat  = 0.0
        self.drag_note_start_pitch = 60
        self.drag_origin_beat      = 0.0
        self.drag_origin_pitch     = 60
        self.preview_key:         Optional[str]  = None
        self.zoom_x = 1.0
        self.zoom_y = 1.0

    @property
    def total_rows(self) -> int:
        return self.MAX_MIDI - self.MIN_MIDI + 1

    def is_black(self, pitch: int) -> bool:
        return pitch % 12 in {1, 3, 6, 8, 10}

    def note_name(self, pitch: int) -> str:
        octave = (pitch // 12) - 1
        return f"{NOTE_NAMES[pitch % 12]}{octave}"

    @property
    def BEAT_W(self) -> float:
        pat = self.daw.current_pattern()
        if pat.length_beats <= 0:
            return 40.0
        vw     = self.width()
        scroll = getattr(self.daw, "piano_scroll", None)
        if scroll: vw = scroll.viewport().width()
        available = max(100.0, float(vw - self.KEYBOARD_W))
        fit       = max(12.0, available / float(pat.length_beats))
        return fit * self.zoom_x

    @property
    def ROW_H(self) -> float:
        vh     = self.height()
        scroll = getattr(self.daw, "piano_scroll", None)
        if scroll: vh = scroll.viewport().height()
        available = max(200.0, float(vh - self.HEADER_H))
        fit       = max(10.0, available / float(self.total_rows))
        return fit * self.zoom_y

    def beat_to_x(self, beat: float) -> float:
        return self.KEYBOARD_W + beat * self.BEAT_W

    def x_to_beat(self, x: float) -> float:
        return max(0.0, (x - self.KEYBOARD_W) / self.BEAT_W)

    def pitch_to_y(self, pitch: int) -> float:
        return self.HEADER_H + (self.MAX_MIDI - pitch) * self.ROW_H

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

    def snap_value(self, value: float) -> float:
        snap = self.daw.current_snap
        if snap <= 0.0: return value
        return round(value / snap) * snap

    def find_note_at(self, x: float, y: float) -> Optional[Note]:
        for note in reversed(self.daw.current_pattern().notes):
            if self.note_rect(note).contains(x, y):
                return note
        return None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#0B0B0B"))

        pat        = self.daw.current_pattern()
        grid_right = self.beat_to_x(pat.length_beats)

        painter.fillRect(0, 0, self.width(), self.HEADER_H, QColor("#151515"))
        painter.fillRect(0, 0, self.KEYBOARD_W, self.height(), QColor("#050505"))

        for pitch in range(self.MAX_MIDI, self.MIN_MIDI - 1, -1):
            y        = self.pitch_to_y(pitch)
            row_rect = QRectF(self.KEYBOARD_W, y, max(0, grid_right - self.KEYBOARD_W), self.ROW_H)
            key_rect = QRectF(0, y, self.KEYBOARD_W, self.ROW_H)
            if self.is_black(pitch):
                key_color, row_color = QColor("#202020"), QColor("#141414")
            else:
                key_color, row_color = QColor("#E6E6E6"), QColor("#181818")
            painter.fillRect(row_rect, row_color)
            painter.fillRect(key_rect, key_color)
            painter.setPen(QPen(QColor("#2A2A2A"), 1))
            painter.drawLine(int(self.KEYBOARD_W), int(y), int(grid_right), int(y))
            painter.setPen(QPen(QColor("#0F0F0F"), 1))
            painter.drawRect(key_rect)
            label_color = QColor("#111") if not self.is_black(pitch) else QColor("#E4E4E4")
            painter.setPen(label_color)
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(key_rect.adjusted(6, 0, -4, 0),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             self.note_name(pitch))

        snap  = self.daw.current_snap
        if snap <= 0.0: snap = 0.25
        steps = int(pat.length_beats / snap) + 1
        for i in range(steps + 1):
            beat  = i * snap
            x     = self.beat_to_x(beat)
            color = (QColor("#555") if abs(beat % 4.0) < 0.001
                     else QColor("#3A3A3A") if i % int(max(1, round(1.0 / snap))) == 0
                     else QColor("#262626"))
            painter.setPen(QPen(color, 1))
            painter.drawLine(int(x), self.HEADER_H, int(x), self.height())

        painter.setPen(QPen(QColor("#C8C8C8"), 1))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        for beat in range(int(pat.length_beats) + 1):
            x = self.beat_to_x(beat)
            painter.drawLine(int(x), 0, int(x), self.HEADER_H)
            if beat < pat.length_beats:
                painter.drawText(int(x + 4), 20, str(beat + 1))

        painter.setPen(QColor("#777"))
        painter.drawText(10, 20, "PIANO")
        painter.drawText(self.KEYBOARD_W + 8, 20, f"Pattern: {pat.name} [{pat.waveform}]")

        for note in pat.notes:
            rect   = self.note_rect(note)
            fill   = QColor(pat.color)
            if note.muted: fill = QColor("#5F5F5F")
            border = QColor("#F4F4F4")
            if self.selected_note is note:
                fill   = fill.lighter(135)
                border = QColor("#FFF")
            painter.fillRect(rect, fill)
            painter.setPen(QPen(border, 1))
            painter.drawRect(rect)
            painter.setPen(QPen(QColor("#0A0A0A"), 1))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(rect.adjusted(4, 0, -4, 0),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             self.note_name(note.pitch))

        if self.daw.playing:
            local = self.daw.playhead_song_beat % pat.length_beats
            px    = self.beat_to_x(local)
            painter.setPen(QPen(QColor("#FFF"), 2))
            painter.drawLine(int(px), self.HEADER_H, int(px), self.height())

        painter.setPen(QPen(QColor("#4A4A4A"), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        pos  = event.position()
        x, y = pos.x(), pos.y()
        if y < self.HEADER_H:
            return
        if x < self.KEYBOARD_W and event.button() == Qt.MouseButton.LeftButton:
            pitch            = self.y_to_pitch(y)
            self.preview_key = f"preview-{time.time_ns()}"
            self.daw.preview_note_on(self.preview_key, pitch)
            return
        clicked            = self.find_note_at(x, y)
        self.selected_note = clicked
        if event.button() == Qt.MouseButton.RightButton:
            if clicked:
                self.daw.current_pattern().notes.remove(clicked)
                self.selected_note = None
                self.note_changed.emit()
                self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        beat  = self.snap_value(self.x_to_beat(x))
        beat  = max(0.0, min(self.daw.current_pattern().length_beats - self.daw.current_snap, beat))
        pitch = self.y_to_pitch(y)
        if clicked:
            rect            = self.note_rect(clicked)
            self.drag_mode  = "resize" if (rect.right() - x) <= 8 else "move"
            self.drag_origin_beat      = beat
            self.drag_origin_pitch     = pitch
            self.drag_note_start_beat  = clicked.start
            self.drag_note_start_pitch = clicked.pitch
        else:
            new_note = Note(pitch=pitch, start=beat, duration=self.daw.default_note_length, velocity=100)
            self.daw.current_pattern().notes.append(new_note)
            self.selected_note = new_note
            self.daw.current_pattern().notes.sort(key=lambda n: (n.start, n.pitch))
            self.drag_mode = None
            self.note_changed.emit()
            self.update()

    def mouseMoveEvent(self, event) -> None:
        pos  = event.position()
        x, y = pos.x(), pos.y()
        if self.drag_mode and self.selected_note:
            beat  = self.snap_value(self.x_to_beat(x))
            beat  = max(0.0, min(self.daw.current_pattern().length_beats, beat))
            pitch = self.y_to_pitch(y)
            if self.drag_mode == "move":
                delta_b = beat - self.drag_origin_beat
                delta_p = pitch - self.drag_origin_pitch
                new_s   = max(0.0, self.drag_note_start_beat + delta_b)
                new_s   = min(self.daw.current_pattern().length_beats - self.selected_note.duration, new_s)
                self.selected_note.start = self.snap_value(new_s)
                self.selected_note.pitch = max(self.MIN_MIDI, min(self.MAX_MIDI, self.drag_note_start_pitch + delta_p))
            elif self.drag_mode == "resize":
                new_d = max(self.daw.current_snap, beat - self.selected_note.start)
                max_d = self.daw.current_pattern().length_beats - self.selected_note.start
                self.selected_note.duration = min(max_d, self.snap_value(new_d))
            self.note_changed.emit()
            self.update()
        else:
            note_under = self.find_note_at(x, y) if x >= self.KEYBOARD_W else None
            if note_under:
                rect = self.note_rect(note_under)
                self.setCursor(Qt.CursorShape.SizeHorCursor if (rect.right() - x) <= 8
                               else Qt.CursorShape.OpenHandCursor)
            elif x < self.KEYBOARD_W:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseReleaseEvent(self, event) -> None:
        self.drag_mode = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if self.preview_key:
            self.daw.preview_note_off(self.preview_key)
            self.preview_key = None

    def leaveEvent(self, event) -> None:
        if self.preview_key:
            self.daw.preview_note_off(self.preview_key)
            self.preview_key = None

    def delete_selected(self) -> None:
        if self.selected_note and self.selected_note in self.daw.current_pattern().notes:
            self.daw.current_pattern().notes.remove(self.selected_note)
            self.selected_note = None
            self.note_changed.emit()
            self.update()

# ============================ MAIN WINDOW ============================

class Nill(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NILL | Terminal Interface DAW")
        self.resize(1600, 950)
        self.setMinimumSize(1200, 700)

        self.bpm                = 140
        self.current_snap       = 0.25
        self.default_note_length = 1.0
        self.playing            = False
        self.loop_enabled       = False   # song-level loop
        self.loop_start         = 0.0    # beats
        self.loop_end           = 16.0   # beats (default 4 bars)
        self.playhead_song_beat = 0.0
        self._last_playback_time = time.perf_counter()
        self._last_active_keys: set = set()
        self._preview_keys:     set = set()
        self.osc_process        = None
        self.visualizer_process = None

        self.synth  = ChiptuneSynth()
        self.stream = None
        self._start_audio()

        self.patterns: List[Pattern] = []
        self.selected_pattern_index  = 0
        self._build_default_patterns()

        self.playlist_tracks: List[PlaylistTrack] = []
        self._build_default_tracks()

        self._build_ui()
        self._bind_shortcuts()
        self.refresh_pattern_buttons()

        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(16)
        self.playback_timer.timeout.connect(self.update_playback)
        self.playback_timer.start()

    # ------------------------------------------------------------------
    # Default content
    # ------------------------------------------------------------------

    def _build_default_patterns(self) -> None:
        defaults = [
            ("Kick",  PATTERN_COLORS[0], "square",   36, 4.0),
            ("Snare", PATTERN_COLORS[1], "noise",    50, 4.0),
            ("Hat",   PATTERN_COLORS[2], "triangle", 62, 4.0),
            ("Bass",  PATTERN_COLORS[3], "saw",      48, 8.0),
            ("Lead",  PATTERN_COLORS[4], "square",   60, 8.0),
            ("Chord", PATTERN_COLORS[5], "saw",      60, 8.0),
            ("ARP",   PATTERN_COLORS[6], "triangle", 72, 4.0),
            ("FX",    PATTERN_COLORS[7], "noise",    80, 4.0),
        ]
        for name, color, wave, base_pitch, length in defaults:
            pat = Pattern(name=name, color=color, waveform=wave, length_beats=length)
            if "Kick" in name:
                for b in range(0, 16, 4):
                    pat.notes.append(Note(pitch=36, start=b / 4.0, duration=0.5))
            elif "Snare" in name:
                for b in [2, 6, 10, 14]:
                    pat.notes.append(Note(pitch=50, start=b / 4.0, duration=0.3))
            elif "Hat" in name:
                for b in range(16):
                    pat.notes.append(Note(pitch=62, start=b / 4.0, duration=0.15))
            elif "Bass" in name:
                for s, p in [(0, 48), (2, 36), (4, 43), (6, 41)]:
                    pat.notes.append(Note(pitch=p, start=s, duration=2))
            elif "Lead" in name:
                for s, p in [(0, 60), (1, 62), (2, 64), (3, 67)]:
                    pat.notes.append(Note(pitch=p, start=s, duration=1))
            self.patterns.append(pat)

    def _build_default_tracks(self) -> None:
        for i in range(8):
            color = PATTERN_COLORS[i % len(PATTERN_COLORS)]
            self.playlist_tracks.append(PlaylistTrack(name=f"Track {i+1}", color=color))
        for t_idx in range(min(4, len(self.patterns))):
            for bar in range(4):
                clip = PlaylistClip(
                    pattern_index=t_idx, track_index=t_idx,
                    start_beat=bar * 4.0,
                    duration_beats=self.patterns[t_idx].length_beats,
                )
                self.playlist_tracks[t_idx].clips.append(clip)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _start_audio(self) -> None:
        if sd is None:
            print("sounddevice not available.")
            return
        try:
            self.stream = sd.OutputStream(
                samplerate=self.synth.sample_rate, channels=1,
                blocksize=self.synth.buffer_size, callback=self.audio_callback,
            )
            self.stream.start()
            print("Audio engine started.")
        except Exception as exc:
            self.stream = None
            print("Audio startup failed:", exc)

    def audio_callback(self, outdata, frames, time_info, status) -> None:
        if status: print(status)
        audio = self.synth.generate_audio(frames)
        outdata[:, 0] = np.clip(audio, -0.97, 0.97)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root  = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ---- Top bar ----
        top_bar    = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        top_layout.addWidget(QLabel("PATTERN:"))
        self._pattern_buttons: List[QPushButton] = []
        for i in range(len(self.patterns)):
            btn = QPushButton(f"P{i+1}")
            btn.setCheckable(True)
            btn.setFixedWidth(36)
            btn.clicked.connect(lambda checked=False, idx=i: self.set_selected_pattern(idx))
            self._pattern_buttons.append(btn)
            top_layout.addWidget(btn)

        # Play button
        self.play_btn = QPushButton("○")
        self.play_btn.setCheckable(True)
        self.play_btn.setObjectName("transportCircle")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.clicked.connect(self.toggle_playback)
        top_layout.addWidget(self.play_btn)
        top_layout.addWidget(QLabel("SPACE"))

        # Loop toggle button
        self.loop_btn = QPushButton("⟲")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setFixedSize(36, 36)
        self.loop_btn.setToolTip(
            "Toggle song loop\n"
            "Drag the blue (start) and orange (end) handles on the ruler to set loop range"
        )
        self.loop_btn.clicked.connect(self.toggle_loop)
        top_layout.addWidget(self.loop_btn)

        # BPM
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(40, 260)
        self.bpm_spin.setValue(self.bpm)
        self.bpm_spin.valueChanged.connect(self.on_bpm_changed)
        top_layout.addWidget(QLabel("BPM:"))
        top_layout.addWidget(self.bpm_spin)

        # Snap
        self.snap_combo = QComboBox()
        for label_txt, val in [("1/4", 0.25), ("1/8", 0.125), ("1/16", 0.0625), ("1/2", 0.5), ("1", 1.0)]:
            self.snap_combo.addItem(label_txt, val)
        self.snap_combo.setCurrentIndex(0)
        self.snap_combo.currentIndexChanged.connect(self.on_snap_changed)
        top_layout.addWidget(QLabel("Snap:"))
        top_layout.addWidget(self.snap_combo)

        # Zoom
        top_layout.addWidget(QLabel("Zoom:"))
        zout   = QPushButton("-")
        zin    = QPushButton("+")
        zreset = QPushButton("Reset")
        zout.setFixedWidth(28); zin.setFixedWidth(28); zreset.setFixedWidth(50)
        zout.clicked.connect(  lambda: self.set_playlist_zoom(self.playlist_view.zoom_x * 0.8))
        zin.clicked.connect(   lambda: self.set_playlist_zoom(self.playlist_view.zoom_x * 1.25))
        zreset.clicked.connect(lambda: self.set_playlist_zoom(1.0))
        top_layout.addWidget(zout)
        top_layout.addWidget(zreset)
        top_layout.addWidget(zin)

        # Command line
        top_layout.addWidget(QLabel(">"))
        self.command_line = QLineEdit()
        self.command_line.setPlaceholderText("set bpm ___ | show osc | show visualizer")
        self.command_line.setMinimumWidth(260)
        self.command_line.returnPressed.connect(self.run_command_line)
        top_layout.addWidget(self.command_line, 1)

        # Add track
        top_layout.addWidget(QLabel("Tracks:"))
        add_t_btn = QPushButton("+")
        add_t_btn.setFixedWidth(28)
        add_t_btn.clicked.connect(self.add_track)
        top_layout.addWidget(add_t_btn)

        outer.addWidget(top_bar)

        # ---- Main splitter: playlist / piano roll ----
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        playlist_container = QWidget()
        pc_layout = QVBoxLayout(playlist_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(
            "PLAYLIST  "
            "[ left-click empty = paint  |  drag right = loop-extend  |  right-click clip = delete  |"
            "  drag clip = move  |  drag edge = resize  |  double-click = edit  |"
            "  drag ruler handles = loop region ]"
        )
        lbl.setStyleSheet("color:#888; font-size:10px; padding:2px;")
        pc_layout.addWidget(lbl)
        self.playlist_scroll = QScrollArea()
        self.playlist_scroll.setWidgetResizable(True)
        self.playlist_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.playlist_view = PlaylistView(self)
        self.playlist_view.pattern_double_clicked.connect(self.on_pattern_double_clicked)
        self.playlist_scroll.setWidget(self.playlist_view)
        self.playlist_scroll.viewport().setAutoFillBackground(False)
        self.playlist_scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.playlist_scroll.viewport().setStyleSheet("background: transparent;")
        pc_layout.addWidget(self.playlist_scroll)
        main_splitter.addWidget(playlist_container)

        piano_container = QWidget()
        pr_layout = QVBoxLayout(piano_container)
        pr_layout.setContentsMargins(0, 0, 0, 0)
        pr_layout.addWidget(QLabel("PIANO ROLL (editing current pattern)"))
        self.piano_scroll = QScrollArea()
        self.piano_scroll.setWidgetResizable(True)
        self.piano_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.piano_roll = PianoRoll(self)
        self.piano_scroll.setWidget(self.piano_roll)
        self.piano_scroll.viewport().setAutoFillBackground(False)
        self.piano_scroll.viewport().setStyleSheet("background: transparent;")
        pr_layout.addWidget(self.piano_scroll)
        main_splitter.addWidget(piano_container)

        main_splitter.setSizes([520, 340])
        outer.addWidget(main_splitter, 1)

        footer = QLabel(
            "NILL DAW  |  ⟲ = song loop  |  blue handle = loop start  |  orange handle = loop end  "
            "|  cmds: set bpm ___  show osc  show visualizer"
        )
        footer.setStyleSheet("color:#9A9A9A; padding:3px;")
        outer.addWidget(footer)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0A0A0A; color: #E6E6E6;
                font-family: Consolas, 'Courier New', monospace; font-size: 11px;
            }
            QPushButton { background: #1A1A1A; border: 1px solid #444; padding: 5px; border-radius: 4px; }
            QPushButton:hover { background: #242424; }
            QPushButton:checked { background: #D9D9D9; color: #080808; border: 1px solid #FFF; }
            QPushButton#transportCircle {
                border-radius: 20px; font-size: 22px; font-weight: bold; padding: 0;
            }
            QPushButton#transportCircle:checked { background: #E6E6E6; color: #080808; border: 2px solid #FFF; }
            QScrollArea { border: 1px solid #4A4A4A; background: #050505; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:horizontal, QScrollBar:vertical { background: #0A0A0A; border: 1px solid #2A2A2A; }
            QScrollBar::handle:horizontal, QScrollBar::handle:vertical {
                background: #777; border: 1px solid #BDBDBD;
            }
            QLineEdit, QComboBox, QSpinBox { background: #141414; border: 1px solid #3F3F3F; padding: 4px; border-radius: 3px; }
            QLabel { color: #E6E6E6; }
            QSplitter::handle { background: #333; }
        """)

        for btn in self.findChildren(QPushButton):
            if btn.objectName() != "transportCircle":
                btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        QApplication.instance().installEventFilter(self)
        QTimer.singleShot(0, self.apply_piano_zoom)
        self.set_selected_pattern(0)

    def _bind_shortcuts(self) -> None:
        for shortcut, slot in [
            ("Ctrl++", lambda: self.set_playlist_zoom(self.playlist_view.zoom_x * 1.25)),
            ("Ctrl+-", lambda: self.set_playlist_zoom(self.playlist_view.zoom_x * 0.8)),
            ("Ctrl+S", self.save_project),
            ("Ctrl+O", self.load_project),
            ("Delete", self.piano_roll.delete_selected),
        ]:
            action = QAction(self)
            action.setShortcut(shortcut)
            action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            action.triggered.connect(slot)
            self.addAction(action)

    # ------------------------------------------------------------------
    # Pattern / zoom helpers
    # ------------------------------------------------------------------

    def current_pattern(self) -> Pattern:
        return self.patterns[self.selected_pattern_index]

    def set_selected_pattern(self, idx: int) -> None:
        self.selected_pattern_index = idx
        for i, btn in enumerate(self._pattern_buttons):
            btn.blockSignals(True)
            btn.setChecked(i == idx)
            btn.blockSignals(False)
        self.piano_roll.update()

    def set_playlist_zoom(self, zoom: float) -> None:
        self.playlist_view.zoom_x = max(0.25, min(4.0, zoom))
        self.playlist_view.updateGeometry()
        self.playlist_view.update()

    def apply_piano_zoom(self) -> None:
        pat = self.current_pattern()
        w   = int(PianoRoll.KEYBOARD_W + pat.length_beats * 50 * self.piano_roll.zoom_x) + 50
        h   = int(PianoRoll.HEADER_H + (PianoRoll.MAX_MIDI - PianoRoll.MIN_MIDI + 1) * 14 * self.piano_roll.zoom_y) + 20
        self.piano_roll.setMinimumWidth(max(600, w))
        self.piano_roll.setMinimumHeight(max(300, h))
        self.piano_roll.updateGeometry()
        self.piano_roll.update()

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def toggle_playback(self) -> None:
        if self.playing:
            self.stop_playback()
            return
        self.playing = True
        self._last_playback_time = time.perf_counter()
        self.update_transport()
        self.playlist_view.update()

    def toggle_loop(self) -> None:
        self.loop_enabled = not self.loop_enabled
        self.loop_btn.setChecked(self.loop_enabled)
        self.playlist_view.update()   # redraw ruler to show/hide shading

    def stop_playback(self) -> None:
        self.playing            = False
        self.playhead_song_beat = 0.0
        self._last_playback_time = time.perf_counter()
        self._release_all()
        self.update_transport()
        self.playlist_view.update()

    def _release_all(self) -> None:
        for key in list(self._last_active_keys):
            self.synth.note_off(key)
        self._last_active_keys.clear()

    def _song_end_beat(self) -> float:
        end = 0.0
        for track in self.playlist_tracks:
            for clip in track.clips:
                end = max(end, clip.start_beat + clip.duration_beats)
        return end

    def update_playback(self) -> None:
        now = time.perf_counter()
        if not self.playing:
            self._last_playback_time = now
            return
        delta = now - self._last_playback_time
        self._last_playback_time = now
        self.playhead_song_beat += delta * (self.bpm / 60.0)

        # ---- Song loop ----
        if self.loop_enabled:
            ls = self.loop_start
            le = self.loop_end
            if le > ls and self.playhead_song_beat >= le:
                self.playhead_song_beat = ls + (self.playhead_song_beat - le) % (le - ls)
                self._release_all()   # clean note-off at loop point

        active      = self._collect_active_notes()
        active_keys = set(active.keys())

        for key in self._last_active_keys - active_keys:
            self.synth.note_off(key)
        for key in active_keys - self._last_active_keys:
            pat, note = active[key]
            self.synth.note_on(
                key=key, pitch=note.pitch,
                velocity=max(0.0, min(1.0, note.velocity / 127.0)),
                waveform=pat.waveform, gain=pat.gain,
            )

        self._last_active_keys = active_keys
        self.playlist_view.update()
        self.piano_roll.update()

    def _collect_active_notes(self) -> Dict[str, Tuple[Pattern, Note]]:
        active:   Dict[str, Tuple[Pattern, Note]] = {}
        beat      = self.playhead_song_beat
        any_solo  = any(t.solo for t in self.playlist_tracks)

        for t_idx, track in enumerate(self.playlist_tracks):
            if track.muted: continue
            if any_solo and not track.solo: continue
            for c_idx, clip in enumerate(track.clips):
                if clip.muted: continue
                if not (clip.start_beat <= beat < clip.start_beat + clip.duration_beats):
                    continue
                pat        = self.patterns[clip.pattern_index]
                local_beat = (beat - clip.start_beat) % max(pat.length_beats, 0.001)
                for n_idx, note in enumerate(pat.notes):
                    if note.muted: continue
                    if note.start <= local_beat < note.start + note.duration:
                        key = f"{t_idx}:{c_idx}:{n_idx}"
                        active[key] = (pat, note)
        return active

    def update_transport(self) -> None:
        if hasattr(self, "play_btn"):
            self.play_btn.blockSignals(True)
            self.play_btn.setChecked(self.playing)
            self.play_btn.setText("●" if self.playing else "○")
            self.play_btn.blockSignals(False)

    # ------------------------------------------------------------------
    # Track management
    # ------------------------------------------------------------------

    def add_track(self) -> None:
        idx   = len(self.playlist_tracks)
        color = PATTERN_COLORS[idx % len(PATTERN_COLORS)]
        self.playlist_tracks.append(PlaylistTrack(name=f"Track {idx+1}", color=color))
        self.playlist_view.updateGeometry()
        self.playlist_view.update()

    def select_track(self, idx: int) -> None:
        pass

    def show_track_context_menu(self, track_idx: int, global_pos) -> None:
        menu  = QMenu(self)
        track = self.playlist_tracks[track_idx]
        menu.addAction("Mute" if not track.muted else "Unmute").triggered.connect(
            lambda: self.toggle_track_mute(track_idx))
        menu.addAction("Solo" if not track.solo else "Unsolo").triggered.connect(
            lambda: self.toggle_track_solo(track_idx))
        menu.addSeparator()
        menu.addAction("Delete Track").triggered.connect(lambda: self.delete_track(track_idx))
        menu.exec(global_pos)

    def toggle_track_mute(self, idx: int) -> None:
        self.playlist_tracks[idx].muted = not self.playlist_tracks[idx].muted
        self.playlist_view.update()

    def toggle_track_solo(self, idx: int) -> None:
        self.playlist_tracks[idx].solo = not self.playlist_tracks[idx].solo
        self.playlist_view.update()

    def delete_track(self, idx: int) -> None:
        if len(self.playlist_tracks) <= 1: return
        self.playlist_tracks.pop(idx)
        for t_i, track in enumerate(self.playlist_tracks):
            for clip in track.clips:
                clip.track_index = t_i
        self.playlist_view.updateGeometry()
        self.playlist_view.update()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on_pattern_double_clicked(self, pattern_index: int) -> None:
        self.set_selected_pattern(pattern_index)

    def on_bpm_changed(self, value: int) -> None:
        self.bpm = value

    def on_snap_changed(self, index: int) -> None:
        self.current_snap = float(self.snap_combo.currentData())

    def preview_note_on(self, key: str, pitch: int) -> None:
        pat = self.current_pattern()
        self._preview_keys.add(key)
        self.synth.note_on(key, pitch, velocity=0.9, waveform=pat.waveform, gain=min(1.0, pat.gain))

    def preview_note_off(self, key: str) -> None:
        if key in self._preview_keys:
            self.synth.note_off(key)
            self._preview_keys.discard(key)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def project_data(self) -> dict:
        return {
            "app": "Nill TERMINAL", "version": 6,
            "bpm": self.bpm, "snap": self.current_snap,
            "loop_enabled": self.loop_enabled,
            "loop_start":   self.loop_start,
            "loop_end":     self.loop_end,
            "patterns": [p.to_dict() for p in self.patterns],
            "tracks":   [t.to_dict() for t in self.playlist_tracks],
        }

    def save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Nill Project",
            str(Path.home() / "nill_project.json"), "Nill Project (*.json)",
        )
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.project_data(), fh, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Nill Project", str(Path.home()), "Nill Project (*.json)",
        )
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            return
        self.stop_playback()
        self.bpm          = int(data.get("bpm", 140))
        self.loop_enabled = bool(data.get("loop_enabled", False))
        self.loop_start   = float(data.get("loop_start", 0.0))
        self.loop_end     = float(data.get("loop_end", 16.0))
        self.patterns        = [Pattern.from_dict(p) for p in data.get("patterns", [])]
        self.playlist_tracks = [PlaylistTrack.from_dict(t) for t in data.get("tracks", [])]
        self.selected_pattern_index = 0
        self.bpm_spin.setValue(self.bpm)
        self.loop_btn.setChecked(self.loop_enabled)
        self.refresh_pattern_buttons()
        self.playlist_view.updateGeometry()
        self.playlist_view.update()
        self.piano_roll.update()

    def refresh_pattern_buttons(self) -> None:
        for i, btn in enumerate(self._pattern_buttons):
            btn.blockSignals(True)
            btn.setChecked(i == self.selected_pattern_index)
            btn.blockSignals(False)

    # ------------------------------------------------------------------
    # Command line
    # ------------------------------------------------------------------

    def run_command_line(self) -> None:
        raw     = self.command_line.text().strip()
        command = " ".join(raw.lower().split())
        if not command:
            return
        if command.startswith("set bpm "):
            try:
                bpm = int(command.removeprefix("set bpm ").strip())
            except ValueError:
                QMessageBox.warning(self, "Command", "Use: set bpm ___"); return
            if not (40 <= bpm <= 260):
                QMessageBox.warning(self, "Command", "BPM must be 40–260."); return
            self.bpm_spin.setValue(bpm)
            self.command_line.clear(); return
        if command == "show osc":
            self.command_line.clear(); self.show_osc(); return
        if command == "show visualizer":
            self.command_line.clear(); self.show_visualizer(); return
        QMessageBox.warning(self, "Command",
                            "Allowed commands:\nset bpm ___\nshow osc\nshow visualizer")

    def show_osc(self) -> None:
        if self.osc_process is not None and self.osc_process.poll() is None:
            QMessageBox.information(self, "OSC", "OSC window already running."); return
        try:
            self.osc_process = subprocess.Popen([sys.executable, str(Path(__file__)), "--nill-osc"])
        except Exception as exc:
            QMessageBox.critical(self, "OSC Failed", str(exc))

    def show_visualizer(self) -> None:
        if self.visualizer_process is not None and self.visualizer_process.poll() is None:
            QMessageBox.information(self, "Visualizer", "Visualizer already running."); return
        try:
            self.visualizer_process = subprocess.Popen(
                [sys.executable, str(Path(__file__)), "--nill-visualizer"])
        except Exception as exc:
            QMessageBox.critical(self, "Visualizer Failed", str(exc))

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            if hasattr(self, "command_line") and self.command_line.hasFocus():
                return False
            self.toggle_playback()
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Delete:
            self.piano_roll.delete_selected(); event.accept(); return
        note = self.piano_roll.selected_note
        if note:
            if event.key() == Qt.Key.Key_Up:
                note.pitch = min(PianoRoll.MAX_MIDI, note.pitch + 1)
                self.piano_roll.update(); return
            if event.key() == Qt.Key.Key_Down:
                note.pitch = max(PianoRoll.MIN_MIDI, note.pitch - 1)
                self.piano_roll.update(); return
            if event.key() == Qt.Key.Key_Left:
                note.start = max(0.0, note.start - self.current_snap)
                self.piano_roll.update(); return
            if event.key() == Qt.Key.Key_Right:
                note.start = min(self.current_pattern().length_beats - note.duration,
                                 note.start + self.current_snap)
                self.piano_roll.update(); return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.stop_playback()
        self.synth.all_notes_off()
        for proc in [self.visualizer_process, self.osc_process]:
            if proc is not None and proc.poll() is None:
                try: proc.terminate()
                except Exception: pass
        if self.stream:
            try: self.stream.stop(); self.stream.close()
            except Exception: pass
        super().closeEvent(event)

# ============================ EMBEDDED OSC ============================

EMBEDDED_OSC_CODE = r'''
import tkinter as tk
from tkinter import Canvas
import math
import numpy as np
from typing import Dict, List, Optional
import threading
import time
import sounddevice as sd

BG_BLACK = '#000000'
BG_DARK = '#111111'
BG_PANEL = '#0a0a0a'
FG_WHITE = '#FFFFFF'
FG_DIM = '#666666'
FG_BRIGHT = '#CCCCCC'
BORDER = '#333333'
KNOB_BODY = '#505050'
KNOB_SURFACE = '#1a1a1a'
FONT_MAIN = 'Consolas'

SR = 44100
BUFFER_SIZE = 2048
AMP = 0.08

WAVEFORMS = ['sine', 'triangle', 'saw', 'square']
WHITE_KEYS = {'a': 60, 's': 62, 'd': 64, 'f': 65, 'g': 67, 'h': 69, 'j': 71, 'k': 72, 'l': 74, ';': 76, "'": 77}
BLACK_KEYS = {'w': 61, 'e': 63, 't': 66, 'y': 68, 'u': 70, 'o': 73, 'p': 75}
ALL_KEYS = {**WHITE_KEYS, **BLACK_KEYS}
OCTAVE_LABELS = {60: 'C3', 72: 'C4', 84: 'C5'}

class PolyphonicSynth:
    def __init__(self) -> None:
        self.active_notes = {}
        self.sample_rate = SR
        self.buffer_size = BUFFER_SIZE
        self._lock = threading.Lock()
        self.last_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.waveform_type = 'sine'
        self.use_global_waveform = False
        self.osc1_wave = 'sine'; self.osc1_detune = 0.0; self.osc1_level = 1.0; self.osc1_phase = 0.0
        self.osc2_wave = 'saw';  self.osc2_detune = 0.0; self.osc2_level = 0.0; self.osc2_phase = 0.0
        self.fx_distortion = 0.0; self.fx_delay_time = 0.3; self.fx_delay_feedback = 0.4
        self.fx_delay_mix = 0.0; self.fx_reverb_mix = 0.0
        self.delay_buffer_size = int(SR * 2.0)
        self.delay_buffer = np.zeros(self.delay_buffer_size, dtype=np.float32)
        self.delay_write_idx = 0
        self.attack_time = 0.01; self.decay_time = 0.1; self.sustain_level = 0.8; self.release_time = 0.3

    def set_waveform(self, w):
        with self._lock: self.waveform_type = w
    def set_osc_wave(self, osc, w):
        with self._lock:
            if osc == 1: self.osc1_wave = w
            elif osc == 2: self.osc2_wave = w
    def set_osc_detune(self, osc, v):
        with self._lock:
            if osc == 1: self.osc1_detune = v
            elif osc == 2: self.osc2_detune = v
    def set_osc_level(self, osc, v):
        with self._lock:
            if osc == 1: self.osc1_level = v
            elif osc == 2: self.osc2_level = v
    def set_osc_phase(self, osc, v):
        with self._lock:
            if osc == 1: self.osc1_phase = v
            elif osc == 2: self.osc2_phase = v

    def note_on(self, midi_note):
        with self._lock:
            freq = 440 * (2 ** ((midi_note - 69) / 12))
            self.active_notes[midi_note] = {
                'freq': freq, 'phase1': self.osc1_phase * np.pi * 2,
                'phase2': self.osc2_phase * np.pi * 2, 'vel': 0.8,
                'start_time': time.time(), 'envelope_stage': 'attack',
                'envelope_value': 0.0, 'release_start': None
            }

    def note_off(self, midi_note):
        with self._lock:
            if midi_note in self.active_notes:
                n = self.active_notes[midi_note]
                if n['envelope_stage'] != 'release':
                    n['envelope_stage'] = 'release'; n['release_start'] = time.time()

    def _get_envelope_value(self, note, current_time):
        elapsed = current_time - note['start_time']
        if note['envelope_stage'] == 'attack':
            if elapsed < self.attack_time: return elapsed / self.attack_time
            else: note['envelope_stage'] = 'decay'; elapsed -= self.attack_time
        if note['envelope_stage'] == 'decay':
            if elapsed < self.decay_time:
                return 1.0 - (1.0 - self.sustain_level) * (elapsed / self.decay_time)
            else: note['envelope_stage'] = 'sustain'; return self.sustain_level
        if note['envelope_stage'] == 'sustain': return self.sustain_level
        if note['envelope_stage'] == 'release':
            if note['release_start'] is None: return 0.0
            re = current_time - note['release_start']
            if re >= self.release_time: return 0.0
            return note.get('envelope_value', self.sustain_level) * (1.0 - re / self.release_time)
        return 0.0

    def set_effect_param(self, name, value):
        with self._lock:
            if name == 'distortion': self.fx_distortion = value
            elif name == 'delay_time': self.fx_delay_time = value
            elif name == 'delay_feedback': self.fx_delay_feedback = value
            elif name == 'delay_mix': self.fx_delay_mix = value
            elif name == 'reverb_mix': self.fx_reverb_mix = value

    def _apply_effects(self, buf):
        if self.fx_distortion > 0.01:
            buf = np.tanh(buf * (1.0 + self.fx_distortion * 10.0))
        if self.fx_delay_mix > 0.01 or self.fx_reverb_mix > 0.01:
            ds = max(1, min(int(self.fx_delay_time * self.sample_rate), self.delay_buffer_size - 1))
            wet = max(self.fx_delay_mix, self.fx_reverb_mix)
            fb  = self.fx_delay_feedback
            if self.fx_reverb_mix > 0.5: fb = 0.7 + self.fx_reverb_mix * 0.25
            out = np.zeros_like(buf)
            for i in range(len(buf)):
                ri = int(self.delay_write_idx - ds)
                if ri < 0: ri += self.delay_buffer_size
                d = self.delay_buffer[ri]
                self.delay_buffer[self.delay_write_idx] = buf[i] + d * fb
                out[i] = buf[i] * (1 - wet) + d * wet
                self.delay_write_idx = (self.delay_write_idx + 1) % self.delay_buffer_size
            buf = out
        return buf

    def _generate_wave(self, phases, wave_type, detune_cents):
        if detune_cents != 0.0:
            phases = (phases * 2 ** (detune_cents / 1200.0)) % 1.0
        if wave_type == 'sine': return np.sin(2 * np.pi * phases)
        elif wave_type == 'square': return np.where(phases < 0.5, 1.0, -1.0)
        elif wave_type == 'saw': return 2 * (phases - 0.5)
        elif wave_type == 'triangle': return 2 * np.abs(2 * (phases - 0.5)) - 1
        return np.sin(2 * np.pi * phases)

    def generate_audio(self, num_samples):
        buf = np.zeros(num_samples, dtype=np.float32)
        ct  = time.time()
        with self._lock:
            ac  = len(self.active_notes)
            if ac == 0:
                self.last_buffer = self._apply_effects(buf); return self.last_buffer
            rem = []
            for mn, nd in list(self.active_notes.items()):
                env = self._get_envelope_value(nd, ct)
                nd['envelope_value'] = env
                if env <= 0.001: rem.append(mn); continue
                t = np.arange(num_samples)
                p1 = (nd['phase1'] / (np.pi*2) + (nd['freq'] / self.sample_rate) * t) % 1.0
                p2 = (nd['phase2'] / (np.pi*2) + (nd['freq'] / self.sample_rate) * t) % 1.0
                w1 = self.waveform_type if self.use_global_waveform else self.osc1_wave
                w2 = self.waveform_type if self.use_global_waveform else self.osc2_wave
                o1 = self._generate_wave(p1, w1, self.osc1_detune)
                o2 = self._generate_wave(p2, w2, self.osc2_detune)
                buf += (o1 * self.osc1_level + o2 * self.osc2_level) * nd['vel'] * env
                nd['phase1'] = p1[-1] * 2 * np.pi; nd['phase2'] = p2[-1] * 2 * np.pi
            for mn in rem: del self.active_notes[mn]
            if ac > 1:
                mx = np.max(np.abs(buf))
                if mx > 0: buf = buf / mx * 0.9
            buf *= AMP
            buf  = self._apply_effects(buf)
            buf  = np.clip(buf, -1.0, 1.0)
            self.last_buffer = buf
            return buf

class RotaryKnob(Canvas):
    def __init__(self, parent, value=0.0, min_val=0.0, max_val=1.0,
                 label="", format_fn=None, on_change=None, size=52, **kw):
        super().__init__(parent, width=size+40, height=size+60, bg=BG_BLACK, highlightthickness=0, **kw)
        self.value = value; self.default_value = value
        self.min_val = min_val; self.max_val = max_val
        self.label = label; self.format_fn = format_fn or (lambda v: f'{v:.0f}')
        self.on_change = on_change; self.size = size
        self.radius = size / 2; self.center_x = self.radius + 20; self.center_y = self.radius + 30
        self.min_angle = -135; self.max_angle = 135
        self.is_dragging = False; self.drag_start_y = 0; self.drag_start_value = 0
        self.bind('<ButtonPress-1>', self._on_mouse_down)
        self.bind('<B1-Motion>', self._on_mouse_drag)
        self.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.bind('<Double-Button-1>', self._on_double_click)
        self.bind('<MouseWheel>', self._on_mouse_wheel)
        self.bind('<Button-4>', self._on_mouse_wheel)
        self.bind('<Button-5>', self._on_mouse_wheel)
        self.draw()

    def _get_angle(self):
        if self.max_val == self.min_val: return self.min_angle
        return self.min_angle + ((self.value - self.min_val) / (self.max_val - self.min_val)) * (self.max_angle - self.min_angle)

    def set_value(self, value):
        old = self.value; self.value = max(self.min_val, min(self.max_val, value))
        if abs(self.value - old) > 0.001:
            self.draw()
            if self.on_change: self.on_change(self.value)

    def reset_to_default(self):
        if abs(self.value - self.default_value) > 0.001: self.set_value(self.default_value)

    def _on_mouse_down(self, e):
        if math.sqrt((e.x-self.center_x)**2+(e.y-self.center_y)**2) < self.radius*0.8:
            self.is_dragging = True; self.drag_start_y = e.y_root; self.drag_start_value = self.value
            self.config(cursor='fleur')

    def _on_mouse_drag(self, e):
        if not self.is_dragging: return
        self.set_value(self.drag_start_value + (self.drag_start_y - e.y_root) * (self.max_val - self.min_val) / 200)

    def _on_mouse_up(self, e): self.is_dragging = False; self.config(cursor='hand2')

    def _on_double_click(self, e):
        if math.sqrt((e.x-self.center_x)**2+(e.y-self.center_y)**2) < self.radius*0.8:
            self.reset_to_default()

    def _on_mouse_wheel(self, e):
        self.set_value(self.value + (0.02 if (e.num == 4 or e.delta > 0) else -0.02) * (self.max_val - self.min_val))
        return 'break'

    def draw(self):
        self.delete('all')
        a = self._get_angle(); ar = math.radians(a)
        self.create_oval(self.center_x-self.radius, self.center_y-self.radius,
                         self.center_x+self.radius, self.center_y+self.radius, fill=KNOB_BODY, outline=BORDER)
        ir = self.radius * 0.75
        self.create_oval(self.center_x-ir, self.center_y-ir, self.center_x+ir, self.center_y+ir, fill=KNOB_SURFACE, outline=BORDER)
        ar2 = self.radius * 0.90
        if self.value > self.min_val:
            self.create_arc(self.center_x-ar2, self.center_y-ar2, self.center_x+ar2, self.center_y+ar2,
                            start=self.min_angle, extent=a-self.min_angle, style='arc', outline=FG_BRIGHT, width=3)
        lsr = self.radius * 0.2; ler = self.radius * 0.6
        sx = self.center_x + lsr * math.cos(ar - math.pi/2); sy = self.center_y + lsr * math.sin(ar - math.pi/2)
        ex = self.center_x + ler * math.cos(ar - math.pi/2); ey = self.center_y + ler * math.sin(ar - math.pi/2)
        self.create_line(sx, sy, ex, ey, fill=FG_BRIGHT, width=2)
        cr = self.radius * 0.15
        self.create_oval(self.center_x-cr, self.center_y-cr, self.center_x+cr, self.center_y+cr, fill=BORDER, outline=KNOB_SURFACE)
        self.create_text(self.center_x, self.center_y+self.radius+25, text=self.label.upper(), font=(FONT_MAIN,7,'bold'), fill=FG_DIM)
        self.create_text(self.center_x, 12, text=self.format_fn(self.value), font=(FONT_MAIN,9,'bold'), fill=FG_WHITE)

class PianoKeyboard(Canvas):
    def __init__(self, parent, start_note=60, num_octaves=2, **kw):
        super().__init__(parent, bg=BG_BLACK, highlightthickness=0, **kw)
        self.start_note = start_note; self.num_octaves = num_octaves
        self.white_key_width = 40; self.black_key_width = 24
        self.white_key_height = 120; self.black_key_height = 75
        self.active_keys = set(); self.key_rects = {}; self._draw_keyboard()

    def _draw_keyboard(self):
        self.delete('all'); self.key_rects = {}; x = 10
        white_notes = [0, 2, 4, 5, 7, 9, 11]
        for octave in range(self.num_octaves):
            for offset in white_notes:
                mn = self.start_note + octave*12 + offset
                kl = next((k.upper() for k, v in WHITE_KEYS.items() if v == mn), None)
                r  = self.create_rectangle(x, 0, x+self.white_key_width, self.white_key_height, fill=BG_DARK, outline=BORDER)
                self.key_rects[mn] = r
                if kl: self.create_text(x+self.white_key_width/2, self.white_key_height-20, text=kl, fill=FG_DIM, font=(FONT_MAIN,10))
                if mn in OCTAVE_LABELS: self.create_text(x+self.white_key_width/2, 15, text=OCTAVE_LABELS[mn], fill=FG_BRIGHT, font=(FONT_MAIN,9,'bold'))
                x += self.white_key_width
        x = 10; black_notes = [(1,0.7),(3,1.7),(6,3.7),(8,4.7),(10,5.7)]
        for octave in range(self.num_octaves):
            for offset, wko in black_notes:
                mn = self.start_note + octave*12 + offset
                if mn in self.key_rects: continue
                kl = next((k.upper() for k, v in BLACK_KEYS.items() if v == mn), None)
                bx = x + wko*self.white_key_width - self.black_key_width/2
                r  = self.create_rectangle(bx, 0, bx+self.black_key_width, self.black_key_height, fill='#1a1a1a', outline=BORDER)
                self.key_rects[mn] = r
                if kl: self.create_text(bx+self.black_key_width/2, self.black_key_height-15, text=kl, fill=FG_DIM, font=(FONT_MAIN,9))
            x += 7*self.white_key_width
        self.config(width=x+10, height=self.white_key_height+10)

    def highlight_key(self, mn, active=True):
        if mn not in self.key_rects: return
        r = self.key_rects[mn]
        if active: self.itemconfig(r, fill=FG_BRIGHT, outline=FG_WHITE); self.active_keys.add(mn)
        else:
            is_b = mn in [61,63,66,68,70,73,75,78,80,82,85,87]
            self.itemconfig(r, fill='#1a1a1a' if is_b else BG_DARK, outline=BORDER); self.active_keys.discard(mn)

def sep(parent, vertical=False):
    return tk.Frame(parent, bg=BORDER, width=1 if vertical else 100, height=1 if not vertical else 100)

def label(parent, text, size=9, color=FG_DIM, bold=False):
    return tk.Label(parent, text=text, bg=BG_BLACK, fg=color, font=(FONT_MAIN, size, 'bold' if bold else 'normal'))

class SerumApp:
    def __init__(self, root):
        self.root = root; root.title('Nill OSC')
        root.configure(bg=BG_BLACK); root.geometry('1200x900'); root.minsize(1000, 800)
        self.synth = PolyphonicSynth()
        self._wt_canvases = []; self._playing_notes = {}
        self._waveform_btns = {}; self._waveform_index = 0
        self.stream = None; self.display_canvas = None; self.animation_id = None; self.piano_keyboard = None
        self._build()
        root.after(100, self._init_mini_waves); root.after(100, self._start_audio)
        root.bind('<KeyPress>', self._on_key_press); root.bind('<KeyRelease>', self._on_key_release)
        root.protocol('WM_DELETE_WINDOW', self._on_closing)

    def _start_audio(self):
        try:
            self.stream = sd.OutputStream(channels=1, samplerate=SR, blocksize=BUFFER_SIZE, callback=self._audio_callback)
            self.stream.start(); print("[SYSTEM] Audio stream started")
        except Exception as e: print(f"[ERROR] Audio: {e}")

    def _audio_callback(self, outdata, frames, time_info, status):
        if status: print(f"[STATUS] {status}")
        outdata[:, 0] = self.synth.generate_audio(frames)

    def _on_closing(self):
        if self.stream: self.stream.stop(); self.stream.close()
        if self.animation_id: self.root.after_cancel(self.animation_id)
        self.root.destroy()

    def _build(self):
        self._build_titlebar()
        body = tk.Frame(self.root, bg=BG_BLACK); body.pack(fill='both', expand=True)
        self._build_left(body); self._build_right(body); self._build_center(body)
        self._build_keyboard(); self._build_bottombar()

    def _build_titlebar(self):
        bar = tk.Frame(self.root, bg=BG_BLACK, height=35); bar.pack(fill='x'); bar.pack_propagate(False)
        sep(bar).pack(side='bottom', fill='x')
        label(bar, 'Nill OSC', size=11, color=FG_WHITE, bold=True).pack(side='left', padx=15)
        label(tk.Frame(bar, bg=BG_BLACK), 'OSC 1 + OSC 2', size=9).pack(side='left', padx=15)

    def _build_bottombar(self):
        bar = tk.Frame(self.root, bg=BG_BLACK, height=28); bar.pack(fill='x', side='bottom'); bar.pack_propagate(False)
        sep(bar).pack(side='top', fill='x')
        label(bar, 'PRESET: DEFAULT_PATCH', size=8).pack(side='left', padx=15)
        self.cpu_label = label(bar, 'VOICES: 0/16', size=8); self.cpu_label.pack(side='right', padx=15)

    def _build_keyboard(self):
        kf = tk.Frame(self.root, bg=BG_BLACK, height=140); kf.pack(fill='x', side='bottom')
        sep(kf).pack(side='top', fill='x')
        self.piano_keyboard = PianoKeyboard(kf, start_note=60, num_octaves=2); self.piano_keyboard.pack(pady=10)

    def _build_left(self, parent):
        outer = tk.Frame(parent, bg=BG_BLACK, width=320); outer.pack(side='left', fill='y'); outer.pack_propagate(False)
        sep(outer, vertical=True).pack(side='right', fill='y')
        cv = tk.Canvas(outer, bg=BG_BLACK, highlightthickness=0)
        sb = tk.Scrollbar(outer, orient='vertical', command=cv.yview, bg=BG_BLACK, troughcolor=BG_DARK, activebackground=FG_DIM)
        inner = tk.Frame(cv, bg=BG_BLACK)
        inner.bind('<Configure>', lambda e: cv.configure(scrollregion=cv.bbox('all')))
        wid = cv.create_window((0,0), window=inner, anchor='nw')
        cv.configure(yscrollcommand=sb.set)
        cv.bind('<Configure>', lambda e: cv.itemconfig(wid, width=e.width))
        cv.bind('<MouseWheel>', lambda e: cv.yview_scroll(int(-1*(e.delta/120)),'units'))
        cv.bind('<Button-4>', lambda e: cv.yview_scroll(-1,'units'))
        cv.bind('<Button-5>', lambda e: cv.yview_scroll(1,'units'))
        cv.pack(side='left', fill='both', expand=True); sb.pack(side='right', fill='y')
        self._build_waveform_selector(inner)
        self._build_osc(inner, 1, 'OSC 1', 'sine')
        self._build_osc(inner, 2, 'OSC 2', 'saw')

    def _build_waveform_selector(self, parent):
        sep(parent).pack(fill='x')
        sec = tk.Frame(parent, bg=BG_BLACK); sec.pack(fill='x', padx=12, pady=10)
        label(sec, '[GLOBAL WAVEFORM]', size=9, color=FG_WHITE, bold=True).pack(anchor='w', pady=(0,6))
        label(sec, 'M=Next N=Prev (overrides OSC)', size=7, color=FG_DIM).pack(anchor='w', pady=(0,8))
        bf = tk.Frame(sec, bg=BG_BLACK); bf.pack(fill='x')
        for i, (wave, text) in enumerate([('sine','SINE'),('triangle','TRI'),('saw','SAW'),('square','SQR')]):
            is_active = (wave == 'sine')
            btn = tk.Button(bf, text=f'[{text}]', bg=FG_BRIGHT if is_active else BORDER,
                            fg=BG_BLACK if is_active else FG_DIM, font=(FONT_MAIN,8,'bold'),
                            relief='flat', width=6, height=2, cursor='hand2',
                            command=lambda w=wave: self._set_global_waveform(w))
            btn.grid(row=i//2, column=i%2, padx=4, pady=4, sticky='ew'); self._waveform_btns[wave] = btn
        bf.columnconfigure(0, weight=1); bf.columnconfigure(1, weight=1)

    def _set_global_waveform(self, waveform):
        if waveform not in WAVEFORMS: return
        self.synth.use_global_waveform = True; self.synth.set_waveform(waveform)
        self._waveform_index = WAVEFORMS.index(waveform)
        for w, b in self._waveform_btns.items():
            b.config(bg=FG_BRIGHT if w==waveform else BORDER, fg=BG_BLACK if w==waveform else FG_DIM)

    def _cycle_waveform(self, d):
        self._waveform_index = (self._waveform_index + d) % len(WAVEFORMS)
        self._set_global_waveform(WAVEFORMS[self._waveform_index])

    def _build_osc(self, parent, osc_num, name, wave_type):
        sep(parent).pack(fill='x')
        sec = tk.Frame(parent, bg=BG_BLACK); sec.pack(fill='x', padx=12, pady=10)
        hdr = tk.Frame(sec, bg=BG_BLACK); hdr.pack(fill='x', pady=(0,8))
        label(hdr, f'[{name}]', size=9, color=FG_WHITE, bold=True).pack(side='left')
        tk.Button(hdr, text='[ON]', bg=FG_BRIGHT, fg=BG_BLACK, font=(FONT_MAIN,8), relief='flat', width=3, cursor='hand2').pack(side='right')
        wc = tk.Canvas(sec, width=280, height=50, bg=BG_BLACK, highlightthickness=1, highlightbackground=BORDER)
        wc.pack(pady=(0,8)); self._wt_canvases.append((wc, wave_type, 2.0))
        wf = tk.Frame(sec, bg=BG_BLACK); wf.pack(fill='x', pady=(0,8))
        label(wf, 'WAVEFORM:', size=7, color=FG_DIM).pack(anchor='w')
        wb = tk.Frame(wf, bg=BG_BLACK); wb.pack(fill='x')
        for wave, text in [('sine','SIN'),('triangle','TRI'),('saw','SAW'),('square','SQR')]:
            tk.Button(wb, text=text, bg=FG_BRIGHT if wave==wave_type else BORDER,
                      fg=BG_BLACK if wave==wave_type else FG_DIM, font=(FONT_MAIN,7),
                      relief='flat', width=5, cursor='hand2',
                      command=lambda w=wave: self._set_osc_wave(osc_num, w)).pack(side='left', padx=2)
        kf = tk.Frame(sec, bg=BG_BLACK); kf.pack(fill='x')
        for lbl, init, mn, mx, fmt, param in [
            ('DET', 0.5, 0.0, 1.0, lambda v: f'{int((v-0.5)*100):+d}', 'detune'),
            ('LEV', 1.0 if osc_num==1 else 0.0, 0.0, 1.0, lambda v: f'{int(v*100)}%', 'level'),
            ('PHS', 0.0, 0.0, 1.0, lambda v: f'{int(v*360)}deg', 'phase'),
        ]:
            cf = tk.Frame(kf, bg=BG_BLACK); cf.pack(side='left', expand=True, fill='x', padx=3)
            vl = label(cf, fmt(init), size=9, color=FG_WHITE, bold=True); vl.pack()
            RotaryKnob(cf, value=init, min_val=mn, max_val=mx, label=lbl, size=52, format_fn=fmt,
                       on_change=lambda v, o=osc_num, p=param, l=vl: self._on_osc_change(o,p,v,l)).pack()

    def _set_osc_wave(self, osc_num, waveform):
        self.synth.use_global_waveform = False; self.synth.set_osc_wave(osc_num, waveform)

    def _on_osc_change(self, osc_num, param, value, lbl):
        if param == 'detune':   lbl.config(text=f'{int((value-0.5)*100):+d}'); self.synth.set_osc_detune(osc_num, int((value-0.5)*100))
        elif param == 'level':  lbl.config(text=f'{int(value*100)}%');           self.synth.set_osc_level(osc_num, value)
        elif param == 'phase':  lbl.config(text=f'{int(value*360)}deg');          self.synth.set_osc_phase(osc_num, value)

    def _build_right(self, parent):
        outer = tk.Frame(parent, bg=BG_BLACK, width=320); outer.pack(side='right', fill='y'); outer.pack_propagate(False)
        sep(outer, vertical=True).pack(side='left', fill='y')
        cv = tk.Canvas(outer, bg=BG_BLACK, highlightthickness=0)
        sb = tk.Scrollbar(outer, orient='vertical', command=cv.yview, bg=BG_BLACK, troughcolor=BG_DARK, activebackground=FG_DIM)
        inner = tk.Frame(cv, bg=BG_BLACK)
        inner.bind('<Configure>', lambda e: cv.configure(scrollregion=cv.bbox('all')))
        wid = cv.create_window((0,0), window=inner, anchor='nw')
        cv.configure(yscrollcommand=sb.set)
        cv.bind('<Configure>', lambda e: cv.itemconfig(wid, width=e.width))
        cv.bind('<MouseWheel>', lambda e: cv.yview_scroll(int(-1*(e.delta/120)),'units'))
        cv.bind('<Button-4>', lambda e: cv.yview_scroll(-1,'units'))
        cv.bind('<Button-5>', lambda e: cv.yview_scroll(1,'units'))
        cv.pack(side='left', fill='both', expand=True); sb.pack(side='right', fill='y')
        self._build_master(inner); self._build_filter(inner); self._build_effects(inner)

    def _build_master(self, parent):
        sep(parent).pack(fill='x')
        sec = tk.Frame(parent, bg=BG_BLACK); sec.pack(fill='x', padx=12, pady=10)
        label(sec, '[MASTER]', size=9, color=FG_WHITE, bold=True).pack(anchor='w', pady=(0,8))
        row = tk.Frame(sec, bg=BG_BLACK); row.pack(fill='x')
        for k in [
            {'key':'master_pitch','label':'PITCH','initial':0.5,'format':lambda v: f'{(v-0.5)*24:+.1f}'},
            {'key':'master_glide','label':'GLIDE','initial':0.0,'format':lambda v: f'{int(v*100)}'},
            {'key':'master_level','label':'LEVEL','initial':1.0,'format':lambda v: f'{int(v*127)}'},
        ]: self._create_knob_column(row, k)

    def _create_knob_column(self, parent, kd):
        f = tk.Frame(parent, bg=BG_BLACK); f.pack(side='left', expand=True, fill='x')
        d = label(f, kd['format'](kd['initial']), size=9, color=FG_WHITE, bold=True); d.pack(pady=(0,4))
        RotaryKnob(f, value=kd['initial'], min_val=0.0, max_val=1.0, label=kd['label'], size=52, format_fn=kd['format']).pack()
        label(f, kd['label'], size=7, color=FG_DIM).pack(pady=(4,0))

    def _build_filter(self, parent):
        sep(parent).pack(fill='x')
        sec = tk.Frame(parent, bg=BG_BLACK); sec.pack(fill='x', padx=12, pady=10)
        label(sec, '[FILTER]', size=9, color=FG_WHITE, bold=True).pack(anchor='w', pady=(0,6))
        br = tk.Frame(sec, bg=BG_BLACK); br.pack(fill='x', pady=(0,8))
        for ft in ['LP','HP','BP','NT']:
            tk.Button(br, text=ft, font=(FONT_MAIN,8), relief='flat', bg=BORDER, fg=FG_DIM,
                      activebackground=BG_BLACK, activeforeground=FG_WHITE, cursor='hand2', width=3).pack(side='left', padx=3)
        row = tk.Frame(sec, bg=BG_BLACK); row.pack(fill='x')
        for k in [
            {'key':'filter_cutoff','label':'CUTOFF','initial':0.5,'format':lambda v: f'{int(v*100)}'},
            {'key':'filter_res',   'label':'RES',   'initial':0.3,'format':lambda v: f'{int(v*100)}'},
            {'key':'filter_drive', 'label':'DRV',   'initial':0.0,'format':lambda v: f'{int(v*100)}'},
        ]: self._create_knob_column(row, k)

    def _build_effects(self, parent):
        sep(parent).pack(fill='x')
        sec = tk.Frame(parent, bg=BG_BLACK); sec.pack(fill='x', padx=12, pady=10)
        label(sec, '[FX]', size=9, color=FG_WHITE, bold=True).pack(anchor='w', pady=(0,6))
        for fx_name, params in [
            ('DISTORTION', [{'key':'fx_dist','label':'AMT','initial':0.0,'format':lambda v: f'{int(v*100)}%'}]),
            ('DELAY', [
                {'key':'fx_delay_time','label':'TIME','initial':0.3,'format':lambda v: f'{v:.2f}s'},
                {'key':'fx_delay_fdbk','label':'FDBK','initial':0.4,'format':lambda v: f'{int(v*100)}%'},
                {'key':'fx_delay_mix', 'label':'MIX', 'initial':0.0,'format':lambda v: f'{int(v*100)}%'},
            ]),
            ('REVERB', [{'key':'fx_reverb','label':'AMT','initial':0.0,'format':lambda v: f'{int(v*100)}%'}]),
        ]:
            box = tk.Frame(sec, bg=BG_PANEL, bd=1, relief='solid'); box.pack(fill='x', pady=2)
            label(tk.Frame(box, bg=BG_PANEL), fx_name, size=7).pack(side='left', padx=5, pady=3)
            row = tk.Frame(box, bg=BG_PANEL); row.pack(fill='x', padx=5, pady=(0,4))
            for k in params: self._create_knob_column(row, k)

    def _build_center(self, parent):
        frame = tk.Frame(parent, bg=BG_BLACK); frame.pack(side='left', fill='both', expand=True)
        hdr = tk.Frame(frame, bg=BG_BLACK, height=35); hdr.pack(fill='x'); hdr.pack_propagate(False)
        sep(hdr).pack(side='bottom', fill='x')
        label(hdr, 'VISUALIZER', size=9, color=FG_WHITE, bold=True).pack(side='left', padx=12, pady=8)
        content = tk.Frame(frame, bg=BG_BLACK); content.pack(fill='both', expand=True, padx=12, pady=12)
        self.display_canvas = tk.Canvas(content, bg=BG_BLACK, highlightthickness=1, highlightbackground=BORDER)
        self.display_canvas.pack(fill='both', expand=True)
        self._animate_wavetable()

    def _animate_wavetable(self):
        if not self.display_canvas: return
        w = self.display_canvas.winfo_width(); h = self.display_canvas.winfo_height()
        if w > 10 and h > 10:
            self.display_canvas.delete('all')
            for x in range(0, w+40, 40): self.display_canvas.create_line(x, 0, x, h, fill=BORDER)
            for y in range(0, h+40, 40): self.display_canvas.create_line(0, y, w, y, fill=BORDER)
            self.display_canvas.create_line(0, h//2, w, h//2, fill=FG_DIM, width=1)
            audio = self.synth.last_buffer
            pts = []
            ds = max(1, len(audio)//w)
            for i in range(0, len(audio), ds):
                pts.extend([int(i/len(audio)*w), h//2 - int(audio[i]*(h/2.5))])
            if len(pts) >= 4: self.display_canvas.create_line(pts, fill=FG_WHITE, width=1, smooth=True)
        self.animation_id = self.root.after(33, self._animate_wavetable)

    def _on_key_press(self, e):
        k = e.keysym.lower()
        if k == 'm': self._cycle_waveform(1); return
        if k == 'n': self._cycle_waveform(-1); return
        if k in ALL_KEYS and k not in self._playing_notes:
            mn = ALL_KEYS[k]; self._playing_notes[k] = mn; self.synth.note_on(mn)
            if self.piano_keyboard: self.piano_keyboard.highlight_key(mn, True)

    def _on_key_release(self, e):
        k = e.keysym.lower()
        if k in self._playing_notes:
            mn = self._playing_notes.pop(k); self.synth.note_off(mn)
            if self.piano_keyboard: self.piano_keyboard.highlight_key(mn, False)

    def _init_mini_waves(self):
        for canvas, wtype, freq in self._wt_canvases:
            canvas.delete('all'); canvas.configure(bg=BG_BLACK)
            w = canvas.winfo_width() or 280; h = canvas.winfo_height() or 50
            for i in range(5): canvas.create_line(0, int(h/4*i), w, int(h/4*i), fill=BORDER, width=1)
            pts = []
            for x in range(w):
                t = (x/w) * math.pi * 2 * freq
                if wtype == 'sine':     y = math.sin(t)
                elif wtype == 'saw':    y = 2*((t%(math.pi*2))/(math.pi*2))-1
                elif wtype == 'square': y = 1 if t%(math.pi*2) < math.pi else -1
                else:                   y = 2*abs(2*((t%(math.pi*2))/(math.pi*2))-1)-1
                pts.extend([x, h/2 - y*h/2.5])
            if len(pts) >= 4: canvas.create_line(pts, fill=FG_WHITE, width=1, smooth=True)

if __name__ == '__main__':
    root = tk.Tk()
    try:
        from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except Exception: pass
    app = SerumApp(root); root.mainloop()
'''

def run_embedded_osc() -> None:
    namespace = {"__name__": "__main__", "__file__": str(Path(__file__))}
    exec(EMBEDDED_OSC_CODE, namespace)

def run_embedded_visualizer() -> None:
    try:
        import pygame
    except ImportError:
        print("Missing module: pygame  — install with 'pip install pygame'"); return

    if sd is None:
        print("Missing sounddevice — install with 'pip install sounddevice'"); return

    NATIVE_W, NATIVE_H = 600, 400
    SAMPLE_RATE = 44100; BLOCK_SIZE = 2048
    GHOST_TRAIL_COUNT = 6; GHOST_FADE = 0.12; FRAME_RADIUS = 20

    print("Scanning for audio devices...")
    try: devices = sd.query_devices()
    except Exception as exc: print(f"Could not query devices: {exc}"); return

    AUDIO_DEVICE = None
    keywords = ["stereo mix","what u hear","loopback","wave out","out mix","realtek stereo","virtual audio"]
    for i, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            if any(kw in str(dev.get("name","")).lower() for kw in keywords):
                AUDIO_DEVICE = i; print(f"Found: [{i}] {dev['name']}"); break

    if AUDIO_DEVICE is None:
        print("No Stereo Mix detected — using default input.")

    pygame.init()
    screen = pygame.display.set_mode((NATIVE_W, NATIVE_H))
    pygame.display.set_caption("Ear Candy")
    clock = pygame.time.Clock()

    audio_block = np.zeros((BLOCK_SIZE, 2), dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_block
        audio_block = indata.copy()

    try:
        stream = sd.InputStream(device=AUDIO_DEVICE, callback=audio_callback,
                                blocksize=BLOCK_SIZE, channels=2, samplerate=SAMPLE_RATE)
        stream.start(); print("Audio initialized successfully")
    except Exception as exc:
        print(f"Audio error: {exc}"); pygame.quit(); return

    def catmull_rom(points, samples=8):
        if len(points) < 4: return points
        smoothed = []
        for i in range(1, len(points) - 2):
            p0, p1, p2, p3 = points[i-1], points[i], points[i+1], points[i+2]
            for t in np.linspace(0, 1, samples, endpoint=False):
                t2 = t*t; t3 = t2*t
                x = 0.5*((2*p1[0])+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
                y = 0.5*((2*p1[1])+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
                smoothed.append((x, y))
        smoothed += [points[-2], points[-1]]
        return smoothed

    visual_surface = pygame.Surface((NATIVE_W, NATIVE_H), pygame.SRCALPHA)
    blur_surface   = pygame.Surface((NATIVE_W, NATIVE_H), pygame.SRCALPHA)
    ghost_trails   = []
    frame_rect     = pygame.Rect(15, 15, NATIVE_W-30, NATIVE_H-30)
    frame_surface  = pygame.Surface((NATIVE_W, NATIVE_H), pygame.SRCALPHA)
    pygame.draw.rect(frame_surface, (255,255,255), frame_rect, width=3, border_radius=FRAME_RADIUS)
    clip_surface   = pygame.Surface((NATIVE_W, NATIVE_H), pygame.SRCALPHA)
    pygame.draw.rect(clip_surface, (255,255,255), frame_rect, border_radius=FRAME_RADIUS)
    PANEL_HEIGHT   = 80

    def draw_glass_panel(surface):
        glass = pygame.Surface((NATIVE_W, PANEL_HEIGHT), pygame.SRCALPHA); glass.fill((30,30,30,160)); surface.blit(glass,(0,0))
        ref   = pygame.Surface((NATIVE_W, PANEL_HEIGHT//2), pygame.SRCALPHA)
        for y in range(ref.get_height()): pygame.draw.line(ref,(255,255,255,int(80*(1-y/ref.get_height()))),(0,y),(NATIVE_W,y))
        surface.blit(ref,(0,0))

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            blur_surface.fill((0,0,0,40)); visual_surface.blit(blur_surface,(0,0))
            mono = np.mean(audio_block, axis=1)
            al   = np.mean(np.abs(mono)) * 10
            if al > 0.001:
                mx = np.max(np.abs(mono))
                if mx > 0: mono = mono / mx
            else:
                t = pygame.time.get_ticks()/1000.0
                mono = np.sin(np.linspace(0,2*np.pi,len(mono))+t*0.5)*0.3
            mono   = np.interp(np.linspace(0,len(mono),frame_rect.width), np.arange(len(mono)), mono)
            mid_y  = frame_rect.y + frame_rect.height/2
            amp    = frame_rect.height * 0.45
            points = [(frame_rect.x+i, int(mid_y+s*amp)) for i, s in enumerate(mono)]
            if len(points) > 4: points = catmull_rom(points, samples=4)
            ghost_trails.insert(0, points)
            if len(ghost_trails) > GHOST_TRAIL_COUNT: ghost_trails.pop()
            if al <= 0.001: ghost_trails.clear()
            wave_surface = pygame.Surface((NATIVE_W, NATIVE_H), pygame.SRCALPHA)
            for idx, trail in enumerate(ghost_trails):
                alpha = int(255*max(0,1-idx*GHOST_FADE))
                if len(trail) > 1: pygame.draw.lines(wave_surface,(255,255,255,alpha),False,trail,2)
            if len(points) > 1: pygame.draw.lines(wave_surface,(255,255,255,255),False,points,5)
            mask = pygame.mask.from_surface(clip_surface)
            ms   = mask.to_surface(setcolor=(255,255,255,255),unsetcolor=(0,0,0,0))
            wave_surface.blit(ms,(0,0),special_flags=pygame.BLEND_RGBA_MULT)
            ib = pygame.Surface((frame_rect.width,frame_rect.height),pygame.SRCALPHA); ib.fill((10,10,10))
            visual_surface.blit(ib,frame_rect.topleft)
            visual_surface.blit(wave_surface,(0,0)); visual_surface.blit(frame_surface,(0,0))
            draw_glass_panel(visual_surface)
            screen.fill((0,0,0)); screen.blit(visual_surface,(0,0))
            pygame.display.flip(); clock.tick(60)
    finally:
        pygame.quit()
        try: stream.stop(); stream.close()
        except Exception: pass

# ============================ MAIN ============================

def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--nill-visualizer":
        run_embedded_visualizer(); return
    if len(sys.argv) > 1 and sys.argv[1] == "--nill-osc":
        run_embedded_osc(); return

    print("Nill | Terminal Interface DAW")
    app    = QApplication(sys.argv)
    window = Nill()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
