#!/usr/bin/env python3
"""
Nill (open source) | Professional Wavetable Synthesizer
Terminal DAW with FL Studio-like Features, Advanced Piano Roll, Audio Engine, and Command Terminal
"""

import sys
import os
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict
from enum import Enum
import queue
import json
import time
from pathlib import Path
import subprocess

try:
    import sounddevice as sd
except (ImportError, OSError):
    # ImportError = package missing; OSError = PortAudio missing on the system.
    sd = None

try:
    from testwavetable import PolyphonicSynth  # type: ignore
except Exception:
    PolyphonicSynth = None  # filled in by the built-in fallback below

# winsound is Windows-only; we replace it with a portable sounddevice beep.
try:
    import winsound  # type: ignore
except ImportError:
    winsound = None


# ---------------------------------------------------------------------------
# Built-in cross-platform polyphonic synth fallback.
# Used when the optional `testwavetable` module isn't available so that the
# DAW still produces sound on every OS that has `sounddevice` installed.
# ---------------------------------------------------------------------------
class _BuiltInPolyphonicSynth:
    """A small, dependency-free polyphonic synth with ADSR envelopes."""

    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        # MIDI pitch -> dict(state, env, phase, freq)
        self._voices: Dict[int, dict] = {}
        self._lock = threading.Lock()
        # ADSR (seconds / level)
        self.attack = 0.005
        self.decay = 0.10
        self.sustain = 0.75
        self.release = 0.20
        self.gain = 0.22  # master gain to keep things polite

    @staticmethod
    def _midi_to_freq(pitch: int) -> float:
        return 440.0 * (2.0 ** ((pitch - 69) / 12.0))

    def note_on(self, pitch: int) -> None:
        with self._lock:
            self._voices[pitch] = {
                "state": "attack",
                "env": 0.0,
                "phase": 0.0,
                "freq": self._midi_to_freq(pitch),
                "released_at_env": 0.0,
            }

    def note_off(self, pitch: int) -> None:
        with self._lock:
            v = self._voices.get(pitch)
            if v is not None and v["state"] != "release":
                v["state"] = "release"
                v["released_at_env"] = v["env"]

    def all_notes_off(self) -> None:
        with self._lock:
            for v in self._voices.values():
                if v["state"] != "release":
                    v["state"] = "release"
                    v["released_at_env"] = v["env"]

    def generate_audio(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float32)
        sr = self.sample_rate
        dt = 1.0 / sr

        a_step = 1.0 / max(1, int(self.attack * sr))
        d_step = (1.0 - self.sustain) / max(1, int(self.decay * sr))
        r_steps = max(1, int(self.release * sr))

        with self._lock:
            dead = []
            for pitch, v in self._voices.items():
                phase = v["phase"]
                freq = v["freq"]
                env = v["env"]
                state = v["state"]
                released = v["released_at_env"]

                phase_inc = 2.0 * np.pi * freq * dt
                # Build the waveform sample-by-sample so envelopes track properly.
                # Mix sine + a soft saw harmonic for some warmth.
                idx = np.arange(frames)
                phases = phase + phase_inc * idx
                # main + 2nd harmonic (a touch) + slight detune
                wave = (
                    0.70 * np.sin(phases)
                    + 0.20 * np.sin(phases * 2.0) * 0.5
                    + 0.10 * np.sin(phases * 1.005)
                )
                v["phase"] = float((phase + phase_inc * frames) % (2.0 * np.pi))

                # Build per-sample envelope
                envs = np.empty(frames, dtype=np.float32)
                for i in range(frames):
                    if state == "attack":
                        env += a_step
                        if env >= 1.0:
                            env = 1.0
                            state = "decay"
                    elif state == "decay":
                        env -= d_step
                        if env <= self.sustain:
                            env = self.sustain
                            state = "sustain"
                    elif state == "sustain":
                        pass  # hold
                    elif state == "release":
                        # Linear release from released_at_env down to 0
                        env -= released / r_steps
                        if env <= 0.0:
                            env = 0.0
                    envs[i] = env

                v["env"] = float(env)
                v["state"] = state

                out += (wave * envs).astype(np.float32)

                if state == "release" and env <= 0.0:
                    dead.append(pitch)

            for p in dead:
                self._voices.pop(p, None)

        # Soft clip / gain
        out = np.tanh(out * self.gain).astype(np.float32)
        return out


# If the optional wavetable engine is unavailable, transparently use ours.
if PolyphonicSynth is None:
    PolyphonicSynth = _BuiltInPolyphonicSynth


def _play_beep(freq: int, duration_ms: int, sample_rate: int = 44100) -> None:
    """Cross-platform replacement for winsound.Beep using sounddevice."""
    if sd is None:
        return
    try:
        n = max(1, int(sample_rate * duration_ms / 1000.0))
        t = np.arange(n) / sample_rate
        tone = 0.25 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
        # tiny fade in/out to avoid clicks
        fade = min(256, n // 20)
        if fade > 0:
            tone[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            tone[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        sd.play(tone, samplerate=sample_rate, blocking=False)
    except Exception:
        pass

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QLabel, QPushButton, QScrollArea, QListWidget,
    QListWidgetItem, QFileDialog, QSplitter, QMessageBox, QComboBox, QSpinBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread, QRect, QSize, QPoint, QEvent, QElapsedTimer
from PySide6.QtGui import QFont, QColor, QKeySequence, QPainter, QBrush, QPen, QTextOption, QLinearGradient, QRadialGradient


class WavetableType(Enum):
    SINE = 0
    SAWTOOTH = 1
    SQUARE = 2
    TRIANGLE = 3
    ANALOG = 4
    NOISE = 5


@dataclass
class OscillatorParams:
    waveform: WavetableType = WavetableType.SINE
    position: float = 0.0
    frequency: float = 440.0
    detune: float = 0.0
    level: float = 1.0
    enabled: bool = True


@dataclass
class Note:
    pitch: int  # 0-95 (0-7 octaves, 12 semitones each)
    start_time: float  # in beats
    duration: float  # in beats
    velocity: int  # 0-127 (volume/loudness)
    is_ghost: bool = False
    is_slide: bool = False
    slide_target: Optional[int] = None
    
    def contains_time(self, beat: float) -> bool:
        """Check if beat falls within this note's duration"""
        return self.start_time <= beat < self.start_time + self.duration
    
    def overlaps_pitch(self, pitch: int) -> bool:
        """Check if note is on the same pitch row"""
        return self.pitch == pitch
    
    def point_in_note(self, beat: float, pitch: int, tolerance: float = 0.1) -> bool:
        """Check if a point is within this note"""
        return (self.start_time <= beat < self.start_time + self.duration and 
                self.pitch == pitch)


class AudioEngine:
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.bpm = 120
        self.bit_depth = 8
        self.running = False
        
        self.osc1 = OscillatorParams(waveform=WavetableType.SINE)
        self.osc2 = OscillatorParams(waveform=WavetableType.SAWTOOTH, detune=12.0)
        
        self.filter_cutoff = 5000.0
        self.filter_resonance = 0.5
        self.filter_type = "LP"
        
        self.amp_envelope = {"attack": 0.01, "decay": 0.1, "sustain": 0.8, "release": 0.3}
        self.filter_envelope = {"attack": 0.05, "decay": 0.2, "sustain": 0.3, "release": 0.5}
        
        self.phase1 = 0.0
        self.phase2 = 0.0
        
        self.command_queue: queue.Queue = queue.Queue()
        self.status_queue: queue.Queue = queue.Queue()
        
        self.audio_thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=False)
        self.audio_thread.start()
        
    def stop(self) -> None:
        self.running = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
            
    def _audio_loop(self) -> None:
        while self.running:
            try:
                cmd = self.command_queue.get_nowait()
                self._process_dsp_command(cmd)
            except queue.Empty:
                pass
            
            buffer = self._generate_buffer()
            
            try:
                self.status_queue.put_nowait({
                    'buffer_size': len(buffer),
                    'rms': float(np.sqrt(np.mean(buffer**2))),
                    'peak': float(np.max(np.abs(buffer)))
                })
            except queue.Full:
                pass
            
            time.sleep(self.buffer_size / self.sample_rate * 0.5)
    
    def _generate_buffer(self) -> np.ndarray:
        buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        if self.osc1.enabled:
            osc1_buffer = self._generate_wavetable(
                self.osc1.frequency, self.osc1.waveform, self.osc1.position
            )
            buffer += osc1_buffer * self.osc1.level
        
        if self.osc2.enabled:
            osc2_freq = self.osc1.frequency * (2 ** (self.osc2.detune / 1200.0))
            osc2_buffer = self._generate_wavetable(
                osc2_freq, self.osc2.waveform, self.osc2.position
            )
            buffer += osc2_buffer * self.osc2.level
        
        max_val = np.max(np.abs(buffer))
        if max_val > 0:
            buffer = buffer / max_val * 0.9
        
        buffer = self._bit_crush(buffer, self.bit_depth)
        
        return buffer
    
    def _generate_wavetable(self, freq: float, waveform: WavetableType, position: float) -> np.ndarray:
        time_array = np.arange(self.buffer_size) / self.sample_rate
        phase = 2 * np.pi * freq * time_array
        
        if waveform == WavetableType.SINE:
            return np.sin(phase)
        elif waveform == WavetableType.SAWTOOTH:
            return 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
        elif waveform == WavetableType.SQUARE:
            return np.sign(np.sin(phase))
        elif waveform == WavetableType.TRIANGLE:
            return 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
        elif waveform == WavetableType.NOISE:
            return np.random.uniform(-1, 1, self.buffer_size)
        else:
            sine = np.sin(phase)
            saw = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
            return sine * (1 - position) + saw * position
    
    def _bit_crush(self, signal: np.ndarray, bits: int) -> np.ndarray:
        if bits >= 16:
            return signal
        
        levels = 2 ** bits
        quantized = np.round(signal * (levels / 2)) / (levels / 2)
        return np.clip(quantized, -1.0, 1.0)
    
    def _process_dsp_command(self, cmd: dict) -> None:
        if cmd['type'] == 'set_bpm':
            self.bpm = cmd['value']
        elif cmd['type'] == 'set_bit_depth':
            self.bit_depth = max(1, min(16, cmd['value']))
        elif cmd['type'] == 'set_filter_cutoff':
            self.filter_cutoff = cmd['value']
        elif cmd['type'] == 'toggle_osc':
            if cmd['osc'] == 1:
                self.osc1.enabled = cmd['value']
            else:
                self.osc2.enabled = cmd['value']
    
    def queue_command(self, cmd: dict) -> None:
        try:
            self.command_queue.put_nowait(cmd)
        except queue.Full:
            pass


class PianoRoll(QWidget):
    """Enhanced interactive piano roll with FL Studio features"""
    
    note_added = Signal(Note)
    note_removed = Signal(Note)
    note_modified = Signal(Note)
    layer_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.notes: List[Note] = []
        self.selected_notes: List[Note] = []
        self.ghost_notes: List[Note] = []
        
        # Layout constants
        self.OCTAVES = 8
        self.PITCHES_PER_OCTAVE = 12
        self.TOTAL_PITCHES = self.OCTAVES * self.PITCHES_PER_OCTAVE  # 96
        
        self.PIXEL_PER_BEAT = 40
        self.PIXEL_PER_PITCH = 16
        
        self.HEADER_HEIGHT = 40
        self.HEADER_WIDTH = 60  # Reduced - just for piano key sidebar
        self.NOTE_AREA_PADDING = 5  # Space between piano keys and note grid
        
        # State
        self.bpm = 120
        self.scale = "Major"
        self.drum_samples: List[str] = []
        self.default_drums = self._generate_default_drums()
        
        # Layering system
        self.layers: dict = {
            "drums": {"notes": [], "visible": True, "color": "#9a9a9a", "sample_name": "Drums", "sample_path": None},
            "bass": {"notes": [], "visible": True, "color": "#bcbcbc", "sample_name": "Bass", "sample_path": None},
            "melody": {"notes": [], "visible": True, "color": "#7e7e7e", "sample_name": "Melody", "sample_path": None}
        }
        self.current_layer = "drums"  # Active layer for editing
        # Visualization mode
        self.viz_mode = "standard"  # standard, velocity, gradient, neon
        
        # Grid settings
        self.show_grid = True
        self.grid_opacity = 0.3
        self.show_beat_lines = True
        self.show_measure_lines = True
        
        # Snap to grid settings
        self.snap_enabled = True
        self.snap_value = 0.25
        
        # Interaction - Drawing
        self.dragging = False
        self.drag_start_pos = None
        self.drag_start_beat = None
        self.drag_pitch = None
        self.is_slide_mode = False
        
        # Interaction - Moving notes
        self.moving_note = None
        self.move_offset_beat = 0.0
        self.move_offset_pitch = 0
        self.adjusting_duration_note = None
        self.adjust_start_beat = 0.0
        self.selecting_rect = None
        self.selection_start = None
        
        # Playback
        self.is_playing = False
        self.playback_position = 0.0
        
        # Current editing velocity
        self.current_velocity = 100
        
        # Hover information
        self.hovered_note = None
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumHeight(400)
        
        # Note name reference for y-axis
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def _generate_default_drums(self) -> Dict[int, np.ndarray]:
        """Generate default 8-bit style drum samples using noise"""
        sample_rate = 44100
        drums = {}
        
        # Kick drum (low frequency noise burst)
        kick_duration = 0.2
        kick_samples = int(kick_duration * sample_rate)
        kick = np.random.normal(0, 1, kick_samples)
        # Low-pass filter approximation
        kick = np.convolve(kick, np.ones(100)/100, mode='same')
        # Exponential decay
        decay = np.exp(-np.linspace(0, 5, kick_samples))
        kick *= decay
        drums[36] = kick  # MIDI note 36 for kick
        
        # Snare drum (high frequency noise with body)
        snare_duration = 0.15
        snare_samples = int(snare_duration * sample_rate)
        snare = np.random.normal(0, 1, snare_samples)
        # High-pass for snap
        snare = np.convolve(snare, [1, -0.9], mode='same')
        # Decay
        decay = np.exp(-np.linspace(0, 8, snare_samples))
        snare *= decay
        drums[38] = snare  # MIDI 38 for snare
        
        # Hi-hat (short high noise)
        hihat_duration = 0.1
        hihat_samples = int(hihat_duration * sample_rate)
        hihat = np.random.normal(0, 1, hihat_samples)
        # High-pass
        hihat = np.convolve(hihat, [1, -0.95], mode='same')
        # Fast decay
        decay = np.exp(-np.linspace(0, 15, hihat_samples))
        hihat *= decay
        drums[42] = hihat  # MIDI 42 for closed hi-hat
        
        # Normalize all to prevent clipping
        for key in drums:
            max_val = np.max(np.abs(drums[key]))
            if max_val > 0:
                drums[key] /= max_val
            drums[key] *= 0.8  # Leave headroom
        
        return drums
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse press for note painting, selection, and moving"""
        if event.button() == Qt.MouseButton.LeftButton:
            beat = self.pixel_to_beat(event.position().toPoint().x())
            pitch = self.pixel_to_pitch(event.position().toPoint().y())
            
            # Check if clicking on existing note (for moving or duration adjustment)
            clicked_note = self.get_note_at(beat, pitch)
            
            if clicked_note:
                # Check if clicking near the right edge for duration adjustment
                note_end_beat = clicked_note.start_time + clicked_note.duration
                note_end_pixel = self.beat_to_pixel(note_end_beat)
                
                if abs(event.position().toPoint().x() - note_end_pixel) < 8:  # Within 8 pixels of right edge
                    # Start duration adjustment
                    self.adjusting_duration_note = clicked_note
                    self.adjust_start_beat = clicked_note.start_time + clicked_note.duration
                elif not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    # Start moving note
                    self.moving_note = clicked_note
                    self.move_offset_beat = clicked_note.start_time - beat
                    self.move_offset_pitch = clicked_note.pitch - pitch
            else:
                # Start drawing new note
                self.start_note_drag(event.position().toPoint(), is_slide=False)
        
        elif event.button() == Qt.MouseButton.RightButton:
            self.erase_note_at(event.position().toPoint())
        
        elif event.button() == Qt.MouseButton.MiddleButton:
            # Start selection rectangle
            self.selection_start = event.position().toPoint()
    
    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move for note duration adjustment and moving"""
        beat = self.pixel_to_beat(event.position().toPoint().x())
        pitch = self.pixel_to_pitch(event.position().toPoint().y())
        
        # Update hover
        self.hovered_note = self.get_note_at(beat, pitch)
        
        if self.dragging:
            self.update()
        elif self.adjusting_duration_note:
            # Adjust note duration
            current_beat = self.snap_beat(self.pixel_to_beat(event.position().toPoint().x()))
            new_duration = max(0.25, current_beat - self.adjusting_duration_note.start_time)
            self.adjusting_duration_note.duration = new_duration
            self.note_modified.emit(self.adjusting_duration_note)
            self.update()
        elif self.moving_note:
            # Move note to new position with snap
            new_beat = self.snap_beat(beat + self.move_offset_beat)
            new_pitch = max(0, min(self.TOTAL_PITCHES - 1, pitch + self.move_offset_pitch))
            
            # Allow free movement like in FL Studio (no overlap restrictions)
            self.moving_note.start_time = new_beat
            self.moving_note.pitch = new_pitch
            self.note_modified.emit(self.moving_note)
            
            self.update()
        elif self.selection_start:
            # Update selection rectangle
            self.selecting_rect = event.position().toPoint()
            self.update()
    
    def mouseReleaseEvent(self, event) -> None:
        """Finish note painting or moving"""
        if self.dragging:
            self.finish_note_drag(event.position().toPoint())
        
        if self.adjusting_duration_note:
            self.adjusting_duration_note = None
            self.update()
        
        if self.moving_note:
            self.moving_note = None
            self.update()
        
        if self.selection_start and self.selecting_rect:
            # Select notes in rectangle
            self.select_notes_in_rect(self.selection_start, self.selecting_rect)
            self.selection_start = None
            self.selecting_rect = None
            self.update()
    
    def keyPressEvent(self, event) -> None:
        """Handle keyboard modifiers for slide mode and spacebar for playback"""
        if event.key() == Qt.Key.Key_Shift:
            self.is_slide_mode = True
        elif event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self.toggle_playback()
        elif event.key() == Qt.Key.Key_Delete:
            # Delete selected notes (only those present in current layer)
            notes_to_remove = [note for note in self.selected_notes if note in self.notes]
            for note in notes_to_remove:
                self.notes.remove(note)
                self.note_removed.emit(note)
            self.selected_notes = [note for note in self.selected_notes if note not in notes_to_remove]
            self.update()
        elif event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Select all
            self.selected_notes = self.notes.copy()
            self.update()
        elif event.key() == Qt.Key.Key_D and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Deselect all
            self.selected_notes.clear()
            self.update()
    
    def keyReleaseEvent(self, event) -> None:
        """Handle keyboard modifiers"""
        if event.key() == Qt.Key.Key_Shift:
            self.is_slide_mode = False
    
    def get_note_at(self, beat: float, pitch: int) -> Optional[Note]:
        """Get note at specific beat and pitch"""
        for note in self.notes:
            if note.contains_time(beat) and note.pitch == int(pitch):
                return note
        return None
    
    def select_notes_in_rect(self, start: QPoint, end: QPoint) -> None:
        """Select all notes within rectangle"""
        x1, x2 = min(start.x(), end.x()), max(start.x(), end.x())
        y1, y2 = min(start.y(), end.y()), max(start.y(), end.y())
        
        start_beat = self.pixel_to_beat(x1)
        end_beat = self.pixel_to_beat(x2)
        start_pitch = self.pixel_to_pitch(y2)
        end_pitch = self.pixel_to_pitch(y1)
        
        self.selected_notes = [
            n for n in self.notes
            if start_beat <= n.start_time < end_beat and start_pitch <= n.pitch <= end_pitch
        ]
    
    def start_note_drag(self, pos: QPoint, is_slide: bool) -> None:
        """Begin dragging to create a note - snap on start"""
        beat = self.pixel_to_beat(pos.x())
        beat = self.snap_beat(beat)
        pitch = self.pixel_to_pitch(pos.y())
        
        if beat < 0 or pitch < 0 or pitch >= self.TOTAL_PITCHES:
            return
        
        self.dragging = True
        self.drag_start_pos = pos
        self.drag_start_beat = beat
        self.drag_pitch = pitch
        self.is_slide_mode = is_slide
    
    def is_white_key(self, pitch: int) -> bool:
        """Check if a pitch is on a white key (C, D, E, F, G, A, B)"""
        note_in_octave = pitch % self.PITCHES_PER_OCTAVE
        return note_in_octave in [0, 2, 4, 5, 7, 9, 11]
    
    def note_has_overlap(self, pitch: int, start_time: float, end_time: float) -> bool:
        """Check if a note at this pitch and time range overlaps with existing notes in the layer"""
        for note in self.layers[self.current_layer]["notes"]:
            if note.pitch == pitch:
                note_end = note.start_time + note.duration
                # Check if time ranges overlap
                if not (end_time <= note.start_time or start_time >= note_end):
                    return True
        return False
    
    def finish_note_drag(self, pos: QPoint) -> None:
        """Complete note creation with snap-to-grid and overlap prevention"""
        if not self.dragging or self.drag_start_beat is None or self.drag_pitch is None:
            return
        
        start_beat = self.drag_start_beat
        end_beat = self.snap_beat(self.pixel_to_beat(pos.x()))
        
        if end_beat <= start_beat:
            end_beat = start_beat + self.snap_value
        
        duration = end_beat - start_beat
        
        # Create the note (allow overlaps like in FL Studio)
        note = Note(
            pitch=int(self.drag_pitch),
            start_time=start_beat,
            duration=duration,
            velocity=self.current_velocity,
            is_slide=self.is_slide_mode,
            slide_target=self.pixel_to_pitch(pos.y()) if self.is_slide_mode else None
        )
        
        # Add to current layer
        self.layers[self.current_layer]["notes"].append(note)
        self.notes = self.layers[self.current_layer]["notes"]
        self.note_added.emit(note)
        
        self.dragging = False
        self.drag_start_pos = None
        self.drag_start_beat = None
        self.drag_pitch = None
        self.update()
    
    def erase_note_at(self, pos: QPoint) -> None:
        """Remove note at clicked position"""
        beat = self.pixel_to_beat(pos.x())
        pitch = self.pixel_to_pitch(pos.y())
        
        to_remove = [
            n for n in self.notes
            if n.start_time <= beat < n.start_time + n.duration and n.pitch == int(pitch)
        ]
        
        if not to_remove:
            to_remove = [
                n for n in self.notes
                if abs(n.start_time - beat) < 0.5 and n.pitch == int(pitch)
            ]
        
        for note in to_remove:
            self.notes.remove(note)
            self.note_removed.emit(note)
        
        self.update()
    
    def snap_beat(self, beat: float) -> float:
        """Snap beat to grid if enabled"""
        if not self.snap_enabled:
            return beat
        return round(beat / self.snap_value) * self.snap_value
    
    def set_snap(self, enabled: bool, snap_value: float = 0.25) -> None:
        """Set snap-to-grid settings"""
        self.snap_enabled = enabled
        self.snap_value = snap_value
    
    def set_velocity(self, velocity: int) -> None:
        """Set current velocity for new notes"""
        self.current_velocity = max(1, min(127, velocity))
    
    def set_grid_visibility(self, show: bool) -> None:
        """Toggle grid visibility"""
        self.show_grid = show
        self.update()
    
    def set_current_layer(self, layer: str) -> None:
        """Switch to different layer"""
        if layer in self.layers:
            self.current_layer = layer
            self.notes = self.layers[layer]["notes"]
            self.selected_notes.clear()
            self.layer_changed.emit(layer)
            self.update()
    
    def get_all_notes(self) -> List[Note]:
        """Get all notes from all visible layers"""
        all_notes = []
        for layer_name, layer_data in self.layers.items():
            if layer_data["visible"]:
                all_notes.extend(layer_data["notes"])
        return all_notes
    
    def toggle_layer_visibility(self, layer: str) -> None:
        """Toggle layer visibility"""
        if layer in self.layers:
            self.layers[layer]["visible"] = not self.layers[layer]["visible"]
            self.update()
    
    def add_note_to_current_layer(self, note: Note) -> None:
        """Add note to current layer"""
        self.layers[self.current_layer]["notes"].append(note)
        self.notes = self.layers[self.current_layer]["notes"]
        self.note_added.emit(note)
    
    def clear_layer(self, layer: Optional[str] = None) -> None:
        """Clear all notes from a layer"""
        target_layer = layer if layer else self.current_layer
        if target_layer in self.layers:
            self.layers[target_layer]["notes"].clear()
            if target_layer == self.current_layer:
                self.notes = []
            self.update()
    
    def pixel_to_beat(self, x: int) -> float:
        """Convert pixel x to beat position"""
        note_start_x = self.HEADER_WIDTH + self.NOTE_AREA_PADDING
        return max(0, (x - note_start_x) / self.PIXEL_PER_BEAT)
    
    def pixel_to_pitch(self, y: int) -> int:
        """Convert pixel y to pitch (0-95)"""
        pitch = self.TOTAL_PITCHES - 1 - ((y - self.HEADER_HEIGHT) // self.PIXEL_PER_PITCH)
        return max(0, min(self.TOTAL_PITCHES - 1, pitch))
    
    def beat_to_pixel(self, beat: float) -> int:
        """Convert beat to pixel x"""
        note_start_x = self.HEADER_WIDTH + self.NOTE_AREA_PADDING
        return int(note_start_x + beat * self.PIXEL_PER_BEAT)
    
    def pitch_to_pixel(self, pitch: int) -> int:
        """Convert pitch to pixel y"""
        return int(self.HEADER_HEIGHT + (self.TOTAL_PITCHES - 1 - pitch) * self.PIXEL_PER_PITCH)
    
    def get_pitch_label(self, pitch: int) -> str:
        """Get note name (C4, C#4, etc.)"""
        octave = pitch // self.PITCHES_PER_OCTAVE
        note = self.note_names[pitch % self.PITCHES_PER_OCTAVE]
        return f"{note}{octave}"
    
    def set_visualization_mode(self, mode: str) -> None:
        """Set the visualization mode"""
        self.viz_mode = mode
        self.update()
    
    def toggle_playback(self) -> None:
        """Toggle playback of MIDI sequence"""
        self.is_playing = not self.is_playing
        self.update()
    
    def stop_playback(self) -> None:
        """Stop playback and reset position"""
        self.is_playing = False
        self.playback_position = 0.0
        self.update()
    
    def get_notes_at_beat(self, beat: float) -> List[Note]:
        """Get all notes playing at a specific beat from all visible layers"""
        all_notes = self.get_all_notes()
        return [n for n in all_notes if n.contains_time(beat)]
    
    def get_playback_info(self) -> dict:
        """Get current playback info for audio engine"""
        active_notes = self.get_notes_at_beat(self.playback_position)
        return {
            'is_playing': self.is_playing,
            'position': self.playback_position,
            'active_notes': active_notes,
            'bpm': self.bpm
        }
    
    def paintEvent(self, event) -> None:
        """Render enhanced piano roll with all features"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor("#0a0a0a"))
        
        # Draw grid and pitch labels
        self.draw_pitch_axis(painter)
        self.draw_time_axis(painter)
        if self.show_grid:
            self.draw_grid(painter)
        
        # Draw notes with selected visualization mode
        self.draw_notes(painter)
        
        # Draw separators to keep notes isolated from black keys
        self.draw_key_separators(painter)
        
        # Draw selection rectangle
        if self.selecting_rect:
            self.draw_selection_rect(painter)
        
        # Draw playback position indicator
        if self.is_playing:
            self.draw_playback_indicator(painter)
    
    def draw_pitch_axis(self, painter: QPainter) -> None:
        """Draw left panel with BeepBox-style piano keys (white keys active, black keys disabled)"""
        # Draw background
        painter.fillRect(0, 0, self.HEADER_WIDTH, self.height(), QColor("#0a0a0a"))
        
        for pitch in range(self.TOTAL_PITCHES):
            y = self.pitch_to_pixel(pitch)
            note_in_octave = pitch % self.PITCHES_PER_OCTAVE
            
            # Determine if white or black key (C, D, E, F, G, A, B are white)
            is_white = self.is_white_key(pitch)
            
            # Draw key background
            if is_white:
                # White keys are active and clickable
                painter.fillRect(0, y, self.HEADER_WIDTH, self.PIXEL_PER_PITCH, QBrush(QColor("#e0e0e0")))
                painter.setPen(QPen(QColor("#888888"), 1))
            else:
                # Black keys are disabled (not clickable)
                painter.fillRect(0, y, self.HEADER_WIDTH, self.PIXEL_PER_PITCH, QBrush(QColor("#0a0a0a")))
                painter.setPen(QPen(QColor("#1a1a1a"), 1))
            
            painter.drawRect(0, y, self.HEADER_WIDTH, self.PIXEL_PER_PITCH)
        
        # Right border
        painter.setPen(QPen(QColor("#2a2a2a"), 2))
        painter.drawLine(self.HEADER_WIDTH - 1, 0, self.HEADER_WIDTH - 1, self.height())
    
    def draw_time_axis(self, painter: QPainter) -> None:
        """Draw top panel with beat markers (BeepBox-inspired)"""
        # Draw background for entire top
        painter.fillRect(0, 0, self.width(), self.HEADER_HEIGHT, QColor("#0a0a0a"))
        
        # Piano key area background
        painter.fillRect(0, 0, self.HEADER_WIDTH, self.HEADER_HEIGHT, QColor("#1a1a1a"))
        
        # Bottom border of header
        painter.setPen(QPen(QColor("#2a2a2a"), 2))
        painter.drawLine(0, self.HEADER_HEIGHT - 1, self.width(), self.HEADER_HEIGHT - 1)
        
        # Vertical separator between piano keys and note area
        painter.drawLine(self.HEADER_WIDTH, 0, self.HEADER_WIDTH, self.HEADER_HEIGHT)
        
        # Beat markers and numbers - ONLY in the note area
        note_start_x = self.HEADER_WIDTH + self.NOTE_AREA_PADDING
        painter.setPen(QPen(QColor("#505050"), 1))
        painter.setFont(QFont("Courier New", 8))
        
        beat = 0
        while True:
            x = self.beat_to_pixel(beat)
            if x > self.width():
                break
            
            # Draw marker line
            marker_height = 8 if int(beat) % 4 == 0 else 4
            painter.drawLine(x, self.HEADER_HEIGHT - marker_height, x, self.HEADER_HEIGHT)
            
            # Draw beat number every 4 beats
            if int(beat) % 4 == 0:
                painter.setPen(QPen(QColor("#808080"), 1))
                painter.drawText(
                    QRect(x - 20, 6, 40, 20),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{int(beat)}"
                )
                painter.setPen(QPen(QColor("#505050"), 1))
            
            beat += 1
    
    def draw_grid(self, painter: QPainter) -> None:
        """Draw grid lines with visual hierarchy (only in note area, not in piano sidebar)"""
        # Vertical lines (beats)
        beat = 0
        while True:
            x = self.beat_to_pixel(beat)
            if x > self.width():
                break
            
            if int(beat) % 4 == 0:
                # Measure line (4 beats per measure)
                painter.setPen(QPen(QColor("#404040"), 2))
            elif int(beat) % 2 == 0:
                # Half measure
                painter.setPen(QPen(QColor("#2a2a2a"), 1.5))
            else:
                # Beat
                painter.setPen(QPen(QColor("#1a1a1a"), 1))
            
            painter.drawLine(x, self.HEADER_HEIGHT, x, self.height())
            beat += 1
        
        # Horizontal lines for all pitches, with darker lines on C notes
        for pitch in range(self.TOTAL_PITCHES):
            y = self.pitch_to_pixel(pitch)
            note_in_octave = pitch % self.PITCHES_PER_OCTAVE
            
            if note_in_octave == 0:
                # C note (octave marker)
                painter.setPen(QPen(QColor("#303030"), 1.5))
            elif note_in_octave in [1, 3, 6, 8, 10]:
                # Black key row
                painter.setPen(QPen(QColor("#141414"), 1))
            else:
                # White key row
                painter.setPen(QPen(QColor("#1a1a1a"), 1))
            
            note_start_x = self.HEADER_WIDTH + self.NOTE_AREA_PADDING
            painter.drawLine(note_start_x, y, self.width(), y)
    
    def draw_key_separators(self, painter: QPainter) -> None:
        """Draw separators between white and black key areas (only in note area, not in piano sidebar)"""
        note_start_x = self.HEADER_WIDTH + self.NOTE_AREA_PADDING
        for pitch in range(self.TOTAL_PITCHES):
            # Draw separator BELOW black keys to visually isolate them
            note_in_octave = pitch % self.PITCHES_PER_OCTAVE
            if note_in_octave in [1, 3, 6, 8, 10]:  # Black keys: C#, D#, F#, G#, A#
                y = self.pitch_to_pixel(pitch) + self.PIXEL_PER_PITCH - 1
                painter.setPen(QPen(QColor("#1a1a1a"), 2))
                painter.drawLine(note_start_x, y, self.width(), y)
    
    def draw_notes(self, painter: QPainter) -> None:
        """Draw note rectangles from all visible layers"""
        # Set clipping region to prevent notes from drawing into piano sidebar
        note_start_x = self.HEADER_WIDTH + self.NOTE_AREA_PADDING
        clip_rect = QRect(note_start_x, self.HEADER_HEIGHT, self.width() - note_start_x, self.height() - self.HEADER_HEIGHT)
        painter.setClipRect(clip_rect)
        
        # Draw notes from all visible layers with lighter opacity for non-current layers
        for layer_name, layer_data in self.layers.items():
            if layer_data["visible"]:
                layer_opacity = 0.4 if layer_name != self.current_layer else 1.0
                for note in layer_data["notes"]:
                    is_selected = note in self.selected_notes and layer_name == self.current_layer
                    self.draw_note(painter, note, is_selected=is_selected, layer_color=layer_data["color"], layer_opacity=layer_opacity)
        
        # Draw ghost notes
        for ghost in self.ghost_notes:
            self.draw_note(painter, ghost, is_ghost=True)
        
        if self.dragging and self.drag_start_beat is not None:
            self.draw_drag_preview(painter)
        
        # Disable clipping for subsequent drawings
        painter.setClipRect(QRect())  # Clear clipping
    
    def draw_note(self, painter: QPainter, note: Note, is_ghost: bool = False, is_selected: bool = False, layer_color: str = "#9a9a9a", layer_opacity: float = 1.0) -> None:
        """Draw a single note with BeepBox-style clean appearance"""
        painter.save()
        painter.setOpacity(layer_opacity)
        x = self.beat_to_pixel(note.start_time)
        y = self.pitch_to_pixel(note.pitch)
        width = int(note.duration * self.PIXEL_PER_BEAT)
        height = self.PIXEL_PER_PITCH - 1
        
        if note.is_slide:
            self._draw_slide_note(painter, note, x, y, width, height, is_selected, layer_color)
        else:
            if self.viz_mode == "velocity":
                self._draw_velocity_note(painter, note, x, y, width, height, is_ghost, is_selected, layer_color)
            elif self.viz_mode == "gradient":
                self._draw_gradient_note(painter, note, x, y, width, height, is_ghost, is_selected, layer_color)
            elif self.viz_mode == "neon":
                self._draw_neon_note(painter, note, x, y, width, height, is_ghost, is_selected, layer_color)
            else:
                self._draw_standard_note(painter, note, x, y, width, height, is_ghost, is_selected, layer_color)

        if self.is_playing and note.contains_time(self.playback_position):
            self._draw_active_note_highlight(painter, x, y, width, height)

        painter.restore()
    
    def _draw_standard_note(self, painter: QPainter, note: Note, x: int, y: int, width: int, height: int, is_ghost: bool, is_selected: bool, layer_color: str) -> None:
        """Draw standard note style (BeepBox-inspired)"""
        if is_selected:
            color = QColor(layer_color)
            color = color.lighter(140)
            border_width = 2
        else:
            color = QColor(layer_color) if not is_ghost else QColor("#888888")
            border_width = 1
        
        # Draw the note rectangle
        painter.fillRect(x, y, width, height, QBrush(color))
        
        # Draw border
        painter.setPen(QPen(color.darker(150), border_width))
        painter.drawRect(x, y, width, height)
        
        # Optional: add highlight on top for depth (BeepBox style)
        highlight_color = color.lighter(120)
        highlight_color.setAlpha(100)
        painter.setPen(QPen(highlight_color, 1))
        painter.drawLine(x + 1, y + 1, x + width - 2, y + 1)

    def _draw_active_note_highlight(self, painter: QPainter, x: int, y: int, width: int, height: int) -> None:
        painter.setPen(QPen(QColor("#ffffff"), 2, Qt.PenStyle.SolidLine))
        painter.drawRect(x - 1, y - 1, width + 2, height + 2)
        painter.setPen(QPen(QColor("#f0f0f0"), 1, Qt.PenStyle.DotLine))
        painter.drawRect(x - 2, y - 2, width + 4, height + 4)

    def _draw_velocity_note(self, painter: QPainter, note: Note, x: int, y: int, width: int, height: int, is_ghost: bool, is_selected: bool, layer_color: str) -> None:
        """Draw note with velocity-based coloring (more subtle)"""
        velocity_ratio = note.velocity / 127.0
        
        # Create color based on velocity - use layer color with brightness variation
        base_color = QColor(layer_color)
        brightness_offset = int((velocity_ratio - 0.5) * 100)
        color = base_color.lighter(100 + brightness_offset) if brightness_offset > 0 else base_color.darker(100 - brightness_offset)
        
        if is_selected:
            color = color.lighter(120)
        
        painter.fillRect(x, y, width, height, QBrush(color))
        painter.setPen(QPen(color.darker(150), 2 if is_selected else 1))
        painter.drawRect(x, y, width, height)
    
    def _draw_gradient_note(self, painter: QPainter, note: Note, x: int, y: int, width: int, height: int, is_ghost: bool, is_selected: bool, layer_color: str) -> None:
        """Draw note with gradient fill (BeepBox-inspired)"""
        gradient = QLinearGradient(x, y, x, y + height)
        start_color = QColor(layer_color)
        end_color = QColor(layer_color).darker(130)
        
        if is_selected:
            start_color = start_color.lighter(120)
            end_color = end_color.lighter(120)
        
        gradient.setColorAt(0, start_color)
        gradient.setColorAt(1, end_color)
        
        painter.fillRect(x, y, width, height, QBrush(gradient))
        painter.setPen(QPen(end_color.darker(150), 2 if is_selected else 1))
        painter.drawRect(x, y, width, height)
    
    def _draw_neon_note(self, painter: QPainter, note: Note, x: int, y: int, width: int, height: int, is_ghost: bool, is_selected: bool, layer_color: str) -> None:
        """Draw note with neon glow effect"""
        color = QColor("#666666")  # Gray neon instead of green
        
        for i in range(3):
            glow_color = QColor(color)
            glow_color.setAlpha(int(80 - i * 25))
            pen = QPen(glow_color, 2 + i)
            painter.setPen(pen)
            painter.drawRect(x - i, y - i, width + 2 * i, height + 2 * i)
        
        painter.setPen(QPen(color, 2 if is_selected else 1))
        painter.fillRect(x, y, width, height, QBrush(color))
    
    def _draw_slide_note(self, painter: QPainter, note: Note, x: int, y: int, width: int, height: int, is_selected: bool, layer_color: str) -> None:
        """Draw slide note with special styling"""
        gradient = QLinearGradient(x, y, x, y + height)
        start_color = QColor(layer_color)
        end_color = QColor(layer_color).darker(120)
        
        if is_selected:
            start_color = start_color.lighter(120)
            end_color = end_color.lighter(120)
        
        gradient.setColorAt(0, start_color)
        gradient.setColorAt(1, end_color)
        
        painter.setPen(QPen(QColor(layer_color), 2 if is_selected else 1))
        painter.fillRect(x, y, width, height, QBrush(gradient))
        
        if note.slide_target is not None:
            target_y = self.pitch_to_pixel(note.slide_target)
            painter.setPen(QPen(QColor(layer_color), 2 if is_selected else 1))
            painter.drawLine(x + width, y, x + width, target_y)
    
    def draw_drag_preview(self, painter: QPainter) -> None:
        """Draw preview of note being created"""
        if self.drag_start_beat is None or self.drag_pitch is None:
            return
        
        x1 = self.beat_to_pixel(self.drag_start_beat)
        y = self.pitch_to_pixel(self.drag_pitch)
        
        current_beat = self.snap_beat(self.pixel_to_beat(self.mapFromGlobal(self.cursor().pos()).x()))
        x2 = self.beat_to_pixel(current_beat)
        
        width = x2 - x1
        
        # Handle dragging backwards - swap if needed
        if width < 0:
            x1, x2 = x2, x1
            width = abs(width)
        
        # Minimum width to show something visible
        if width < self.PIXEL_PER_BEAT // 4:
            width = self.PIXEL_PER_BEAT // 4
        
        height = self.PIXEL_PER_PITCH - 1
        
        # Draw the preview note
        painter.setPen(QPen(QColor("#9a9a9a"), 2))
        painter.fillRect(x1, y, width, height, QBrush(QColor("#9a9a9a")))
        painter.setPen(QPen(QColor("#666666"), 2))
        painter.drawRect(x1, y, width, height)
        
        # Draw endpoint markers
        painter.setPen(QPen(QColor("#cccccc"), 1.5))
        painter.drawLine(x1, y - 4, x1, y - 1)
        painter.drawLine(x1 + width, y - 4, x1 + width, y - 1)
    
    def draw_selection_rect(self, painter: QPainter) -> None:
        """Draw selection rectangle"""
        if not self.selection_start or not self.selecting_rect:
            return
        
        rect = QRect(self.selection_start, self.selecting_rect).normalized()
        painter.setPen(QPen(QColor("#cccccc"), 1, Qt.PenStyle.DashLine))
        painter.drawRect(rect)
        painter.fillRect(rect, QBrush(QColor(200, 200, 200, 28)))
    
    def draw_playback_indicator(self, painter: QPainter) -> None:
        """Draw the playback position line"""
        x = self.beat_to_pixel(self.playback_position)
        
        # Thin white line for clear playback indication
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawLine(x, self.HEADER_HEIGHT, x, self.height())
        
        # Red highlight behind the white line for playback focus
        painter.setPen(QPen(QColor("#e8e8e8"), 3))
        painter.drawLine(x, self.HEADER_HEIGHT, x, self.height())
        
        painter.fillRect(x - 3, 2, 6, self.HEADER_HEIGHT - 4, QBrush(QColor("#e8e8e8")))
    
    def export_midi(self) -> dict:
        """Export notes as JSON-compatible MIDI data with layers"""
        return {
            'layers': {
                layer_name: {
                    'notes': [
                        {
                            'pitch': n.pitch,
                            'start_time': n.start_time,
                            'duration': n.duration,
                            'velocity': n.velocity,
                            'is_slide': n.is_slide,
                            'slide_target': n.slide_target
                        }
                        for n in layer_data["notes"]
                    ],
                    'visible': layer_data["visible"],
                    'sample_name': layer_data["sample_name"],
                    'sample_path': layer_data.get("sample_path")
                }
                for layer_name, layer_data in self.layers.items()
            },
            'bpm': self.bpm,
            'scale': self.scale
        }
    
    def load_midi(self, data: dict) -> None:
        """Load notes from JSON-compatible MIDI data with layers"""
        self.bpm = data.get('bpm', 120)
        self.scale = data.get('scale', 'Major')
        
        # Clear all layers
        for layer_name in self.layers:
            self.layers[layer_name]["notes"].clear()
        
        # Load layer data if present
        if 'layers' in data:
            for layer_name, layer_data in data['layers'].items():
                if layer_name in self.layers:
                    self.layers[layer_name]["visible"] = layer_data.get("visible", True)
                    self.layers[layer_name]["sample_name"] = layer_data.get("sample_name", layer_name)
                    self.layers[layer_name]["sample_path"] = layer_data.get("sample_path")
                    
                    for note_data in layer_data.get('notes', []):
                        note = Note(
                            pitch=note_data['pitch'],
                            start_time=note_data['start_time'],
                            duration=note_data['duration'],
                            velocity=note_data.get('velocity', 100),
                            is_slide=note_data.get('is_slide', False),
                            slide_target=note_data.get('slide_target')
                        )
                        self.layers[layer_name]["notes"].append(note)
        
        # Update current layer notes
        self.notes = self.layers[self.current_layer]["notes"]
        self.update()
    
    def clear(self) -> None:
        """Clear all notes from all layers"""
        for layer_name in self.layers:
            self.layers[layer_name]["notes"].clear()
        self.notes.clear()
        self.selected_notes.clear()
        self.ghost_notes.clear()
        self.update()


class TerminalParser:
    """Command parser for terminal interface"""
    
    def __init__(self, audio_engine: AudioEngine, callback: Callable):
        self.audio_engine = audio_engine
        self.callback = callback
        
        self.commands = {
            'help': self.cmd_help,
            'clear_midi': self.cmd_clear_midi,
            'set_scale': self.cmd_set_scale,
            'set_bpm': self.cmd_set_bpm,
            'toggle_osc': self.cmd_toggle_osc,
            'export_midi': self.cmd_export_midi,
            'import_midi': self.cmd_import_midi,
            'import_drums': self.cmd_import_drums,
            'set_viz': self.cmd_set_viz,
            'snap': self.cmd_snap,
            'velocity': self.cmd_velocity,
            'grid': self.cmd_grid,
            'seek': self.cmd_seek,
        }
    
    def parse(self, command: str) -> dict:
        """Parse and execute command"""
        parts = command.strip().split()
        
        if not parts:
            return {'success': False, 'message': 'Empty command'}
        
        cmd = parts[0].lstrip('/')
        args = parts[1:]
        
        if cmd in self.commands:
            try:
                return self.commands[cmd](args)
            except Exception as e:
                return {'success': False, 'message': f'Error: {str(e)}'}
        
        return {'success': False, 'message': f'Unknown command: {cmd}'}
    
    def cmd_help(self, args) -> dict:
        """Show help"""
        help_text = """
        === CORE COMMANDS ===
        /help - Show this help
        /clear_midi - Clear all notes
        /set_scale <scale> - Set scale (Major, Minor, etc)
        /set_bpm <bpm> - Set tempo
        /toggle_osc - Toggle oscillators
        /set_viz <mode> - Set visualization (standard, velocity, gradient, neon)
        /export_midi <file> - Export to JSON
        /import_midi <file> - Import from JSON
        /import_drums <directory> - Import custom drum samples
        
        === EDITING ===
        /snap <on|off|1|0.5|0.25|0.125> - Snap-to-grid control
        /velocity <1-127> - Set velocity for new notes
        /grid <on|off> - Toggle grid visibility
        
        === TOOLS ===
        /seek - Launch audio visualizer
        
        === KEYBOARD SHORTCUTS ===
        Spacebar: Play/Stop | Shift+Drag: Slide | Right-Click: Delete
        Ctrl+A: Select All | Ctrl+D: Deselect | Delete: Remove Selected
        Middle-Click+Drag: Select Region | Drag Note: Move Note
        """
        return {'success': True, 'message': help_text}
    
    def cmd_clear_midi(self, args) -> dict:
        self.callback('clear_midi')
        return {'success': True, 'message': 'MIDI cleared'}
    
    def cmd_set_scale(self, args) -> dict:
        if not args:
            return {'success': False, 'message': 'Scale name required'}
        scale = args[0]
        self.callback('scale_changed', scale=scale)
        return {'success': True, 'message': f'Scale set to {scale}'}
    
    def cmd_set_bpm(self, args) -> dict:
        if not args:
            return {'success': False, 'message': 'BPM value required'}
        try:
            bpm = int(args[0])
            self.audio_engine.queue_command({'type': 'set_bpm', 'value': bpm})
            return {'success': True, 'message': f'BPM set to {bpm}'}
        except ValueError:
            return {'success': False, 'message': 'Invalid BPM value'}
    
    def cmd_toggle_osc(self, args) -> dict:
        # Toggle both oscillators
        self.audio_engine.osc1.enabled = not self.audio_engine.osc1.enabled
        self.audio_engine.osc2.enabled = not self.audio_engine.osc2.enabled
        status1 = 'ON' if self.audio_engine.osc1.enabled else 'OFF'
        status2 = 'ON' if self.audio_engine.osc2.enabled else 'OFF'
        
        # Launch the wavetable test window
        try:
            subprocess.Popen([sys.executable, 'testwavetable.py'], cwd=os.path.dirname(__file__))
        except Exception as e:
            return {'success': False, 'message': f'Failed to launch wavetable: {e}'}
        
        return {'success': True, 'message': f'OSC 1: {status1}, OSC 2: {status2} | Wavetable launched'}
    
    def cmd_set_viz(self, args) -> dict:
        if not args:
            return {'success': False, 'message': 'Mode required (standard, velocity, gradient, neon)'}
        mode = args[0]
        if mode not in ['standard', 'velocity', 'gradient', 'neon']:
            return {'success': False, 'message': 'Invalid mode'}
        self.callback('set_viz', mode=mode)
        return {'success': True, 'message': f'Visualization set to {mode}'}
    
    def cmd_snap(self, args) -> dict:
        if not args:
            return {'success': False, 'message': 'Usage: /snap <on|off|1|0.5|0.25|0.125>'}
        
        arg = args[0].lower()
        
        if arg == 'off' or arg == '0':
            self.callback('set_snap', enabled=False)
            return {'success': True, 'message': 'Snap-to-grid: OFF'}
        elif arg == 'on':
            self.callback('set_snap', enabled=True, snap_value=0.25)
            return {'success': True, 'message': 'Snap-to-grid: ON (1/16 beat)'}
        else:
            try:
                snap_value = float(arg)
                if snap_value <= 0:
                    return {'success': False, 'message': 'Snap value must be positive'}
                self.callback('set_snap', enabled=True, snap_value=snap_value)
                return {'success': True, 'message': f'Snap-to-grid: ON ({snap_value} beat)'}
            except ValueError:
                return {'success': False, 'message': 'Invalid snap value'}
    
    def cmd_velocity(self, args) -> dict:
        """Set velocity for new notes"""
        if not args:
            return {'success': False, 'message': 'Velocity required (1-127)'}
        try:
            velocity = int(args[0])
            if velocity < 1 or velocity > 127:
                return {'success': False, 'message': 'Velocity must be 1-127'}
            self.callback('set_velocity', velocity=velocity)
            return {'success': True, 'message': f'Velocity set to {velocity}'}
        except ValueError:
            return {'success': False, 'message': 'Invalid velocity value'}
    
    def cmd_grid(self, args) -> dict:
        """Toggle grid visibility"""
        if not args:
            return {'success': False, 'message': 'Usage: /grid <on|off>'}
        
        if args[0].lower() == 'on':
            self.callback('set_grid', enabled=True)
            return {'success': True, 'message': 'Grid: ON'}
        else:
            self.callback('set_grid', enabled=False)
            return {'success': True, 'message': 'Grid: OFF'}
    
    def cmd_seek(self, args) -> dict:
        """Launch audio visualizer"""
        self.callback('launch_seek')
        return {'success': True, 'message': 'Launching audio visualizer...'}
    
    def cmd_export_midi(self, args) -> dict:
        filename = args[0] if args else 'export.json'
        self.callback('export_midi', filename=filename)
        return {'success': True, 'message': f'Exported to {filename}'}
    
    def cmd_import_midi(self, args) -> dict:
        filename = args[0] if args else 'import.json'
        self.callback('import_midi', filename=filename)
        return {'success': True, 'message': f'Imported from {filename}'}
    
    def cmd_import_drums(self, args) -> dict:
        if not args:
            return {'success': False, 'message': 'Usage: /import_drums <directory>'}
        directory = args[0]
        self.callback('import_drums', directory=directory)
        return {'success': True, 'message': f'Imported drums from {directory}'}


class TerminalGUI(QWidget):
    """Terminal interface"""
    
    def __init__(self, audio_engine: AudioEngine, parent=None):
        super().__init__(parent)
        self.audio_engine = audio_engine
        self.parser: Optional[TerminalParser] = None
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Courier New", 9))
        self.log.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #e0e0e0;
                border: none;
            }
        """)
        layout.addWidget(self.log, 1)
        
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(6, 6, 6, 6)
        input_layout.setSpacing(6)
        
        prompt = QLabel("C:>")
        prompt.setFont(QFont("Courier New", 9))
        prompt.setStyleSheet("color: #e0e0e0; min-width: 30px;")
        input_layout.addWidget(prompt)
        
        self.input_field = QLineEdit()
        self.input_field.setFont(QFont("Courier New", 9))
        self.input_field.returnPressed.connect(self.on_command_entered)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #000000;
                color: #e0e0e0;
                border: none;
                border-top: 1px solid #2a2a2a;
            }
        """)
        input_layout.addWidget(self.input_field)
        
        input_widget = QWidget()
        input_widget.setLayout(input_layout)
        input_widget.setStyleSheet("background-color: #000000; border-top: 1px solid #2a2a2a;")
        layout.addWidget(input_widget)
        
        self.setLayout(layout)
    
    def set_parser(self, parser: TerminalParser) -> None:
        self.parser = parser
    
    def add_log_line(self, text: str, line_type: str = 'info') -> None:
        """Add line to terminal output"""
        colors = {
            'success': '#d0d0d0',
            'error': '#909090',
            'warning': '#b8b8b8',
            'info': '#e0e0e0'
        }
        color = colors.get(line_type, '#e0e0e0')
        self.log.append(f'<span style="color: {color};">{text}</span>')
    
    def on_command_entered(self) -> None:
        """Handle command submission"""
        command = self.input_field.text().strip()
        self.input_field.clear()
        
        if not command:
            return
        
        self.add_log_line(f"> {command}", 'info')
        
        if self.parser:
            result = self.parser.parse(command)
            message_type = 'success' if result['success'] else 'error'
            self.add_log_line(result['message'], message_type)


class NillApplication(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nill 2 | FL Studio-like DAW [Enhanced]")
        self.setGeometry(50, 50, 1600, 900)
        
        # Initialize components
        self.audio_engine = AudioEngine()
        self.audio_engine.start()
        
        self.piano_roll = PianoRoll()
        self.terminal_gui = TerminalGUI(self.audio_engine)

        self.poly_synth = PolyphonicSynth() if PolyphonicSynth else None
        self.poly_playing_pitches = set()
        self.poly_stream = None
        if self.poly_synth:
            self._start_poly_audio()
        else:
            print('Warning: PolyphonicSynth import failed. Live synth audio unavailable.')
        
        self.parser = TerminalParser(self.audio_engine, self.handle_parser_command)
        self.terminal_gui.set_parser(self.parser)
        
        self.init_ui()
        self.apply_styles()

        self.last_playback_notes = []

        # Spacebar playback event filter
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

        # Playback timer with precise timing
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_playback)
        self.playback_timer.start(20)  # More frequent updates for smoother timing
        
        # High-precision elapsed timer for accurate playback timing
        self.playback_elapsed = QElapsedTimer()
        self.last_elapsed_time = 0  # Track last elapsed time for delta calculation
        self.last_active_pitches = set()  # Track last active pitches to prevent jitter
        self.playback_start_time = 0  # Use perf_counter for ultra-precise timing
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)
    
    def init_ui(self) -> None:
        """Setup main UI layout"""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Piano Roll
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("PIANO ROLL | Left-Click: Draw | Right-Click: Delete | Middle-Drag: Select | Drag Note: Move | Spacebar / Play button")
        title.setFont(QFont("Courier New", 8))
        title.setStyleSheet("""
            background-color: #1a1a1a;
            color: #e0e0e0;
            border-bottom: 1px solid #2a2a2a;
            padding: 6px;
        """)
        left_layout.addWidget(title)
        left_layout.addWidget(self.piano_roll, 1)
        
        splitter.addWidget(left_panel)
        
        # Right: Terminal + Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        title2 = QLabel("LAYERS & CONTROLS")
        title2.setFont(QFont("Courier New", 9))
        title2.setStyleSheet("""
            background-color: #1a1a1a;
            color: #e0e0e0;
            border-bottom: 1px solid #2a2a2a;
            padding: 6px;
        """)
        right_layout.addWidget(title2)
        
        # Control panel
        ctrl_panel = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(6, 6, 6, 6)
        ctrl_layout.setSpacing(6)
        
        # Layer buttons
        layer_label = QLabel("LAYER (SAMPLES):")
        layer_label.setFont(QFont("Courier New", 9))
        layer_label.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        ctrl_layout.addWidget(layer_label)
        
        self.layer_buttons = {}
        for layer_name in ["drums", "bass", "melody"]:
            btn = QPushButton(layer_name.upper())
            btn.setFont(QFont("Courier New", 8))
            btn.setCheckable(True)
            if layer_name == "drums":
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, l=layer_name: self.on_layer_selected(l))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2a2a2a;
                    color: #e0e0e0;
                    border: 1px solid #444444;
                    padding: 6px;
                    border-radius: 2px;
                }
                QPushButton:checked {
                    background-color: #444444;
                    border: 2px solid #888888;
                }
                QPushButton:hover {
                    background-color: #333333;
                }
            """)
            ctrl_layout.addWidget(btn)
            self.layer_buttons[layer_name] = btn
        
        ctrl_layout.addSpacing(10)

        # Play / Stop button
        self.play_button = QPushButton("Play")
        self.play_button.setFont(QFont("Courier New", 8))
        self.play_button.setShortcut(QKeySequence("Space"))
        self.play_button.setAutoDefault(False)
        self.play_button.setDefault(False)
        self.play_button.clicked.connect(self.on_play_button_clicked)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #444444;
                padding: 6px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """)
        ctrl_layout.addWidget(self.play_button)

        ctrl_layout.addSpacing(6)

        # Drum sample import
        self.import_drum_button = QPushButton("Import Drum Sample")
        self.import_drum_button.setFont(QFont("Courier New", 8))
        self.import_drum_button.clicked.connect(self.import_drum_sample)
        self.import_drum_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #444444;
                padding: 6px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """)
        ctrl_layout.addWidget(self.import_drum_button)

        self.drum_sample_label = QLabel("Drum sample: Default")
        self.drum_sample_label.setFont(QFont("Courier New", 8))
        self.drum_sample_label.setStyleSheet("color: #cccccc;")
        ctrl_layout.addWidget(self.drum_sample_label)

        ctrl_layout.addSpacing(10)
        
        # Velocity control
        vel_label = QLabel("Velocity:")
        vel_label.setFont(QFont("Courier New", 9))
        vel_label.setStyleSheet("color: #e0e0e0;")
        ctrl_layout.addWidget(vel_label)
        
        self.velocity_spin = QSpinBox()
        self.velocity_spin.setMinimum(1)
        self.velocity_spin.setMaximum(127)
        self.velocity_spin.setValue(100)
        self.velocity_spin.valueChanged.connect(self.on_velocity_changed)
        self.velocity_spin.setStyleSheet("""
            QSpinBox {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #2a2a2a;
            }
        """)
        ctrl_layout.addWidget(self.velocity_spin)
        
        ctrl_layout.addSpacing(10)
        
        # Visualization mode
        viz_label = QLabel("Visualization:")
        viz_label.setFont(QFont("Courier New", 9))
        viz_label.setStyleSheet("color: #e0e0e0;")
        ctrl_layout.addWidget(viz_label)
        
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(["standard", "velocity", "gradient", "neon"])
        self.viz_combo.currentTextChanged.connect(self.on_viz_changed)
        self.viz_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #2a2a2a;
            }
        """)
        ctrl_layout.addWidget(self.viz_combo)
        
        ctrl_layout.addStretch()
        
        ctrl_panel.setStyleSheet("background-color: #0a0a0a;")
        right_layout.addWidget(ctrl_panel)
        
        right_layout.addWidget(self.terminal_gui, 1)
        
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        self.update_sample_display()
    
    def apply_styles(self) -> None:
        """Apply global stylesheet"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a0a;
                border: 1px solid #2a2a2a;
            }
            QWidget {
                background-color: #0a0a0a;
            }
            QLabel {
                font-family: 'Courier New', monospace;
            }
        """)
    
    def on_velocity_changed(self, value: int) -> None:
        """Handle velocity change"""
        self.piano_roll.set_velocity(value)
    
    def import_drum_sample(self) -> None:
        """Open a file dialog to load a drum sample"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Drum Sample",
            "",
            "Audio Files (*.wav *.mp3 *.aiff *.flac);;All Files (*)"
        )
        if not filename:
            return

        sample_name = os.path.basename(filename)
        self.piano_roll.layers["drums"]["sample_name"] = sample_name
        self.piano_roll.layers["drums"]["sample_path"] = filename
        self.update_sample_display()
        self.terminal_gui.add_log_line(f"Loaded drum sample: {sample_name}", 'success')

    def update_sample_display(self) -> None:
        """Update displayed drum sample info"""
        sample_name = self.piano_roll.layers["drums"].get("sample_name", "Default")
        sample_path = self.piano_roll.layers["drums"].get("sample_path")
        display_name = sample_name if sample_path else sample_name
        self.drum_sample_label.setText(f"Drum sample: {display_name}")

    def on_viz_changed(self, text: str) -> None:
        """Handle visualization mode change"""
        self.piano_roll.set_visualization_mode(text)

    def on_play_button_clicked(self) -> None:
        """Handle playback button clicks."""
        was_playing = self.piano_roll.is_playing
        self.piano_roll.toggle_playback()
        
        # Handle timing when playback state changes
        if self.piano_roll.is_playing and not was_playing:
            # Starting playback
            self.playback_elapsed.start()
            self.last_elapsed_time = 0
            self.playback_start_time = time.perf_counter()
        elif not self.piano_roll.is_playing and was_playing:
            # Stopping playback
            self.last_elapsed_time = 0
            self.playback_start_time = 0
            self.last_active_pitches.clear()
        
        self.play_button.setText("Pause" if self.piano_roll.is_playing else "Play")

    def on_space_pressed(self) -> None:
        """Shortcut fallback that toggles playback."""
        if self.terminal_gui.input_field.hasFocus():
            return
        self.on_play_button_clicked()

    def play_note_sound(self, note: Note) -> None:
        """Play a short tone for a single note (cross-platform)."""
        freq = int(440.0 * (2 ** ((note.pitch - 69) / 12.0)))
        duration_ms = max(30, int((note.duration * 60000.0) / self.piano_roll.bpm))
        if winsound is not None:
            threading.Thread(target=winsound.Beep, args=(freq, duration_ms), daemon=True).start()
        else:
            threading.Thread(target=_play_beep, args=(freq, duration_ms), daemon=True).start()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            if not self.terminal_gui.input_field.hasFocus():
                self.on_play_button_clicked()
                return True
        return super().eventFilter(obj, event)
    
    def on_layer_selected(self, layer: str) -> None:
        """Handle layer selection"""
        # Update button states
        for btn_name, btn in self.layer_buttons.items():
            btn.setChecked(btn_name == layer)
        
        # Switch layer
        self.piano_roll.set_current_layer(layer)
        self.import_drum_button.setEnabled(layer == "drums")
        self.update_sample_display()
        self.terminal_gui.add_log_line(f"Layer switched to: {layer.upper()}", 'success')
    
    def handle_parser_command(self, command: str, *args, **kwargs) -> None:
        """Handle commands from terminal parser"""
        if command == 'clear_midi':
            self.piano_roll.clear()
            self.terminal_gui.add_log_line("Piano roll cleared", 'success')
        
        elif command == 'scale_changed':
            scale = kwargs.get('scale', 'Major')
            self.piano_roll.scale = scale
            self.terminal_gui.add_log_line(f"Scale set to {scale}", 'success')
        
        elif command == 'export_midi':
            filename = kwargs.get('filename', 'export.json')
            data = self.piano_roll.export_midi()
            try:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                self.terminal_gui.add_log_line(f"Exported to {filename}", 'success')
            except Exception as e:
                self.terminal_gui.add_log_line(f"Export failed: {str(e)}", 'error')
        
        elif command == 'import_midi':
            filename = kwargs.get('filename', 'import.json')
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                self.piano_roll.load_midi(data)
                self.terminal_gui.add_log_line(f"Imported from {filename}", 'success')
            except Exception as e:
                self.terminal_gui.add_log_line(f"Import failed: {str(e)}", 'error')
        
        elif command == 'import_drums':
            directory = kwargs['directory']  # type: str
            try:
                import os
                drum_map = {
                    'kick.wav': 36,
                    'snare.wav': 38,
                    'hihat.wav': 42,
                    'crash.wav': 49,
                    'tom.wav': 41
                }
                try:
                    import soundfile as sf  # type: ignore[import]
                except ImportError:
                    sf = None
                    self.terminal_gui.add_log_line("soundfile not available for drum import", 'error')
                    return
                for file in os.listdir(directory):
                    if file.endswith('.wav') and file in drum_map:
                        path = os.path.join(directory, file)
                        pitch = drum_map[file]
                        # Load WAV file
                        try:
                            data, samplerate = sf.read(path, dtype='float32')
                            if len(data.shape) > 1:
                                data = data[:, 0]  # mono
                            self.piano_roll.default_drums[pitch] = data
                        except Exception as e:
                            self.terminal_gui.add_log_line(f"Failed to load {file}: {str(e)}", 'error')
                self.terminal_gui.add_log_line(f"Drums imported from {directory}", 'success')
            except Exception as e:
                self.terminal_gui.add_log_line(f"Drum import failed: {str(e)}", 'error')
        
        elif command == 'set_viz':
            mode = kwargs.get('mode', 'standard')
            self.piano_roll.set_visualization_mode(mode)
            self.viz_combo.blockSignals(True)
            self.viz_combo.setCurrentText(mode)
            self.viz_combo.blockSignals(False)
            self.terminal_gui.add_log_line(f"Visualization set to {mode}", 'success')
        
        elif command == 'set_snap':
            enabled = kwargs.get('enabled', True)
            snap_value = kwargs.get('snap_value', 0.25)
            self.piano_roll.set_snap(enabled, snap_value)
            if enabled:
                self.terminal_gui.add_log_line(f"Snap-to-grid enabled (step: {snap_value})", 'success')
            else:
                self.terminal_gui.add_log_line("Snap-to-grid disabled", 'success')
        
        elif command == 'set_velocity':
            velocity = kwargs.get('velocity', 100)
            self.piano_roll.set_velocity(velocity)
            self.velocity_spin.blockSignals(True)
            self.velocity_spin.setValue(velocity)
            self.velocity_spin.blockSignals(False)
            self.terminal_gui.add_log_line(f"Velocity set to {velocity}", 'success')
        
        elif command == 'set_grid':
            enabled = kwargs.get('enabled', True)
            self.piano_roll.set_grid_visibility(enabled)
            self.terminal_gui.add_log_line(f"Grid: {'ON' if enabled else 'OFF'}", 'success')
        
        elif command == 'launch_seek':
            self.launch_audio_visualizer()
    
    def launch_audio_visualizer(self) -> None:
        """Launch the audio visualizer in a separate process"""
        import subprocess
        import sys
        from pathlib import Path
        try:
            # Launch the visualizer as a separate process
            subprocess.Popen([sys.executable, 'testvisualizer.py'], 
                           cwd=Path.cwd())
            self.terminal_gui.add_log_line("Audio visualizer launched", 'success')
        except Exception as e:
            self.terminal_gui.add_log_line(f"Failed to launch visualizer: {str(e)}", 'error')
    
    def update_playback(self) -> None:
        """Update playback position and active notes with precise timing"""
        if self.piano_roll.is_playing:
            # Use perf_counter for ultra-precise timing
            current_time = time.perf_counter()
            if hasattr(self, 'playback_start_time') and self.playback_start_time > 0:
                elapsed_seconds = current_time - self.playback_start_time
                beats_per_second = self.piano_roll.bpm / 60.0
                self.piano_roll.playback_position = elapsed_seconds * beats_per_second
            else:
                # Fallback to elapsed timer
                if hasattr(self, 'playback_elapsed') and self.playback_elapsed.isValid():
                    current_elapsed = self.playback_elapsed.elapsed()
                    delta_ms = current_elapsed - self.last_elapsed_time
                    self.last_elapsed_time = current_elapsed
                    
                    beats_per_second = self.piano_roll.bpm / 60.0
                    advance_beats = (delta_ms / 1000.0) * beats_per_second
                    self.piano_roll.playback_position += advance_beats
                else:
                    # Final fallback
                    beats_per_second = self.piano_roll.bpm / 60.0
                    advance_beats = beats_per_second * 0.02  # 20ms interval
                    self.piano_roll.playback_position += advance_beats
            
            max_beat = max([n.start_time + n.duration for n in self.piano_roll.notes], default=16)
            if self.piano_roll.playback_position > max_beat:
                self.piano_roll.playback_position = 0.0
                self.playback_start_time = time.perf_counter()  # Reset for loop
                self.last_active_pitches.clear()  # Clear tracking for loop
            
            active_notes = self.piano_roll.get_notes_at_beat(self.piano_roll.playback_position)
            active_pitches = {note.pitch for note in active_notes}

            # Ensure poly synth audio stream is active when playing
            if self.poly_synth and (not self.poly_stream or not hasattr(self.poly_stream, 'active') or not self.poly_stream.active):
                self._start_poly_audio()

            if self.poly_synth:
                # Only trigger note events if pitches actually changed
                if active_pitches != self.last_active_pitches:
                    added = active_pitches - self.last_active_pitches
                    removed = self.last_active_pitches - active_pitches
                    
                    if self.piano_roll.current_layer == "drums":
                        # Play drum samples
                        for pitch in added:
                            if pitch in self.piano_roll.default_drums:
                                if sd:
                                    sd.play(self.piano_roll.default_drums[pitch], samplerate=44100)
                    else:
                        # Synth notes
                        for pitch in added:
                            self.poly_synth.note_on(pitch)
                        for pitch in removed:
                            self.poly_synth.note_off(pitch)
                    self.last_active_pitches = active_pitches.copy()
                    self.poly_playing_pitches = active_pitches.copy()  # Keep in sync
        else:
            if self.poly_synth:
                # Turn off all currently playing notes
                for pitch in list(self.last_active_pitches):
                    self.poly_synth.note_off(pitch)
                self.poly_playing_pitches.clear()
                self.last_active_pitches.clear()  # Clear tracking when stopped
            
            # Stop the poly synth audio stream to prevent endless playing
            if self.poly_stream and hasattr(self.poly_stream, 'active') and self.poly_stream.active:
                self.poly_stream.stop()
            
            # Reset timing when playback stops
            self.last_elapsed_time = 0
            self.playback_start_time = 0

        if hasattr(self, 'play_button'):
            self.play_button.setText("Pause" if self.piano_roll.is_playing else "Play")
        self.piano_roll.update()
    
    def update_status(self) -> None:
        """Update status from audio engine"""
        try:
            status = self.audio_engine.status_queue.get_nowait()
        except queue.Empty:
            pass

    def _start_poly_audio(self) -> None:
        if not sd or not self.poly_synth:
            return
        try:
            self.poly_stream = sd.OutputStream(
                channels=1,
                samplerate=self.poly_synth.sample_rate,
                blocksize=self.poly_synth.buffer_size,
                callback=self._poly_audio_callback
            )
            self.poly_stream.start()
        except Exception as e:
            self.terminal_gui.add_log_line(f'Poly synth audio failed: {e}', 'error')

    def _poly_audio_callback(self, outdata, frames, time_info, status):
        if status:
            return
        assert self.poly_synth is not None
        buffer = self.poly_synth.generate_audio(frames)
        outdata[:, 0] = buffer

    def closeEvent(self, event) -> None:
        """Cleanup on exit"""
        self.audio_engine.stop()
        event.accept()


def main() -> None:
    """Entry point"""
    print("=" * 60)
    print("Nill | Terminal Wavetable DAW")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    window = NillApplication()
    window.show()
    
    window.terminal_gui.add_log_line("Nill Initialized", 'success')
    window.terminal_gui.add_log_line("Audio Engine: OPERATIONAL", 'success')
    window.terminal_gui.add_log_line("Piano Roll Ready with FL Studio Features", 'success')
    window.terminal_gui.add_log_line("Type /help for commands", 'info')
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
