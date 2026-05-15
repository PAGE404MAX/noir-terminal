    #!/usr/bin/env python3
"""
Nill | BeepBox-inspired Terminal/ GUI DAW
Fixed playback (proper note-off, voice reset) + simpler chiptune synth like BeepBox.
"""

import sys
import os
import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
import queue
import json
import time
from pathlib import Path

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QLabel, QPushButton, QSplitter, QMessageBox,
    QComboBox, QSpinBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QElapsedTimer
from PySide6.QtGui import QFont, QColor, QPainter, QBrush, QPen, QLinearGradient


# ==================== IMPROVED BEEPBOX-LIKE SYNTH ====================
class ChiptuneSynth:
    """Simple, BeepBox-inspired polyphonic chiptune synth.
    Supports square, triangle, saw, noise with quick envelopes.
    Fixed voice management to prevent dragging notes.
    """
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self._voices: Dict[int, dict] = {}  # pitch -> voice state
        self._lock = threading.Lock()
        
        # BeepBox style defaults - very snappy
        self.attack = 0.002
        self.decay = 0.08
        self.sustain = 0.6
        self.release = 0.15
        self.gain = 0.35
        
        self.waveform = "square"  # square, triangle, saw, noise (global for simplicity)
        self.pulse_width = 0.5    # for square wave

    @staticmethod
    def midi_to_freq(pitch: int) -> float:
        return 440.0 * (2.0 ** ((pitch - 69) / 12.0))

    def note_on(self, pitch: int, velocity: float = 1.0) -> None:
        with self._lock:
            # Critical fix: if voice exists, force reset it (prevents dragging last note)
            if pitch in self._voices:
                self._voices[pitch]["state"] = "attack"
                self._voices[pitch]["env"] = 0.0
                self._voices[pitch]["phase"] = 0.0
                self._voices[pitch]["velocity"] = velocity
                return
            
            self._voices[pitch] = {
                "state": "attack",
                "env": 0.0,
                "phase": 0.0,
                "freq": self.midi_to_freq(pitch),
                "velocity": velocity,
                "released_at_env": 0.0,
            }

    def note_off(self, pitch: int) -> None:
        with self._lock:
            v = self._voices.get(pitch)
            if v and v["state"] != "release":
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
        r_step = 1.0 / max(1, int(self.release * sr))

        with self._lock:
            dead = []
            for pitch, v in list(self._voices.items()):
                phase = v["phase"]
                freq = v["freq"]
                env = v.get("env", 0.0)
                state = v["state"]
                vel = v.get("velocity", 1.0)
                released = v.get("released_at_env", env)

                phase_inc = 2.0 * np.pi * freq * dt
                idx = np.arange(frames)
                phases = (phase + phase_inc * idx) % (2.0 * np.pi)

                # BeepBox-style waveforms
                if self.waveform == "square":
                    wave = np.sign(np.sin(phases)) * (1 if np.random.random() > self.pulse_width else -1)
                elif self.waveform == "triangle":
                    wave = (2 / np.pi) * np.arcsin(np.sin(phases))
                elif self.waveform == "saw":
                    wave = (phases / np.pi) - 1.0
                elif self.waveform == "noise":
                    wave = np.random.uniform(-1.0, 1.0, frames)
                else:
                    wave = np.sin(phases)  # fallback sine

                # Per-sample envelope (cleaner version)
                envs = np.zeros(frames, dtype=np.float32)
                current_env = env
                current_state = state

                for i in range(frames):
                    if current_state == "attack":
                        current_env += a_step
                        if current_env >= 1.0:
                            current_env = 1.0
                            current_state = "decay"
                    elif current_state == "decay":
                        current_env -= d_step
                        if current_env <= self.sustain:
                            current_env = self.sustain
                            current_state = "sustain"
                    elif current_state == "sustain":
                        current_env = self.sustain
                    elif current_state == "release":
                        current_env -= r_step
                        if current_env <= 0.0:
                            current_env = 0.0
                            current_state = "dead"

                    envs[i] = current_env

                v["env"] = float(current_env)
                v["state"] = current_state
                v["phase"] = float((phase + phase_inc * frames) % (2.0 * np.pi))

                out += wave * envs * vel

                if current_state == "dead" or (current_state == "release" and current_env <= 0.01):
                    dead.append(pitch)

            for p in dead:
                self._voices.pop(p, None)

        # Soft limiting + gain (keeps it loud but clean like BeepBox)
        out = np.tanh(out * self.gain * 1.8)
        return out.astype(np.float32)


# Rest of the application (simplified PianoRoll for better compatibility with new synth)
# We keep most of your original UI but integrate the new synth and fix playback logic.

@dataclass
class Note:
    pitch: int
    start_time: float
    duration: float
    velocity: int = 100


class PianoRoll(QWidget):
    """Simplified for better BeepBox-like playback"""
    note_added = Signal(Note)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.notes: List[Note] = []
        self.is_playing = False
        self.playback_position = 0.0
        self.bpm = 140  # BeepBox default is around here
        
        self.PIXEL_PER_BEAT = 35
        self.PIXEL_PER_PITCH = 14
        self.OCTAVES = 5
        self.TOTAL_PITCHES = self.OCTAVES * 12
        
        self.setMinimumSize(900, 500)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))
        
        # Draw grid (BeepBox style - high contrast)
        for beat in range(33):  # generous pattern length
            x = 60 + beat * self.PIXEL_PER_BEAT
            if x > self.width(): break
            color = QColor("#444444") if beat % 4 == 0 else QColor("#222222")
            painter.setPen(QPen(color, 1))
            painter.drawLine(x, 30, x, self.height())
        
        for p in range(self.TOTAL_PITCHES):
            y = 30 + p * self.PIXEL_PER_PITCH
            if y > self.height(): break
            painter.setPen(QPen(QColor("#333333"), 1))
            painter.drawLine(60, y, self.width(), y)
            
            # Piano key labels
            if p % 12 == 0:
                painter.setPen(QPen(QColor("#aaaaaa")))
                painter.drawText(5, y + 12, self.note_names[0] + str(5 - p//12))
        
        # Draw notes (bright colors like BeepBox)
        for note in self.notes:
            x = 60 + int(note.start_time * self.PIXEL_PER_BEAT)
            y = 30 + (self.TOTAL_PITCHES - 1 - note.pitch) * self.PIXEL_PER_PITCH
            w = max(8, int(note.duration * self.PIXEL_PER_BEAT))
            h = self.PIXEL_PER_PITCH - 2
            
            color = QColor("#00ffcc") if note.pitch % 12 in [0,4,7] else QColor("#ff00aa")
            painter.fillRect(x, y, w, h, QBrush(color))
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawRect(x, y, w, h)
        
        if self.is_playing:
            x = 60 + int(self.playback_position * self.PIXEL_PER_BEAT)
            painter.setPen(QPen(QColor("#ffff00"), 3))
            painter.drawLine(x, 30, x, self.height())
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            beat = max(0, (event.position().x() - 60) // self.PIXEL_PER_BEAT)
            pitch = self.TOTAL_PITCHES - 1 - int((event.position().y() - 30) // self.PIXEL_PER_PITCH)
            pitch = max(0, min(self.TOTAL_PITCHES-1, pitch))
            
            # Toggle note (more BeepBox-like)
            for i, n in enumerate(self.notes):
                if abs(n.start_time - beat) < 0.6 and n.pitch == pitch:
                    self.notes.pop(i)
                    self.update()
                    return
            
            self.notes.append(Note(pitch=pitch, start_time=float(beat), duration=1.0))
            self.update()
    
    def toggle_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.playback_position = 0.0
        self.update()
    
    def get_notes_at_beat(self, beat: float) -> List[Note]:
        return [n for n in self.notes if n.start_time <= beat < n.start_time + n.duration]


class NillBeepBox(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nill | BeepBox Edition (Fixed Playback)")
        self.setGeometry(100, 100, 1200, 700)
        
        self.synth = ChiptuneSynth()
        self.stream = None
        self._start_audio()
        
        self.piano_roll = PianoRoll()
        self.init_ui()
        
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_playback)
        self.playback_timer.start(16)  # ~60fps
        
        self.last_active = set()
    
    def _start_audio(self):
        if not sd:
            print("sounddevice not available")
            return
        try:
            self.stream = sd.OutputStream(
                samplerate=self.synth.sample_rate,
                channels=1,
                blocksize=self.synth.buffer_size,
                callback=self.audio_callback
            )
            self.stream.start()
            print("Chiptune synth started successfully")
        except Exception as e:
            print("Audio startup failed:", e)
    
    def audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(status)
        buffer = self.synth.generate_audio(frames)
        outdata[:, 0] = buffer
        outdata[:, 0] = np.clip(outdata[:, 0], -0.95, 0.95)
    
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Controls
        ctrl = QWidget()
        ctrl.setFixedWidth(280)
        cl = QVBoxLayout(ctrl)
        
        title = QLabel("NILL - BEEPBOX MODE")
        title.setFont(QFont("Courier New", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #00ffcc; padding: 10px;")
        cl.addWidget(title)
        
        self.wave_combo = QComboBox()
        self.wave_combo.addItems(["square", "triangle", "saw", "noise"])
        self.wave_combo.currentTextChanged.connect(self.change_waveform)
        cl.addWidget(QLabel("Waveform:"))
        cl.addWidget(self.wave_combo)
        
        btn_play = QPushButton("PLAY / STOP (SPACE)")
        btn_play.clicked.connect(self.piano_roll.toggle_playback)
        btn_play.setStyleSheet("padding: 12px; font-size: 16px; background: #222;")
        cl.addWidget(btn_play)
        
        cl.addWidget(QLabel("Click grid to toggle notes.\nLast note drag fixed."))
        
        cl.addStretch()
        layout.addWidget(ctrl)
        layout.addWidget(self.piano_roll, 1)
        
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0a0a0a; color: #ddd; }
            QPushButton { background: #1a1a1a; border: 2px solid #00ffcc; padding: 8px; }
        """)
    
    def change_waveform(self, wf: str):
        self.synth.waveform = wf
        print(f"Waveform changed to {wf}")
    
    def update_playback(self):
        if not self.piano_roll.is_playing:
            # Ensure all notes are off
            if self.last_active:
                for p in self.last_active:
                    self.synth.note_off(p)
                self.last_active.clear()
            return
        
        # Advance playback
        beats_per_sec = self.piano_roll.bpm / 60.0
        self.piano_roll.playback_position += beats_per_sec * (16 / 1000.0)  # 16ms tick
        
        if self.piano_roll.playback_position > 32:  # loop at 32 beats
            self.piano_roll.playback_position = 0.0
        
        current_notes = self.piano_roll.get_notes_at_beat(self.piano_roll.playback_position)
        current_pitches = {n.pitch for n in current_notes}
        
        # Fixed note management
        to_off = self.last_active - current_pitches
        to_on = current_pitches - self.last_active
        
        for p in to_off:
            self.synth.note_off(p)
        for p in to_on:
            vel = 0.8 if any(n.pitch == p and n.velocity > 90 for n in current_notes) else 0.6
            self.synth.note_on(p, vel)
        
        self.last_active = current_pitches.copy()
        self.piano_roll.update()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.piano_roll.toggle_playback()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if hasattr(self.synth, '_voices'):
            self.synth.all_notes_off()
        event.accept()


def main():
    print("Nill - BeepBox Edition")
    print("Fixed: Note dragging / stuck last note")
    print("New synth: square/triangle/saw/noise with proper voice reset")
    print("Controls: Click grid to place notes. Space = play. Change waveform in sidebar.\n")
    
    app = QApplication(sys.argv)
    window = NillBeepBox()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
