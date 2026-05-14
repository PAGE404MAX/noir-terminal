"""
SERUM 2 | TERMINAL EDITION — Python/Tkinter
FOCUS: LEFT PANEL (OSC 1 & OSC 2) - FULLY FUNCTIONAL

FEATURES (THIS VERSION):
1. OSC 1-2 - Waveform, Detune, Level, Phase (All Working)
2. Working Scrollbars (Left & Right Panels)
3. Double-Click Knob Reset
4. Visual Piano Keyboard
5. M/N Waveform Cycling (Global Override)
6. Thread-safe Audio Engine
"""

import tkinter as tk
from tkinter import Canvas
import math
import numpy as np
from typing import Dict, List, Optional, Callable
import threading
import time
import sounddevice as sd

# --- Terminal Greyscale Palette ---
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
        self.active_notes: Dict[int, Dict] = {}
        self.sample_rate = SR
        self.buffer_size = BUFFER_SIZE
        self._lock = threading.Lock()
        self.last_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        
        # Global waveform override (M/N cycling)
        self.waveform_type = 'sine'
        self.use_global_waveform = False
        
        # OSC 1 Parameters
        self.osc1_wave = 'sine'
        self.osc1_detune = 0.0
        self.osc1_level = 1.0
        self.osc1_phase = 0.0
        
        # OSC 2 Parameters
        self.osc2_wave = 'saw'
        self.osc2_detune = 0.0
        self.osc2_level = 0.0
        self.osc2_phase = 0.0
        
        # FX Parameters
        self.fx_distortion = 0.0
        self.fx_delay_time = 0.3
        self.fx_delay_feedback = 0.4
        self.fx_delay_mix = 0.0
        self.fx_reverb_mix = 0.0
        self.delay_buffer_size = int(SR * 2.0)
        self.delay_buffer = np.zeros(self.delay_buffer_size, dtype=np.float32)
        self.delay_write_idx = 0

        # ADSR Envelope parameters
        self.attack_time = 0.01   # seconds
        self.decay_time = 0.1     # seconds
        self.sustain_level = 0.8  # 0-1
        self.release_time = 0.3   # seconds

    def set_waveform(self, waveform: str) -> None:
        with self._lock:
            self.waveform_type = waveform

    def set_osc_wave(self, osc: int, wave: str) -> None:
        with self._lock:
            if osc == 1:
                self.osc1_wave = wave
            elif osc == 2:
                self.osc2_wave = wave

    def set_osc_detune(self, osc: int, value: float) -> None:
        with self._lock:
            if osc == 1:
                self.osc1_detune = value
            elif osc == 2:
                self.osc2_detune = value

    def set_osc_level(self, osc: int, value: float) -> None:
        with self._lock:
            if osc == 1:
                self.osc1_level = value
            elif osc == 2:
                self.osc2_level = value

    def set_osc_phase(self, osc: int, value: float) -> None:
        with self._lock:
            if osc == 1:
                self.osc1_phase = value
            elif osc == 2:
                self.osc2_phase = value

    def note_on(self, midi_note: int) -> None:
        with self._lock:
            freq = 440 * (2 ** ((midi_note - 69) / 12))
            self.active_notes[midi_note] = {
                'freq': freq, 
                'phase1': self.osc1_phase * np.pi * 2, 
                'phase2': self.osc2_phase * np.pi * 2, 
                'vel': 0.8,
                'start_time': time.time(),
                'envelope_stage': 'attack',
                'envelope_value': 0.0,
                'release_start': None
            }

    def note_off(self, midi_note: int) -> None:
        with self._lock:
            if midi_note in self.active_notes:
                note = self.active_notes[midi_note]
                if note['envelope_stage'] != 'release':
                    note['envelope_stage'] = 'release'
                    note['release_start'] = time.time()

    def _get_envelope_value(self, note: Dict, current_time: float) -> float:
        """Calculate current envelope value based on ADSR stage"""
        elapsed = current_time - note['start_time']

        if note['envelope_stage'] == 'attack':
            if elapsed < self.attack_time:
                return elapsed / self.attack_time
            else:
                note['envelope_stage'] = 'decay'
                elapsed -= self.attack_time

        if note['envelope_stage'] == 'decay':
            if elapsed < self.decay_time:
                decay_amount = 1.0 - self.sustain_level
                return 1.0 - decay_amount * (elapsed / self.decay_time)
            else:
                note['envelope_stage'] = 'sustain'
                return self.sustain_level

        if note['envelope_stage'] == 'sustain':
            return self.sustain_level

        if note['envelope_stage'] == 'release':
            if note['release_start'] is None:
                return 0.0
            release_elapsed = current_time - note['release_start']
            if release_elapsed >= self.release_time:
                return 0.0
            else:
                sustain_value = note.get('envelope_value', self.sustain_level)
                return sustain_value * (1.0 - release_elapsed / self.release_time)

        return 0.0

    def set_effect_param(self, name: str, value: float) -> None:
        with self._lock:
            if name == 'distortion': self.fx_distortion = value
            elif name == 'delay_time': self.fx_delay_time = value
            elif name == 'delay_feedback': self.fx_delay_feedback = value
            elif name == 'delay_mix': self.fx_delay_mix = value
            elif name == 'reverb_mix': self.fx_reverb_mix = value

    def _apply_effects(self, buffer: np.ndarray) -> np.ndarray:
        if self.fx_distortion > 0.01:
            drive = 1.0 + (self.fx_distortion * 10.0)
            buffer = np.tanh(buffer * drive)
        if self.fx_delay_mix > 0.01 or self.fx_reverb_mix > 0.01:
            delay_samples = int(self.fx_delay_time * self.sample_rate)
            delay_samples = max(1, min(delay_samples, self.delay_buffer_size - 1))
            wet_mix = max(self.fx_delay_mix, self.fx_reverb_mix)
            feedback = self.fx_delay_feedback
            if self.fx_reverb_mix > 0.5: feedback = 0.7 + (self.fx_reverb_mix * 0.25)
            output = np.zeros_like(buffer)
            for i in range(len(buffer)):
                read_idx = int(self.delay_write_idx - delay_samples)
                if read_idx < 0: read_idx += self.delay_buffer_size
                delayed_sample = self.delay_buffer[read_idx]
                self.delay_buffer[self.delay_write_idx] = buffer[i] + (delayed_sample * feedback)
                output[i] = (buffer[i] * (1.0 - wet_mix)) + (delayed_sample * wet_mix)
                self.delay_write_idx += 1
                if self.delay_write_idx >= self.delay_buffer_size: self.delay_write_idx = 0
            buffer = output
        return buffer

    def _generate_wave(self, phases: np.ndarray, wave_type: str, detune_cents: float) -> np.ndarray:
        if detune_cents != 0.0:
            detune_ratio = 2 ** (detune_cents / 1200.0)
            phases = (phases * detune_ratio) % 1.0
        
        if wave_type == 'sine':
            return np.sin(2 * np.pi * phases)
        elif wave_type == 'square':
            return np.where(phases < 0.5, 1.0, -1.0)
        elif wave_type == 'saw':
            return 2 * (phases - 0.5)
        elif wave_type == 'triangle':
            return 2 * np.abs(2 * (phases - 0.5)) - 1
        return np.sin(2 * np.pi * phases)

    def generate_audio(self, num_samples: int) -> np.ndarray:
        buffer = np.zeros(num_samples, dtype=np.float32)
        current_time = time.time()
        
        with self._lock:
            active_count = len(self.active_notes)
            
            if active_count == 0:
                buffer = self._apply_effects(buffer)
                self.last_buffer = buffer
                return buffer
            
            notes_to_remove = []
            
            for midi_note, note_data in list(self.active_notes.items()):
                envelope = self._get_envelope_value(note_data, current_time)
                note_data['envelope_value'] = envelope

                if envelope <= 0.001:  # Very quiet, remove note
                    notes_to_remove.append(midi_note)
                    continue

                freq = note_data['freq']
                phase1 = note_data['phase1']
                phase2 = note_data['phase2']
                vel = note_data['vel']
                
                t = np.arange(num_samples)
                
                phases1 = (phase1 / (np.pi * 2) + (freq / self.sample_rate) * t) % 1.0
                phases2 = (phase2 / (np.pi * 2) + (freq / self.sample_rate) * t) % 1.0
                
                wave1 = self.waveform_type if self.use_global_waveform else self.osc1_wave
                wave2 = self.waveform_type if self.use_global_waveform else self.osc2_wave
                
                osc1 = self._generate_wave(phases1, wave1, self.osc1_detune)
                osc2 = self._generate_wave(phases2, wave2, self.osc2_detune)
                
                mix = (osc1 * self.osc1_level) + (osc2 * self.osc2_level)
                buffer += mix * vel * envelope  # Apply envelope
                
                # Update phases for next buffer
                note_data['phase1'] = phases1[-1] * 2 * np.pi
                note_data['phase2'] = phases2[-1] * 2 * np.pi
            
            # Remove finished notes
            for midi_note in notes_to_remove:
                del self.active_notes[midi_note]
            
            if active_count > 1:
                max_val = np.max(np.abs(buffer))
                if max_val > 0: buffer = buffer / max_val * 0.9
            buffer *= AMP
        
        buffer = self._apply_effects(buffer)
        buffer = np.clip(buffer, -1.0, 1.0)
        self.last_buffer = buffer
        return buffer


class RotaryKnob(Canvas):
    def __init__(self, parent, value=0.0, min_val=0.0, max_val=1.0, 
                 label="", format_fn=None, on_change=None, size=52, **kw):
        super().__init__(parent, width=size + 40, height=size + 60, 
                        bg=BG_BLACK, highlightthickness=0, **kw)
        
        self.value = value
        self.default_value = value
        self.min_val = min_val
        self.max_val = max_val
        self.label = label
        self.format_fn = format_fn or (lambda v: f'{v:.0f}')
        self.on_change = on_change
        self.size = size
        self.radius = size / 2
        self.center_x = self.radius + 20
        self.center_y = self.radius + 30
        self.min_angle = -135
        self.max_angle = 135
        self.is_dragging = False
        self.drag_start_y = 0
        self.drag_start_value = 0
        
        self.bind('<Button-1>', self._on_mouse_down)
        self.bind('<B1-Motion>', self._on_mouse_drag)
        self.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.bind('<Double-Button-1>', self._on_double_click)
        self.bind('<MouseWheel>', self._on_mouse_wheel)
        self.bind('<Button-4>', self._on_mouse_wheel)
        self.bind('<Button-5>', self._on_mouse_wheel)
        
        self.draw()
    
    def _get_angle(self):
        if self.max_val == self.min_val:
            return self.min_angle
        normalized = (self.value - self.min_val) / (self.max_val - self.min_val)
        return self.min_angle + normalized * (self.max_angle - self.min_angle)
    
    def set_value(self, value):
        old_value = self.value
        self.value = max(self.min_val, min(self.max_val, value))
        if abs(self.value - old_value) > 0.001:
            self.draw()
            if self.on_change:
                self.on_change(self.value)
    
    def reset_to_default(self):
        if abs(self.value - self.default_value) > 0.001:
            self.set_value(self.default_value)
    
    def _on_mouse_down(self, event):
        dx = event.x - self.center_x
        dy = event.y - self.center_y
        distance = math.sqrt(dx**2 + dy**2)
        if distance < self.radius * 0.8:
            self.is_dragging = True
            self.drag_start_y = event.y_root
            self.drag_start_value = self.value
            self.config(cursor='fleur')
    
    def _on_mouse_drag(self, event):
        if not self.is_dragging:
            return
        delta_y = self.drag_start_y - event.y_root
        sensitivity = (self.max_val - self.min_val) / 200
        self.set_value(self.drag_start_value + delta_y * sensitivity)
    
    def _on_mouse_up(self, event):
        self.is_dragging = False
        self.config(cursor='hand2')
    
    def _on_double_click(self, event):
        dx = event.x - self.center_x
        dy = event.y - self.center_y
        distance = math.sqrt(dx**2 + dy**2)
        if distance < self.radius * 0.8:
            self.reset_to_default()
    
    def _on_mouse_wheel(self, event):
        delta = 0.02 if (event.num == 4 or event.delta > 0) else -0.02
        self.set_value(self.value + delta * (self.max_val - self.min_val))
        return 'break'
    
    def draw(self):
        self.delete('all')
        angle = self._get_angle()
        angle_rad = math.radians(angle)
        
        self.create_oval(self.center_x - self.radius, self.center_y - self.radius,
                        self.center_x + self.radius, self.center_y + self.radius,
                        fill=KNOB_BODY, outline=BORDER, width=1)
        
        inner_r = self.radius * 0.75
        self.create_oval(self.center_x - inner_r, self.center_y - inner_r,
                        self.center_x + inner_r, self.center_y + inner_r,
                        fill=KNOB_SURFACE, outline=BORDER, width=1)
        
        arc_radius = self.radius * 0.90
        if self.value > self.min_val:
            self.create_arc(self.center_x - arc_radius, self.center_y - arc_radius,
                          self.center_x + arc_radius, self.center_y + arc_radius,
                          start=self.min_angle, extent=angle - self.min_angle,
                          style='arc', outline=FG_BRIGHT, width=3)
        
        line_start_r = self.radius * 0.2
        line_end_r = self.radius * 0.6
        start_x = self.center_x + line_start_r * math.cos(angle_rad - math.pi/2)
        start_y = self.center_y + line_start_r * math.sin(angle_rad - math.pi/2)
        end_x = self.center_x + line_end_r * math.cos(angle_rad - math.pi/2)
        end_y = self.center_y + line_end_r * math.sin(angle_rad - math.pi/2)
        self.create_line(start_x, start_y, end_x, end_y, fill=FG_BRIGHT, width=2)
        
        self.create_oval(self.center_x - self.radius * 0.15, self.center_y - self.radius * 0.15,
                        self.center_x + self.radius * 0.15, self.center_y + self.radius * 0.15,
                        fill=BORDER, outline=KNOB_SURFACE, width=1)
        
        self.create_text(self.center_x, self.center_y + self.radius + 25,
                        text=self.label.upper(), font=(FONT_MAIN, 7, 'bold'), fill=FG_DIM)
        self.create_text(self.center_x, 12, text=self.format_fn(self.value),
                        font=(FONT_MAIN, 9, 'bold'), fill=FG_WHITE)


class PianoKeyboard(Canvas):
    def __init__(self, parent, start_note=60, num_octaves=2, **kw):
        super().__init__(parent, bg=BG_BLACK, highlightthickness=0, **kw)
        self.start_note = start_note
        self.num_octaves = num_octaves
        self.white_key_width = 40
        self.black_key_width = 24
        self.white_key_height = 120
        self.black_key_height = 75
        self.active_keys = set()
        self.key_rects = {}
        self._draw_keyboard()
    
    def _draw_keyboard(self):
        self.delete('all')
        self.key_rects = {}
        x = 10
        white_notes = [0, 2, 4, 5, 7, 9, 11]
        
        for octave in range(self.num_octaves):
            for offset in white_notes:
                midi_note = self.start_note + (octave * 12) + offset
                key_letter = None
                for k, v in WHITE_KEYS.items():
                    if v == midi_note:
                        key_letter = k.upper()
                        break
                
                rect = self.create_rectangle(x, 0, x + self.white_key_width, self.white_key_height,
                                           fill=BG_DARK, outline=BORDER, width=1)
                self.key_rects[midi_note] = rect
                
                if key_letter:
                    self.create_text(x + self.white_key_width / 2, self.white_key_height - 20,
                                   text=key_letter, fill=FG_DIM, font=(FONT_MAIN, 10))
                if midi_note in OCTAVE_LABELS:
                    self.create_text(x + self.white_key_width / 2, 15,
                                   text=OCTAVE_LABELS[midi_note], fill=FG_BRIGHT, font=(FONT_MAIN, 9, 'bold'))
                x += self.white_key_width
        
        x = 10
        black_notes = [(1, 0.7), (3, 1.7), (6, 3.7), (8, 4.7), (10, 5.7)]
        for octave in range(self.num_octaves):
            for offset, white_key_offset in black_notes:
                midi_note = self.start_note + (octave * 12) + offset
                if midi_note in self.key_rects:
                    continue
                key_letter = None
                for k, v in BLACK_KEYS.items():
                    if v == midi_note:
                        key_letter = k.upper()
                        break
                black_x = x + (white_key_offset * self.white_key_width) - (self.black_key_width / 2)
                rect = self.create_rectangle(black_x, 0, black_x + self.black_key_width, self.black_key_height,
                                           fill='#1a1a1a', outline=BORDER, width=1)
                self.key_rects[midi_note] = rect
                if key_letter:
                    self.create_text(black_x + self.black_key_width / 2, self.black_key_height - 15,
                                   text=key_letter, fill=FG_DIM, font=(FONT_MAIN, 9))
            x += 7 * self.white_key_width
        
        self.config(width=x + 10, height=self.white_key_height + 10)
    
    def highlight_key(self, midi_note, active=True):
        if midi_note not in self.key_rects:
            return
        rect_id = self.key_rects[midi_note]
        if active:
            self.itemconfig(rect_id, fill=FG_BRIGHT, outline=FG_WHITE)
            self.active_keys.add(midi_note)
        else:
            is_black = midi_note in [61, 63, 66, 68, 70, 73, 75, 78, 80, 82, 85, 87]
            self.itemconfig(rect_id, fill='#1a1a1a' if is_black else BG_DARK, outline=BORDER)
            self.active_keys.discard(midi_note)


def sep(parent, vertical=False):
    return tk.Frame(parent, bg=BORDER, width=1 if vertical else 100, height=1 if not vertical else 100)

def label(parent, text, size=9, color=FG_DIM, bold=False):
    return tk.Label(parent, text=text, bg=BG_BLACK, fg=color, font=(FONT_MAIN, size, 'bold' if bold else 'normal'))


class SerumApp:
    def __init__(self, root):
        self.root = root
        root.title('SERUM 2 [TERMINAL EDITION]')
        root.configure(bg=BG_BLACK)
        root.geometry('1200x900')
        root.minsize(1000, 800)
        
        self.synth = PolyphonicSynth()
        self._wt_canvases = []
        self._playing_notes = {}
        self._waveform_var = tk.StringVar(value='sine')
        self._waveform_btns = {}
        self._waveform_index = 0
        self._osc1_wave_var = tk.StringVar(value='sine')
        self._osc2_wave_var = tk.StringVar(value='saw')
        
        self.stream = None
        self.display_canvas = None
        self.animation_id = None
        self.piano_keyboard = None
        
        self._build()
        root.after(100, self._init_mini_waves)
        root.after(100, self._start_audio)
        root.bind('<KeyPress>', self._on_key_press)
        root.bind('<KeyRelease>', self._on_key_release)
        root.protocol('WM_DELETE_WINDOW', self._on_closing)

    def _start_audio(self):
        try:
            self.stream = sd.OutputStream(channels=1, samplerate=SR, blocksize=BUFFER_SIZE, callback=self._audio_callback)
            self.stream.start()
            print("[SYSTEM] Audio stream started")
        except Exception as e:
            print(f"[ERROR] Audio: {e}")

    def _audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(f"[STATUS] {status}")
        outdata[:, 0] = self.synth.generate_audio(frames)

    def _on_closing(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
        self.root.destroy()

    def _build(self):
        self._build_titlebar()
        body = tk.Frame(self.root, bg=BG_BLACK)
        body.pack(fill='both', expand=True)
        self._build_left(body)
        self._build_right(body)
        self._build_center(body)
        self._build_keyboard()
        self._build_bottombar()

    def _build_titlebar(self):
        bar = tk.Frame(self.root, bg=BG_BLACK, height=35)
        bar.pack(fill='x')
        bar.pack_propagate(False)
        sep(bar).pack(side='bottom', fill='x')
        label(bar, 'SERUM 2 // TERMINAL EDITION', size=11, color=FG_WHITE, bold=True).pack(side='left', padx=15)
        right = tk.Frame(bar, bg=BG_BLACK)
        right.pack(side='right', padx=15)
        label(right, 'OSC 1 + OSC 2 | Independent Controls', size=9).pack(side='left', padx=15)

    def _build_bottombar(self):
        bar = tk.Frame(self.root, bg=BG_BLACK, height=28)
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)
        sep(bar).pack(side='top', fill='x')
        label(bar, 'PRESET: DEFAULT_PATCH', size=8).pack(side='left', padx=15)
        self.cpu_label = label(bar, 'VOICES: 0/16', size=8)
        self.cpu_label.pack(side='right', padx=15)

    def _build_keyboard(self):
        kb_frame = tk.Frame(self.root, bg=BG_BLACK, height=140)
        kb_frame.pack(fill='x', side='bottom')
        sep(kb_frame).pack(side='top', fill='x')
        self.piano_keyboard = PianoKeyboard(kb_frame, start_note=60, num_octaves=2)
        self.piano_keyboard.pack(pady=10)

    def _build_left(self, parent):
        """Left panel with OSC 1 & OSC 2 - WORKING SCROLLBAR"""
        outer = tk.Frame(parent, bg=BG_BLACK, width=320)
        outer.pack(side='left', fill='y')
        outer.pack_propagate(False)
        sep(outer, vertical=True).pack(side='right', fill='y')
        
        # Canvas with scrollbar
        cv = tk.Canvas(outer, bg=BG_BLACK, highlightthickness=0)
        sb = tk.Scrollbar(outer, orient='vertical', command=cv.yview, 
                         bg=BG_BLACK, troughcolor=BG_DARK, activebackground=FG_DIM)
        inner = tk.Frame(cv, bg=BG_BLACK)
        
        inner.bind('<Configure>', lambda e: cv.configure(scrollregion=cv.bbox('all')))
        win_id = cv.create_window((0, 0), window=inner, anchor='nw')
        cv.configure(yscrollcommand=sb.set)
        cv.bind('<Configure>', lambda e: cv.itemconfig(win_id, width=e.width))
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            cv.yview_scroll(int(-1*(event.delta/120)), 'units')
        cv.bind('<MouseWheel>', _on_mousewheel)
        cv.bind('<Button-4>', lambda e: cv.yview_scroll(-1, 'units'))
        cv.bind('<Button-5>', lambda e: cv.yview_scroll(1, 'units'))
        
        cv.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        
        # Build sections
        self._build_waveform_selector(inner)
        self._build_osc(inner, 1, 'OSC 1', 'sine', 0, 100, 0)
        self._build_osc(inner, 2, 'OSC 2', 'saw', 0, 0, 0)

    def _build_waveform_selector(self, parent):
        sep(parent).pack(fill='x')
        sec = tk.Frame(parent, bg=BG_BLACK)
        sec.pack(fill='x', padx=12, pady=10)
        
        label(sec, '[GLOBAL WAVEFORM]', size=9, color=FG_WHITE, bold=True).pack(anchor='w', pady=(0, 6))
        label(sec, 'M=Next N=Previous (overrides OSC)', size=7, color=FG_DIM).pack(anchor='w', pady=(0, 8))
        
        btn_frame = tk.Frame(sec, bg=BG_BLACK)
        btn_frame.pack(fill='x')
        
        waveforms = [('sine', 'SINE'), ('triangle', 'TRI'), ('saw', 'SAW'), ('square', 'SQR')]
        for i, (wave, text) in enumerate(waveforms):
            row = i // 2
            col = i % 2
            is_active = (wave == 'sine')
            btn = tk.Button(btn_frame, text=f'[{text}]', bg=FG_BRIGHT if is_active else BORDER,
                          fg=BG_BLACK if is_active else FG_DIM, font=(FONT_MAIN, 8, 'bold'),
                          relief='flat', width=6, height=2, cursor='hand2',
                          command=lambda w=wave: self._set_global_waveform(w))
            btn.grid(row=row, column=col, padx=4, pady=4, sticky='ew')
            self._waveform_btns[wave] = btn
        
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

    def _set_global_waveform(self, waveform: str):
        if waveform not in WAVEFORMS:
            return
        self.synth.use_global_waveform = True
        self.synth.set_waveform(waveform)
        self._waveform_index = WAVEFORMS.index(waveform)
        for wave, btn in self._waveform_btns.items():
            btn.config(bg=FG_BRIGHT if wave == waveform else BORDER, 
                      fg=BG_BLACK if wave == waveform else FG_DIM)

    def _cycle_waveform(self, direction: int):
        self._waveform_index = (self._waveform_index + direction) % len(WAVEFORMS)
        self._set_global_waveform(WAVEFORMS[self._waveform_index])

    def _build_osc(self, parent, osc_num, name, wave_type, detune, level, phase):
        """Build OSC section with waveform buttons + 4 knobs"""
        sep(parent).pack(fill='x')
        sec = tk.Frame(parent, bg=BG_BLACK)
        sec.pack(fill='x', padx=12, pady=10)
        
        # Header
        hdr = tk.Frame(sec, bg=BG_BLACK)
        hdr.pack(fill='x', pady=(0, 8))
        label(hdr, f'[{name}]', size=9, color=FG_WHITE, bold=True).pack(side='left')
        
        on_var = tk.BooleanVar(value=True)
        tog_btn = tk.Button(hdr, text='[ON]', bg=FG_BRIGHT, fg=BG_BLACK, font=(FONT_MAIN, 8),
                           relief='flat', width=3, cursor='hand2')
        tog_btn.pack(side='right')
        
        # Wavetable display
        wt_canvas = tk.Canvas(sec, width=280, height=50, bg=BG_BLACK, 
                             highlightthickness=1, highlightbackground=BORDER)
        wt_canvas.pack(pady=(0, 8))
        self._wt_canvases.append((wt_canvas, wave_type, 2.0))
        
        # Waveform buttons
        wave_frame = tk.Frame(sec, bg=BG_BLACK)
        wave_frame.pack(fill='x', pady=(0, 8))
        label(wave_frame, 'WAVEFORM:', size=7, color=FG_DIM).pack(anchor='w')
        
        wave_btns = tk.Frame(wave_frame, bg=BG_BLACK)
        wave_btns.pack(fill='x')
        for wave, text in [('sine', 'SIN'), ('triangle', 'TRI'), ('saw', 'SAW'), ('square', 'SQR')]:
            btn = tk.Button(wave_btns, text=text, bg=FG_BRIGHT if wave == wave_type else BORDER,
                          fg=BG_BLACK if wave == wave_type else FG_DIM, font=(FONT_MAIN, 7),
                          relief='flat', width=5, cursor='hand2',
                          command=lambda w=wave: self._set_osc_wave(osc_num, w))
            btn.pack(side='left', padx=2)
        
        # 4 Knobs in a row: Detune, Level, Phase
        knob_frame = tk.Frame(sec, bg=BG_BLACK)
        knob_frame.pack(fill='x')
        for i, (lbl, init, min_v, max_v, fmt, param) in enumerate([
            ('DET', 0.5, 0.0, 1.0, lambda v: f'{int((v-0.5)*100):+d}', 'detune'),
            ('LEV', 1.0 if osc_num == 1 else 0.0, 0.0, 1.0, lambda v: f'{int(v*100)}%', 'level'),
            ('PHS', 0.0, 0.0, 1.0, lambda v: f'{int(v*360)}°', 'phase')
        ]):
            ctrl = tk.Frame(knob_frame, bg=BG_BLACK)
            ctrl.pack(side='left', expand=True, fill='x', padx=3)
            
            val_lbl = label(ctrl, fmt(init), size=9, color=FG_WHITE, bold=True)
            val_lbl.pack()
            
            knob = RotaryKnob(ctrl, value=init, min_val=min_v, max_val=max_v,
                            label=lbl, size=52, format_fn=fmt,
                            on_change=lambda v, o=osc_num, p=param, l=val_lbl: self._on_osc_change(o, p, v, l))
            knob.pack()

    def _set_osc_wave(self, osc_num, waveform):
        self.synth.use_global_waveform = False
        self.synth.set_osc_wave(osc_num, waveform)
    
    def _on_osc_change(self, osc_num, param, value, label):
        if param == 'detune':
            label.config(text=f'{int((value-0.5)*100):+d}')
            self.synth.set_osc_detune(osc_num, int((value - 0.5) * 100))
        elif param == 'level':
            label.config(text=f'{int(value*100)}%')
            self.synth.set_osc_level(osc_num, value)
        elif param == 'phase':
            label.config(text=f'{int(value*360)}°')
            self.synth.set_osc_phase(osc_num, value)

    def _build_right(self, parent):
        """Right panel - Master, Filter, FX (as before)"""
        outer = tk.Frame(parent, bg=BG_BLACK, width=320)
        outer.pack(side='right', fill='y')
        outer.pack_propagate(False)
        sep(outer, vertical=True).pack(side='left', fill='y')
        
        cv = tk.Canvas(outer, bg=BG_BLACK, highlightthickness=0)
        sb = tk.Scrollbar(outer, orient='vertical', command=cv.yview, 
                         bg=BG_BLACK, troughcolor=BG_DARK, activebackground=FG_DIM)
        inner = tk.Frame(cv, bg=BG_BLACK)
        
        inner.bind('<Configure>', lambda e: cv.configure(scrollregion=cv.bbox('all')))
        win_id = cv.create_window((0, 0), window=inner, anchor='nw')
        cv.configure(yscrollcommand=sb.set)
        cv.bind('<Configure>', lambda e: cv.itemconfig(win_id, width=e.width))
        cv.bind('<MouseWheel>', lambda e: cv.yview_scroll(int(-1*(e.delta/120)), 'units'))
        cv.bind('<Button-4>', lambda e: cv.yview_scroll(-1, 'units'))
        cv.bind('<Button-5>', lambda e: cv.yview_scroll(1, 'units'))
        
        cv.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        
        self._build_master(inner)
        self._build_filter(inner)
        self._build_effects(inner)

    def _build_master(self, parent):
        sep(parent).pack(fill='x')
        sec = tk.Frame(parent, bg=BG_BLACK)
        sec.pack(fill='x', padx=12, pady=10)
        label(sec, '[MASTER]', size=9, color=FG_WHITE, bold=True).pack(anchor='w', pady=(0, 8))
        
        row = tk.Frame(sec, bg=BG_BLACK)
        row.pack(fill='x')
        for k in [
            {'key': 'master_pitch', 'label': 'PITCH', 'initial': 0.5, 'format': lambda v: f'{(v-0.5)*24:+.1f}'},
            {'key': 'master_glide', 'label': 'GLIDE', 'initial': 0.0, 'format': lambda v: f'{int(v*100)}'},
            {'key': 'master_level', 'label': 'LEVEL', 'initial': 1.0, 'format': lambda v: f'{int(v*127)}'}
        ]:
            self._create_knob_column(row, k)

    def _create_knob_column(self, parent, knob_def):
        f = tk.Frame(parent, bg=BG_BLACK)
        f.pack(side='left', expand=True, fill='x')
        disp = label(f, knob_def['format'](knob_def['initial']), size=9, color=FG_WHITE, bold=True)
        disp.pack(pady=(0, 4))
        knob = RotaryKnob(f, value=knob_def['initial'], min_val=0.0, max_val=1.0,
                         label=knob_def['label'], size=52, format_fn=knob_def['format'])
        knob.pack()
        label(f, knob_def['label'], size=7, color=FG_DIM).pack(pady=(4, 0))

    def _build_filter(self, parent):
        sep(parent).pack(fill='x')
        sec = tk.Frame(parent, bg=BG_BLACK)
        sec.pack(fill='x', padx=12, pady=10)
        label(sec, '[FILTER]', size=9, color=FG_WHITE, bold=True).pack(anchor='w', pady=(0, 6))
        
        btn_row = tk.Frame(sec, bg=BG_BLACK)
        btn_row.pack(fill='x', pady=(0, 8))
        for ft in ['LP', 'HP', 'BP', 'NT']:
            b = tk.Button(btn_row, text=ft, font=(FONT_MAIN, 8), relief='flat', 
                         bg=BORDER, fg=FG_DIM, activebackground=BG_BLACK, 
                         activeforeground=FG_WHITE, cursor='hand2', width=3)
            b.pack(side='left', padx=3)
        
        row = tk.Frame(sec, bg=BG_BLACK)
        row.pack(fill='x')
        for k in [
            {'key': 'filter_cutoff', 'label': 'CUTOFF', 'initial': 0.5, 'format': lambda v: f'{int(v*100)}'},
            {'key': 'filter_res', 'label': 'RES', 'initial': 0.3, 'format': lambda v: f'{int(v*100)}'},
            {'key': 'filter_drive', 'label': 'DRV', 'initial': 0.0, 'format': lambda v: f'{int(v*100)}'}
        ]:
            self._create_knob_column(row, k)

    def _build_effects(self, parent):
        sep(parent).pack(fill='x')
        sec = tk.Frame(parent, bg=BG_BLACK)
        sec.pack(fill='x', padx=12, pady=10)
        label(sec, '[FX]', size=9, color=FG_WHITE, bold=True).pack(anchor='w', pady=(0, 6))
        
        for fx_name, params in [
            ('DISTORTION', [{'key': 'fx_dist', 'label': 'AMT', 'initial': 0.0, 'format': lambda v: f'{int(v*100)}%'}]),
            ('DELAY', [
                {'key': 'fx_delay_time', 'label': 'TIME', 'initial': 0.3, 'format': lambda v: f'{v:.2f}s'},
                {'key': 'fx_delay_fdbk', 'label': 'FDBK', 'initial': 0.4, 'format': lambda v: f'{int(v*100)}%'},
                {'key': 'fx_delay_mix', 'label': 'MIX', 'initial': 0.0, 'format': lambda v: f'{int(v*100)}%'}
            ]),
            ('REVERB', [{'key': 'fx_reverb', 'label': 'AMT', 'initial': 0.0, 'format': lambda v: f'{int(v*100)}%'}])
        ]:
            box = tk.Frame(sec, bg=BG_PANEL, bd=1, relief='solid')
            box.pack(fill='x', pady=2)
            label(tk.Frame(box, bg=BG_PANEL), fx_name, size=7).pack(side='left', padx=5, pady=3)
            row = tk.Frame(box, bg=BG_PANEL)
            row.pack(fill='x', padx=5, pady=(0, 4))
            for k in params:
                self._create_knob_column(row, k)

    def _build_center(self, parent):
        frame = tk.Frame(parent, bg=BG_BLACK)
        frame.pack(side='left', fill='both', expand=True)
        
        hdr = tk.Frame(frame, bg=BG_BLACK, height=35)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        sep(hdr).pack(side='bottom', fill='x')
        label(hdr, 'VISUALIZER', size=9, color=FG_WHITE, bold=True).pack(side='left', padx=12, pady=8)
        
        content = tk.Frame(frame, bg=BG_BLACK)
        content.pack(fill='both', expand=True, padx=12, pady=12)
        
        self.display_canvas = tk.Canvas(content, bg=BG_BLACK, highlightthickness=1, highlightbackground=BORDER)
        self.display_canvas.pack(fill='both', expand=True)
        
        self._animate_wavetable()

    def _animate_wavetable(self):
        if not self.display_canvas:
            return
        w = self.display_canvas.winfo_width()
        h = self.display_canvas.winfo_height()
        
        if w > 10 and h > 10:
            self.display_canvas.delete('all')
            for x in range(0, w + 40, 40):
                self.display_canvas.create_line(x, 0, x, h, fill=BORDER)
            for y in range(0, h + 40, 40):
                self.display_canvas.create_line(0, y, w, y, fill=BORDER)
            self.display_canvas.create_line(0, h // 2, w, h // 2, fill=FG_DIM, width=1)
            
            audio = self.synth.last_buffer
            pts = []
            downsample = max(1, len(audio) // w)
            for i in range(0, len(audio), downsample):
                x = int((i / len(audio)) * w)
                y = h // 2 - int(audio[i] * (h / 2.5))
                pts.extend([x, y])
            if len(pts) >= 4:
                self.display_canvas.create_line(pts, fill=FG_WHITE, width=1, smooth=True)
        
        self.animation_id = self.root.after(33, self._animate_wavetable)

    def _on_key_press(self, e):
        key = e.keysym.lower()
        if key == 'm':
            self._cycle_waveform(1)
            return
        elif key == 'n':
            self._cycle_waveform(-1)
            return
        if key in ALL_KEYS and key not in self._playing_notes:
            midi_note = ALL_KEYS[key]
            self._playing_notes[key] = midi_note
            self.synth.note_on(midi_note)
            if self.piano_keyboard:
                self.piano_keyboard.highlight_key(midi_note, True)

    def _on_key_release(self, e):
        key = e.keysym.lower()
        if key in self._playing_notes:
            midi_note = self._playing_notes.pop(key)
            self.synth.note_off(midi_note)
            if self.piano_keyboard:
                self.piano_keyboard.highlight_key(midi_note, False)

    def _init_mini_waves(self):
        for canvas, wtype, freq in self._wt_canvases:
            canvas.delete('all')
            canvas.configure(bg=BG_BLACK)
            w = canvas.winfo_width() or 280
            h = canvas.winfo_height() or 50
            for i in range(5):
                canvas.create_line(0, int(h / 4 * i), w, int(h / 4 * i), fill=BORDER, width=1)
            pts = []
            for x in range(w):
                t = (x / w) * math.pi * 2 * freq
                if wtype == 'sine':
                    y = math.sin(t)
                elif wtype == 'saw':
                    y = 2 * ((t % (math.pi * 2)) / (math.pi * 2)) - 1
                elif wtype == 'square':
                    y = 1 if t % (math.pi * 2) < math.pi else -1
                else:
                    y = 2 * abs(2 * ((t % (math.pi * 2)) / (math.pi * 2)) - 1) - 1
                pts.extend([x, h / 2 - y * h / 2.5])
            if len(pts) >= 4:
                canvas.create_line(pts, fill=FG_WHITE, width=1, smooth=True)


if __name__ == '__main__':
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = SerumApp(root)
    root.mainloop()
