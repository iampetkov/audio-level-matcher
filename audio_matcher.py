#!/usr/bin/env python3
"""
Audio Level Matcher — Game A → Game B  (v3)
Adds: Session Timeline — visual multi-track arrangement with save/load templates.
"""

import os, sys, threading, subprocess, time, tempfile, shutil, json, uuid
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

SHORT_THRESHOLD  = 1.0
MEDIUM_THRESHOLD = 5.0
SILENCE_DBFS     = -80.0
MAX_GAIN_DB      = 30.0
MIN_GAIN_DB      = -30.0
AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".ogg"}
SOX              = "/opt/homebrew/bin/sox"

# ── Audio analysis ─────────────────────────────────────────────────────────────

@dataclass
class AudioStats:
    path: str; duration: float; sample_rate: int; channels: int
    peak_dbfs: float; rms_dbfs: float; lufs_integrated: float
    lufs_shortterm: float; strategy: str

# ── Session timeline data ───────────────────────────────────────────────────────

RULER_H      = 24
TRACK_H      = 36
TRACK_PAD    = 3
MUTE_BTN_W   = 22   # width of the mute button in the label column
SOLO_BTN_W   = 18   # width of the solo button in the label column
LABEL_W      = 110
DEFAULT_PPS  = 80.0   # pixels per second (zoom level)
TL_MIN_SEC   = 20.0   # minimum visible timeline duration
TL_H_FALLBACK = RULER_H + TRACK_H  # used only at widget creation

BLOCK_COLS = ["#3d7ab5","#5a9441","#b84040","#8855b0","#b87830",
              "#3a9090","#7060b0","#b05880","#60a060","#a06840"]

@dataclass(eq=False)
class SessionBlock:
    bid:       str
    slot_name: str
    file_abs:  str
    file_rel:  str
    track:     int
    start_sec: float
    dur:       float
    gain_db:   float
    loop:      bool
    trim_in:   float
    trim_out:  float
    fade_in:   float
    fade_out:  float
    color:     str

def _to_dbfs(v):
    return max(20.0 * np.log10(v), SILENCE_DBFS) if v > 0 else SILENCE_DBFS

def _from_db(db):
    return 10.0 ** (db / 20.0)

def _lighter(hex_col, factor=0.55):
    """Blend a hex color toward white by factor (0=original, 1=white)."""
    r = int(hex_col[1:3], 16)
    g = int(hex_col[3:5], 16)
    b = int(hex_col[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

_WF_LOADING = object()  # sentinel: waveform is being loaded in background

def analyze(path):
    data, sr = sf.read(path, always_2d=True)
    dur  = len(data) / sr
    mono = data.mean(axis=1)
    peak = _to_dbfs(float(np.max(np.abs(data))))
    rms  = _to_dbfs(float(np.sqrt(np.mean(mono**2))))
    meter = pyln.Meter(sr)
    try:
        li = meter.integrated_loudness(data) if dur >= 0.4 else SILENCE_DBFS
        if not np.isfinite(li): li = SILENCE_DBFS
    except: li = SILENCE_DBFS
    lst = SILENCE_DBFS
    if dur >= 3.0:
        bs, hs = int(sr*3), int(sr*1)
        for i in range(0, len(data)-bs+1, hs):
            try:
                v = meter.integrated_loudness(data[i:i+bs])
                if np.isfinite(v) and v > lst: lst = v
            except: pass
    strat = "peak_rms" if dur < SHORT_THRESHOLD else ("rms_lufs" if dur < MEDIUM_THRESHOLD else "lufs")
    return AudioStats(path, dur, sr, data.shape[1], peak, rms, li, lst, strat)

def compute_gain_db(ref, src):
    if ref.rms_dbfs <= SILENCE_DBFS and src.rms_dbfs <= SILENCE_DBFS:
        return 0.0, "both silent"
    s = src.strategy
    if s == "peak_rms":
        gp, gr = ref.peak_dbfs-src.peak_dbfs, ref.rms_dbfs-src.rms_dbfs
        if ref.rms_dbfs <= SILENCE_DBFS or src.rms_dbfs <= SILENCE_DBFS:
            g, r = gp, f"peak only: {gp:+.1f} dB"
        else:
            g, r = 0.5*gp+0.5*gr, f"peak+RMS: {0.5*gp+0.5*gr:+.1f} dB"
    elif s == "rms_lufs":
        gr = ref.rms_dbfs-src.rms_dbfs
        if ref.lufs_shortterm > SILENCE_DBFS and src.lufs_shortterm > SILENCE_DBFS:
            gl = ref.lufs_shortterm-src.lufs_shortterm
            g, r = 0.4*gr+0.6*gl, f"RMS+LUFS: {0.4*gr+0.6*gl:+.1f} dB"
        else: g, r = gr, f"RMS: {gr:+.1f} dB"
    else:
        if ref.lufs_integrated > SILENCE_DBFS and src.lufs_integrated > SILENCE_DBFS:
            g, r = ref.lufs_integrated-src.lufs_integrated, f"LUFS: {ref.lufs_integrated-src.lufs_integrated:+.1f} dB"
        elif ref.lufs_shortterm > SILENCE_DBFS and src.lufs_shortterm > SILENCE_DBFS:
            g, r = ref.lufs_shortterm-src.lufs_shortterm, f"st-LUFS: {ref.lufs_shortterm-src.lufs_shortterm:+.1f} dB"
        else:
            g, r = ref.rms_dbfs-src.rms_dbfs, f"RMS: {ref.rms_dbfs-src.rms_dbfs:+.1f} dB"
    return max(MIN_GAIN_DB, min(MAX_GAIN_DB, g)), r

def _scurve_fade(length, fade_in=True):
    """S-curve (equal power cosine) fade envelope."""
    t = np.linspace(0.0, 1.0, length)
    if fade_in:
        return 0.5 * (1.0 - np.cos(np.pi * t))
    else:
        return 0.5 * (1.0 + np.cos(np.pi * t))

def apply_and_save(src_path, out_path, gain_db,
                   trim_in=0.0, trim_out=None,
                   fade_in_dur=0.0, fade_out_dur=0.0):
    """Read src, apply gain, trim, s-curve fades, write to out_path."""
    data, sr = sf.read(src_path, always_2d=True)
    total_dur = len(data) / sr

    # Trim
    s_in  = int(trim_in * sr)
    s_out = int((trim_out if trim_out is not None else total_dur) * sr)
    s_in  = max(0, min(s_in,  len(data)))
    s_out = max(s_in, min(s_out, len(data)))
    data  = data[s_in:s_out]

    # Gain
    data = data * _from_db(gain_db)

    # Fades
    fi_samps = int(fade_in_dur  * sr)
    fo_samps = int(fade_out_dur * sr)
    fi_samps = min(fi_samps, len(data) // 2)
    fo_samps = min(fo_samps, len(data) // 2)

    if fi_samps > 1:
        env = _scurve_fade(fi_samps, fade_in=True)
        data[:fi_samps] *= env[:, np.newaxis]
    if fo_samps > 1:
        env = _scurve_fade(fo_samps, fade_in=False)
        data[-fo_samps:] *= env[:, np.newaxis]

    # Peak limit
    pk = np.max(np.abs(data))
    if pk > 0.9999: data = data * (0.9999 / pk)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with sf.SoundFile(src_path) as f: sub, fmt = f.subtype, f.format
    sf.write(out_path, data, sr, subtype=sub, format=fmt)

def find_audio_files(folder):
    root = Path(folder)
    # key = lowercased rel path for case-insensitive matching; value = (orig_rel, abs_path)
    result = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTENSIONS:
            rel = str(p.relative_to(root))
            result[rel.lower()] = (rel, str(p))
    return result

def match_files(a_dir, b_dir):
    a, b = find_audio_files(a_dir), find_audio_files(b_dir)
    pairs = []
    for key in b:
        if key in a:
            orig_rel, b_abs = b[key]   # use Game B's original-case rel path for output
            _, a_abs = a[key]
            pairs.append((orig_rel, a_abs, b_abs))
    return sorted(pairs)

def all_b_files(b_dir):
    """Return sorted list of (orig_rel, abs_path) for every audio file in b_dir."""
    b = find_audio_files(b_dir)
    return sorted(b[k] for k in b)

# ── Waveform downsampling ──────────────────────────────────────────────────────

def build_waveform(path, n_bins=600):
    """Return (peaks_pos, peaks_neg) arrays of length n_bins, range 0-1."""
    data, sr = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)
    total = len(mono)
    bins  = np.array_split(mono, n_bins)
    pos   = np.array([max(0.0, float(b.max())) if len(b) else 0.0 for b in bins])
    neg   = np.array([min(0.0, float(b.min())) if len(b) else 0.0 for b in bins])
    peak  = max(pos.max(), -neg.min(), 1e-6)
    return pos / peak, neg / peak   # normalised to ±1

# ── Temp dir ───────────────────────────────────────────────────────────────────

_TEMP_DIR = tempfile.mkdtemp(prefix="alm_")

def _make_preview(src_path, gain_db, trim_in, trim_out, fi, fo):
    tmp = os.path.join(_TEMP_DIR, "preview.wav")
    apply_and_save(src_path, tmp, gain_db, trim_in, trim_out, fi, fo)
    return tmp

def _cleanup_temp():
    try: shutil.rmtree(_TEMP_DIR, ignore_errors=True)
    except: pass

# ── Player ─────────────────────────────────────────────────────────────────────

class Player:
    def __init__(self):
        self._proc     = None
        self._path     = None
        self._duration = 0.0
        self._position = 0.0
        self._start_ts = None
        self._lock     = threading.Lock()
        self.on_stop   = None

    def play(self, path, duration, start_pos=0.0):
        self.stop(save_position=False)
        with self._lock:
            self._path     = path
            self._duration = duration
            self._position = max(0.0, min(start_pos, duration))
            self._start_ts = time.monotonic()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self, save_position=True):
        with self._lock:
            proc = self._proc
            if save_position and self._start_ts is not None:
                elapsed = time.monotonic() - self._start_ts
                self._position = min(self._position + elapsed, self._duration)
            elif not save_position:
                self._position = 0.0
            self._proc = None; self._start_ts = None
        if proc:
            try: proc.kill()
            except: pass

    def is_playing(self):
        with self._lock: return self._proc is not None

    def get_position(self):
        with self._lock:
            if self._start_ts is None: return self._position
            return min(self._position + (time.monotonic()-self._start_ts), self._duration)

    def get_resume_pos(self):
        with self._lock: return self._position

    def _run(self):
        with self._lock:
            path = self._path; pos = self._position
        sox = SOX if os.path.isfile(SOX) else shutil.which("sox")
        if sox:
            cmd = [sox, path, "-d", "trim", str(pos)]
        else:
            cmd = ["afplay", path]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with self._lock: self._proc = proc
        proc.wait()
        with self._lock:
            natural = self._proc is not None
            self._proc = None; self._start_ts = None
            if natural: self._position = 0.0
        if natural and self.on_stop:
            self.on_stop()

# ── Waveform Canvas ────────────────────────────────────────────────────────────
# ── Waveform Canvas ────────────────────────────────────────────────────────────

CANVAS_H      = 90    # height per waveform
CANVAS_PAD    = 6     # vertical padding
TAB_H         = 18    # handle tab height at top of canvas
TAB_W         = 14    # handle tab width
HIT_TOL       = 12    # click tolerance in pixels
FADE_COLOR    = "#4a90d9"
TRIM_COLOR    = "#e05a00"
PLAYHEAD_COL  = "#ff3030"
REF_COLOR     = "#3a7a3a"
OUT_COLOR     = "#1a5fa8"

# ── Waveform Canvas ────────────────────────────────────────────────────────────

CANVAS_H     = 90
CANVAS_PAD   = 6
TAB_H        = 18
TAB_W        = 14
HIT_TOL      = 12
FADE_COLOR   = "#4a90d9"
TRIM_COLOR   = "#e05a00"
PLAYHEAD_COL = "#ff3030"
REF_COLOR    = "#3a7a3a"
OUT_COLOR    = "#1a5fa8"
END_COLOR    = "#cccccc"   # fill beyond file end

class WaveformPanel(tk.Frame):
    """
    Two stacked waveforms on a shared pixels-per-second scale.
    Each canvas shows its own file duration; grey fill beyond the shorter file.
    Trim/fade handles are on the output canvas only.
    Numeric entries allow precise values.
    """

    def __init__(self, parent, on_change=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_change = on_change

        self._ref_dur    = 1.0   # actual durations
        self._out_dur    = 1.0
        self._total_dur  = 1.0   # max(ref, out) — sets px/sec scale
        self._ref_wf     = None  # (pos_arr, neg_arr) normalised
        self._out_wf     = None
        self._trim_in_s  = 0.0   # seconds (absolute, not fractions)
        self._trim_out_s = 1.0
        self._fade_in_s  = 0.0
        self._fade_out_s = 0.0
        self._ref_ph_s   = 0.0   # playhead seconds, per file
        self._out_ph_s   = 0.0
        self._drag       = None
        self._drag_cv    = None
        self._width      = 600

        self._build()

    # ── properties (seconds) ─────────────────────────────────────────────────

    @property
    def trim_in(self):      return self._trim_in_s
    @property
    def trim_out(self):     return self._trim_out_s
    @property
    def fade_in_dur(self):  return self._fade_in_s
    @property
    def fade_out_dur(self): return self._fade_out_s

    def reset_handles(self):
        self._trim_in_s  = 0.0
        self._trim_out_s = self._out_dur
        self._fade_in_s  = 0.0
        self._fade_out_s = 0.0
        self._sync_entries()
        self._draw()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _s2x(self, seconds):
        """Convert seconds → canvas pixel x, using shared px/sec scale."""
        w = self._width
        return int(seconds / self._total_dur * w) if self._total_dur > 0 else 0

    def _x2s(self, x):
        """Convert canvas pixel x → seconds."""
        w = self._width
        return max(0.0, x / w * self._total_dur) if w > 0 else 0.0

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self):
        # Legend
        leg = tk.Frame(self)
        leg.pack(fill="x", pady=(0,2))
        tk.Label(leg, text="● Reference", font=("Helvetica", 10, "bold"),
                 fg=REF_COLOR).pack(side="left", padx=4)
        tk.Label(leg, text="● Output (trim/fade handles)",
                 font=("Helvetica", 10, "bold"), fg=OUT_COLOR).pack(side="left", padx=12)
        tk.Label(leg, text="▐ Trim IN/OUT", font=("Helvetica", 10),
                 fg=TRIM_COLOR).pack(side="left", padx=12)
        tk.Label(leg, text="┊ Fade IN/OUT", font=("Helvetica", 10),
                 fg=FADE_COLOR).pack(side="left", padx=12)

        # Ref canvas
        self._ref_canvas = tk.Canvas(self, height=CANVAS_H, bg="#f5f5f5",
                                     highlightthickness=1,
                                     highlightbackground="#bbb")
        self._ref_canvas.pack(fill="x", pady=(0,1))

        # Output canvas
        self._out_canvas = tk.Canvas(self, height=CANVAS_H, bg="#eef3fb",
                                     highlightthickness=1,
                                     highlightbackground="#bbb")
        self._out_canvas.pack(fill="x", pady=(0,1))

        # Shared time ruler
        self._ruler = tk.Canvas(self, height=20, bg="#e0e0e0",
                                highlightthickness=0)
        self._ruler.pack(fill="x")

        # Numeric controls
        ctrl = tk.Frame(self, pady=6)
        ctrl.pack(fill="x")

        def _ef(parent, label, color, cb):
            tk.Label(parent, text=label, font=("Helvetica", 10),
                     fg=color).pack(side="left", padx=(8,2))
            var = tk.StringVar(value="0.00")
            e = tk.Entry(parent, textvariable=var, width=6,
                         font=("Helvetica", 10), fg=color, justify="center")
            e.pack(side="left", padx=(0,2))
            tk.Label(parent, text="s", font=("Helvetica", 10)).pack(side="left")
            e.bind("<Return>",   lambda ev: cb(var))
            e.bind("<FocusOut>", lambda ev: cb(var))
            return var

        self._ti_var = _ef(ctrl, "Trim IN",  TRIM_COLOR, self._set_trim_in)
        self._to_var = _ef(ctrl, "Trim OUT", TRIM_COLOR, self._set_trim_out)
        self._fi_var = _ef(ctrl, "Fade IN",  FADE_COLOR, self._set_fade_in)
        self._fo_var = _ef(ctrl, "Fade OUT", FADE_COLOR, self._set_fade_out)
        tk.Button(ctrl, text="↺ Reset", font=("Helvetica", 10),
                  command=self.reset_handles).pack(side="right", padx=8)

        # Cursor readout
        self._cursor_lbl = tk.Label(self, text="", font=("Helvetica", 9),
                                     fg="#888", anchor="w")
        self._cursor_lbl.pack(fill="x", padx=4)

        # Events
        for cv in (self._ref_canvas, self._out_canvas):
            cv.bind("<Configure>",       self._on_resize)
            cv.bind("<ButtonPress-1>",   self._on_press)
            cv.bind("<B1-Motion>",       self._on_drag)
            cv.bind("<ButtonRelease-1>", self._on_release)
            cv.bind("<Motion>",          self._on_hover)
            cv.bind("<Leave>", lambda e: self._cursor_lbl.config(text=""))

    # ── entry callbacks ───────────────────────────────────────────────────────

    def _parse(self, var):
        try:    return max(0.0, float(var.get().replace("s","")))
        except: return None

    def _set_trim_in(self, var):
        v = self._parse(var)
        if v is None: return
        self._trim_in_s = min(v, self._trim_out_s - 0.001)
        self._fade_in_s = min(self._fade_in_s,
                              self._trim_out_s - self._trim_in_s - 0.001)
        self._sync_entries(); self._draw()
        if self.on_change: self.on_change()

    def _set_trim_out(self, var):
        v = self._parse(var)
        if v is None: return
        self._trim_out_s = max(v, self._trim_in_s + 0.001)
        self._fade_out_s = min(self._fade_out_s,
                               self._trim_out_s - self._trim_in_s - 0.001)
        self._sync_entries(); self._draw()
        if self.on_change: self.on_change()

    def _set_fade_in(self, var):
        v = self._parse(var)
        if v is None: return
        region = self._trim_out_s - self._trim_in_s
        self._fade_in_s = min(v, region - self._fade_out_s - 0.001)
        self._fade_in_s = max(0.0, self._fade_in_s)
        self._sync_entries(); self._draw()
        if self.on_change: self.on_change()

    def _set_fade_out(self, var):
        v = self._parse(var)
        if v is None: return
        region = self._trim_out_s - self._trim_in_s
        self._fade_out_s = min(v, region - self._fade_in_s - 0.001)
        self._fade_out_s = max(0.0, self._fade_out_s)
        self._sync_entries(); self._draw()
        if self.on_change: self.on_change()

    def _sync_entries(self):
        self._ti_var.set(f"{self._trim_in_s:.2f}")
        self._to_var.set(f"{self._trim_out_s:.2f}")
        self._fi_var.set(f"{self._fade_in_s:.2f}")
        self._fo_var.set(f"{self._fade_out_s:.2f}")

    # ── public API ────────────────────────────────────────────────────────────

    def set_data(self, ref_path, out_path, ref_dur, out_dur):
        self._ref_dur    = max(ref_dur, 0.001)
        self._out_dur    = max(out_dur, 0.001)
        self._total_dur  = max(self._ref_dur, self._out_dur)
        self._trim_in_s  = 0.0
        self._trim_out_s = self._out_dur
        self._fade_in_s  = 0.0
        self._fade_out_s = 0.0
        self._ref_ph_s   = 0.0
        self._out_ph_s   = 0.0
        self._sync_entries()
        threading.Thread(target=self._load_wf,
                         args=(ref_path, out_path), daemon=True).start()

    def _load_wf(self, ref_path, out_path):
        try:    self._ref_wf = build_waveform(ref_path)
        except: self._ref_wf = None
        try:    self._out_wf = build_waveform(out_path)
        except: self._out_wf = None
        self.after(0, self._draw)

    def clear(self):
        self._ref_wf = None; self._out_wf = None
        for cv in (self._ref_canvas, self._out_canvas, self._ruler):
            cv.delete("all")
        self._cursor_lbl.config(text="")

    def set_ref_playhead(self, seconds):
        self._ref_ph_s = seconds
        self._draw_playheads_only()

    def set_out_playhead(self, seconds):
        self._out_ph_s = seconds
        self._draw_playheads_only()

    # ── drawing ───────────────────────────────────────────────────────────────

    def _on_resize(self, event):
        self._width = max(event.width, 100)
        self._draw()

    def _draw(self):
        self._draw_canvas(self._ref_canvas, self._ref_wf,
                          REF_COLOR, self._ref_dur, is_output=False)
        self._draw_canvas(self._out_canvas, self._out_wf,
                          OUT_COLOR, self._out_dur, is_output=True)
        self._draw_ruler()
        self._draw_playheads_only()

    def _draw_canvas(self, cv, wf, color, file_dur, is_output):
        cv.delete("all")
        w  = cv.winfo_width() or self._width
        h  = CANVAS_H
        mid = h // 2

        end_x = self._s2x(file_dur)   # where this file ends in pixels

        # Region beyond file end
        if end_x < w:
            cv.create_rectangle(end_x, 0, w, h, fill="#d8d8d8", outline="")
            cv.create_line(end_x, 0, end_x, h, fill="#999", width=1, dash=(4,3))
            cv.create_text(end_x+4, h//2, text="end", anchor="w",
                           fill="#999", font=("Helvetica", 8))

        # Grey trim-out region (before end_x)
        ti_x = self._s2x(self._trim_in_s)
        to_x = self._s2x(self._trim_out_s)
        to_x = min(to_x, end_x)
        cv.create_rectangle(0,    0, ti_x, h, fill="#e0e0e0", outline="")
        cv.create_rectangle(to_x, 0, end_x, h, fill="#e0e0e0", outline="")

        # Waveform (only within file bounds)
        if wf is not None:
            pos, neg = wf
            n   = len(pos)
            amp = mid - CANVAS_PAD
            pts_top, pts_bot = [], []
            for i in range(n):
                # map bin i to seconds within the file
                t = i / n * file_dur
                x = self._s2x(t)
                pts_top.append((x, mid - int(pos[i] * amp)))
                pts_bot.append((x, mid - int(neg[i] * amp)))
            poly = pts_top + pts_bot[::-1]
            flat = [c for pt in poly for c in pt]
            if len(flat) >= 4:
                cv.create_polygon(flat, fill=color, outline="")
        cv.create_line(0, mid, end_x, mid, fill="#ccc", width=1)

        # Fade shading (output only)
        if is_output:
            fi_x = self._s2x(self._trim_in_s  + self._fade_in_s)
            fo_x = self._s2x(self._trim_out_s - self._fade_out_s)
            if fi_x > ti_x:
                cv.create_rectangle(ti_x, 0, fi_x, h,
                                    fill=FADE_COLOR, outline="", stipple="gray50")
            if fo_x < to_x:
                cv.create_rectangle(fo_x, 0, to_x, h,
                                    fill=FADE_COLOR, outline="", stipple="gray50")

        if is_output:
            # Trim lines — full height
            cv.create_line(ti_x, 0, ti_x, h, fill=TRIM_COLOR, width=2)
            cv.create_line(to_x, 0, to_x, h, fill=TRIM_COLOR, width=2)
            # Trim tabs at BOTTOM corners
            self._draw_tab(cv, ti_x, "IN",  TRIM_COLOR, facing="right",  bottom=True, h=h)
            self._draw_tab(cv, to_x, "OUT", TRIM_COLOR, facing="left",   bottom=True, h=h)
            # Fade lines + tabs at top
            fi_abs = self._s2x(self._trim_in_s  + self._fade_in_s)
            fo_abs = self._s2x(self._trim_out_s - self._fade_out_s)
            cv.create_line(fi_abs, 0, fi_abs, h, fill=FADE_COLOR, width=2, dash=(5,3))
            cv.create_line(fo_abs, 0, fo_abs, h, fill=FADE_COLOR, width=2, dash=(5,3))
            self._draw_tab(cv, fi_abs, "FI", FADE_COLOR, facing="right", small=True, h=h)
            self._draw_tab(cv, fo_abs, "FO", FADE_COLOR, facing="left",  small=True, h=h)
        else:
            # Reference: just draw trim lines with no tabs (visual guide only)
            cv.create_line(ti_x, 0, ti_x, h, fill=TRIM_COLOR, width=1, dash=(3,3))
            cv.create_line(to_x, 0, to_x, h, fill=TRIM_COLOR, width=1, dash=(3,3))

        # Duration label
        cv.create_text(min(end_x-4, w-4), 4,
                       text=f"{file_dur:.2f}s", anchor="ne",
                       fill="#666", font=("Helvetica", 8))

    def _draw_tab(self, cv, x, label, color, facing="right", small=False,
                  bottom=False, h=CANVAS_H):
        tw = TAB_W if not small else 10
        th = TAB_H if not small else 14
        x0, x1 = (x, x+tw) if facing == "right" else (x-tw, x)
        if bottom:
            y0, y1 = h - th, h      # anchor to bottom
        else:
            y0, y1 = 0, th          # anchor to top
        cv.create_rectangle(x0, y0, x1, y1, fill=color, outline="white", width=1)
        cv.create_text((x0+x1)//2, (y0+y1)//2, text=label, fill="white",
                       font=("Helvetica", 7 if small else 8, "bold"), anchor="center")

    def _draw_ruler(self):
        cv  = self._ruler
        cv.delete("all")
        w   = cv.winfo_width() or self._width
        dur = self._total_dur
        for interval in [0.05,0.1,0.25,0.5,1,2,5,10,30,60]:
            if dur / interval <= 25: break
        t = 0.0
        while t <= dur + 1e-6:
            x = self._s2x(t)
            cv.create_line(x, 0, x, 8, fill="#666")
            label = (f"{t:.2f}" if dur < 2 else
                     f"{t:.1f}" if dur < 15 else f"{int(t)}s")
            cv.create_text(x, 14, text=label, fill="#444",
                           font=("Helvetica", 8), anchor="center")
            t = round(t + interval, 6)
        # Shade beyond the shorter file
        min_dur = min(self._ref_dur, self._out_dur)
        min_x   = self._s2x(min_dur)
        if min_x < w:
            cv.create_rectangle(min_x, 0, w, 20, fill="#d0d0d0", outline="")

    def _draw_playheads_only(self):
        self._ref_canvas.delete("playhead")
        x = self._s2x(self._ref_ph_s)
        self._ref_canvas.create_line(x, 0, x, CANVAS_H,
                                     fill=PLAYHEAD_COL, width=2, tags="playhead")
        self._out_canvas.delete("playhead")
        x = self._s2x(self._out_ph_s)
        self._out_canvas.create_line(x, 0, x, CANVAS_H,
                                     fill=PLAYHEAD_COL, width=2, tags="playhead")

    # ── interaction ───────────────────────────────────────────────────────────

    def _handle_positions_s(self):
        """All handle positions in seconds."""
        return {
            "ti": self._trim_in_s,
            "to": self._trim_out_s,
            "fi": self._trim_in_s  + self._fade_in_s,
            "fo": self._trim_out_s - self._fade_out_s,
        }

    def _hit_handle(self, x, y, canvas):
        # All handles are output-canvas only
        if canvas is not self._out_canvas:
            return None
        pos = self._handle_positions_s()
        h   = CANVAS_H
        mid = h // 2

        # Hard y split: top half → fade handles, bottom half → trim handles
        # This gives unambiguous selection regardless of x overlap.
        if y <= mid:
            # Top half — only check fade handles
            for name, s in [("fi", pos["fi"]), ("fo", pos["fo"])]:
                if abs(x - self._s2x(s)) <= HIT_TOL:
                    return name
        else:
            # Bottom half — only check trim handles
            for name, s in [("ti", pos["ti"]), ("to", pos["to"])]:
                if abs(x - self._s2x(s)) <= HIT_TOL:
                    return name
        return None

    def _on_hover(self, event):
        cv     = event.widget
        handle = self._hit_handle(event.x, event.y, cv)
        cv.configure(cursor="sb_h_double_arrow" if handle else "crosshair")
        t      = self._x2s(event.x)
        hint   = {"ti":"Trim IN","to":"Trim OUT","fi":"Fade IN","fo":"Fade OUT"}
        label  = hint.get(handle, "")
        self._cursor_lbl.config(
            text=f"{label}  {t:.3f}s" if label else f"{t:.3f}s")

    def _on_press(self, event):
        self._drag    = self._hit_handle(event.x, event.y, event.widget)
        self._drag_cv = event.widget

    def _on_drag(self, event):
        if not self._drag: return
        s = self._x2s(event.x)

        if self._drag == "ti":
            self._trim_in_s  = max(0.0, min(s, self._trim_out_s - 0.001))
            self._fade_in_s  = min(self._fade_in_s,
                                   self._trim_out_s - self._trim_in_s - 0.001)

        elif self._drag == "to":
            self._trim_out_s = max(s, self._trim_in_s + 0.001)
            self._trim_out_s = min(self._trim_out_s, self._out_dur)
            self._fade_out_s = min(self._fade_out_s,
                                   self._trim_out_s - self._trim_in_s - 0.001)

        elif self._drag == "fi":
            max_fi = self._trim_out_s - self._trim_in_s - self._fade_out_s - 0.001
            self._fade_in_s = max(0.0, min(s - self._trim_in_s, max_fi))

        elif self._drag == "fo":
            max_fo = self._trim_out_s - self._trim_in_s - self._fade_in_s - 0.001
            self._fade_out_s = max(0.0, min(self._trim_out_s - s, max_fo))

        self._sync_entries()
        self._draw()
        if self.on_change: self.on_change()

    def _on_release(self, event):
        self._drag = None

# ── Single-file waveform panel ─────────────────────────────────────────────────

class SingleFilePanel(tk.Frame):
    """
    One waveform canvas with trim/fade handles, ruler, and numeric entries.
    Mirrors WaveformPanel but for a single file (no reference track).
    """

    def __init__(self, parent, on_change=None, on_drag_start=None, on_drag_end=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_change     = on_change
        self.on_drag_start = on_drag_start
        self.on_drag_end   = on_drag_end

        self._dur        = 1.0
        self._wf         = None
        self._trim_in_s  = 0.0
        self._trim_out_s = 1.0
        self._fade_in_s  = 0.0
        self._fade_out_s = 0.0
        self._ph_s       = 0.0
        self._drag       = None
        self._width      = 600

        self._build()

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def trim_in(self):      return self._trim_in_s
    @property
    def trim_out(self):     return self._trim_out_s
    @property
    def fade_in_dur(self):  return self._fade_in_s
    @property
    def fade_out_dur(self): return self._fade_out_s

    def reset_handles(self):
        self._trim_in_s  = 0.0
        self._trim_out_s = self._dur
        self._fade_in_s  = 0.0
        self._fade_out_s = 0.0
        self._sync_entries()
        self._draw()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _s2x(self, s):
        return int(s / self._dur * self._width) if self._dur > 0 else 0

    def _x2s(self, x):
        return max(0.0, x / self._width * self._dur) if self._width > 0 else 0.0

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self):
        leg = tk.Frame(self)
        leg.pack(fill="x", pady=(0, 2))
        tk.Label(leg, text="● Waveform", font=("Helvetica", 10, "bold"),
                 fg=OUT_COLOR).pack(side="left", padx=4)
        tk.Label(leg, text="▐ Trim IN/OUT", font=("Helvetica", 10),
                 fg=TRIM_COLOR).pack(side="left", padx=12)
        tk.Label(leg, text="┊ Fade IN/OUT", font=("Helvetica", 10),
                 fg=FADE_COLOR).pack(side="left", padx=12)

        self._canvas = tk.Canvas(self, height=CANVAS_H, bg="#eef3fb",
                                 highlightthickness=1, highlightbackground="#bbb")
        self._canvas.pack(fill="x", pady=(0, 1))

        self._ruler = tk.Canvas(self, height=20, bg="#e0e0e0",
                                highlightthickness=0)
        self._ruler.pack(fill="x")

        ctrl = tk.Frame(self, pady=6)
        ctrl.pack(fill="x")

        def _ef(parent, label, color, cb):
            tk.Label(parent, text=label, font=("Helvetica", 10),
                     fg=color).pack(side="left", padx=(8, 2))
            var = tk.StringVar(value="0.00")
            e = tk.Entry(parent, textvariable=var, width=6,
                         font=("Helvetica", 10), fg=color, justify="center")
            e.pack(side="left", padx=(0, 2))
            tk.Label(parent, text="s", font=("Helvetica", 10)).pack(side="left")
            e.bind("<Return>",   lambda ev: cb(var))
            e.bind("<FocusOut>", lambda ev: cb(var))
            return var

        self._ti_var = _ef(ctrl, "Trim IN",  TRIM_COLOR, self._set_trim_in)
        self._to_var = _ef(ctrl, "Trim OUT", TRIM_COLOR, self._set_trim_out)
        self._fi_var = _ef(ctrl, "Fade IN",  FADE_COLOR, self._set_fade_in)
        self._fo_var = _ef(ctrl, "Fade OUT", FADE_COLOR, self._set_fade_out)
        tk.Button(ctrl, text="↺ Reset", font=("Helvetica", 10),
                  command=self.reset_handles).pack(side="right", padx=8)

        self._cursor_lbl = tk.Label(self, text="", font=("Helvetica", 9),
                                    fg="#888", anchor="w")
        self._cursor_lbl.pack(fill="x", padx=4)

        self._canvas.bind("<Configure>",       self._on_resize)
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>",          self._on_hover)
        self._canvas.bind("<Leave>", lambda e: self._cursor_lbl.config(text=""))

    # ── entry callbacks ───────────────────────────────────────────────────────

    def _parse(self, var):
        try:    return max(0.0, float(var.get().replace("s", "")))
        except: return None

    def _set_trim_in(self, var):
        v = self._parse(var)
        if v is None: return
        self._trim_in_s = min(v, self._trim_out_s - 0.001)
        self._fade_in_s = min(self._fade_in_s,
                              self._trim_out_s - self._trim_in_s - 0.001)
        self._sync_entries(); self._draw()
        if self.on_change: self.on_change()

    def _set_trim_out(self, var):
        v = self._parse(var)
        if v is None: return
        self._trim_out_s = max(v, self._trim_in_s + 0.001)
        self._fade_out_s = min(self._fade_out_s,
                               self._trim_out_s - self._trim_in_s - 0.001)
        self._sync_entries(); self._draw()
        if self.on_change: self.on_change()

    def _set_fade_in(self, var):
        v = self._parse(var)
        if v is None: return
        region = self._trim_out_s - self._trim_in_s
        self._fade_in_s = max(0.0, min(v, region - self._fade_out_s - 0.001))
        self._sync_entries(); self._draw()
        if self.on_change: self.on_change()

    def _set_fade_out(self, var):
        v = self._parse(var)
        if v is None: return
        region = self._trim_out_s - self._trim_in_s
        self._fade_out_s = max(0.0, min(v, region - self._fade_in_s - 0.001))
        self._sync_entries(); self._draw()
        if self.on_change: self.on_change()

    def _sync_entries(self):
        self._ti_var.set(f"{self._trim_in_s:.2f}")
        self._to_var.set(f"{self._trim_out_s:.2f}")
        self._fi_var.set(f"{self._fade_in_s:.2f}")
        self._fo_var.set(f"{self._fade_out_s:.2f}")

    # ── public API ────────────────────────────────────────────────────────────

    def set_data(self, path, dur):
        self._dur        = max(dur, 0.001)
        self._trim_in_s  = 0.0
        self._trim_out_s = self._dur
        self._fade_in_s  = 0.0
        self._fade_out_s = 0.0
        self._ph_s       = 0.0
        self._sync_entries()
        threading.Thread(target=self._load_wf, args=(path,), daemon=True).start()

    def _load_wf(self, path):
        try:    self._wf = build_waveform(path)
        except: self._wf = None
        self.after(0, self._draw)

    def clear(self):
        self._wf = None
        self._canvas.delete("all")
        self._ruler.delete("all")
        self._cursor_lbl.config(text="")

    def set_playhead(self, seconds):
        self._ph_s = seconds
        self._canvas.delete("playhead")
        x = self._s2x(self._ph_s)
        self._canvas.create_line(x, 0, x, CANVAS_H,
                                 fill=PLAYHEAD_COL, width=2, tags="playhead")

    # ── drawing ───────────────────────────────────────────────────────────────

    def _on_resize(self, event):
        self._width = max(event.width, 100)
        self._draw()

    def _draw(self):
        cv  = self._canvas
        cv.delete("all")
        w   = cv.winfo_width() or self._width
        h   = CANVAS_H
        mid = h // 2

        ti_x = self._s2x(self._trim_in_s)
        to_x = self._s2x(self._trim_out_s)

        cv.create_rectangle(0,    0, ti_x, h, fill="#e0e0e0", outline="")
        cv.create_rectangle(to_x, 0, w,    h, fill="#e0e0e0", outline="")

        if self._wf is not None:
            pos, neg = self._wf
            n   = len(pos)
            amp = mid - CANVAS_PAD
            pts_top, pts_bot = [], []
            for i in range(n):
                t = i / n * self._dur
                x = self._s2x(t)
                pts_top.append((x, mid - int(pos[i] * amp)))
                pts_bot.append((x, mid - int(neg[i] * amp)))
            poly = pts_top + pts_bot[::-1]
            flat = [c for pt in poly for c in pt]
            if len(flat) >= 4:
                cv.create_polygon(flat, fill=OUT_COLOR, outline="")
        cv.create_line(0, mid, w, mid, fill="#ccc", width=1)

        fi_x = self._s2x(self._trim_in_s  + self._fade_in_s)
        fo_x = self._s2x(self._trim_out_s - self._fade_out_s)
        if fi_x > ti_x:
            cv.create_rectangle(ti_x, 0, fi_x, h,
                                fill=FADE_COLOR, outline="", stipple="gray50")
        if fo_x < to_x:
            cv.create_rectangle(fo_x, 0, to_x, h,
                                fill=FADE_COLOR, outline="", stipple="gray50")

        cv.create_line(ti_x, 0, ti_x, h, fill=TRIM_COLOR, width=2)
        cv.create_line(to_x, 0, to_x, h, fill=TRIM_COLOR, width=2)

        # Reuse WaveformPanel tab drawing helper inline
        for x, label, facing, bottom in [
            (ti_x, "IN",  "right", True),
            (to_x, "OUT", "left",  True),
            (fi_x, "FI",  "right", False),
            (fo_x, "FO",  "left",  False),
        ]:
            tw = TAB_W if label in ("IN", "OUT") else 10
            th = TAB_H if label in ("IN", "OUT") else 14
            small = label in ("FI", "FO")
            x0, x1 = (x, x+tw) if facing == "right" else (x-tw, x)
            y0, y1 = (h-th, h) if bottom else (0, th)
            cv.create_rectangle(x0, y0, x1, y1,
                                fill=TRIM_COLOR if not small else FADE_COLOR,
                                outline="white", width=1)
            cv.create_text((x0+x1)//2, (y0+y1)//2, text=label, fill="white",
                           font=("Helvetica", 7 if small else 8, "bold"),
                           anchor="center")

        cv.create_line(fi_x, 0, fi_x, h, fill=FADE_COLOR, width=2, dash=(5, 3))
        cv.create_line(fo_x, 0, fo_x, h, fill=FADE_COLOR, width=2, dash=(5, 3))

        cv.create_text(w - 4, 4, text=f"{self._dur:.2f}s", anchor="ne",
                       fill="#666", font=("Helvetica", 8))

        # Ruler
        rc = self._ruler
        rc.delete("all")
        dur = self._dur
        for interval in [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60]:
            if dur / interval <= 25: break
        t = 0.0
        while t <= dur + 1e-6:
            x = self._s2x(t)
            rc.create_line(x, 0, x, 8, fill="#666")
            label = (f"{t:.2f}" if dur < 2 else
                     f"{t:.1f}" if dur < 15 else f"{int(t)}s")
            rc.create_text(x, 14, text=label, fill="#444",
                           font=("Helvetica", 8), anchor="center")
            t = round(t + interval, 6)

        self.set_playhead(self._ph_s)

    # ── interaction ───────────────────────────────────────────────────────────

    def _handle_positions_s(self):
        return {
            "ti": self._trim_in_s,
            "to": self._trim_out_s,
            "fi": self._trim_in_s  + self._fade_in_s,
            "fo": self._trim_out_s - self._fade_out_s,
        }

    def _hit_handle(self, x, y):
        pos = self._handle_positions_s()
        mid = CANVAS_H // 2
        if y <= mid:
            for name, s in [("fi", pos["fi"]), ("fo", pos["fo"])]:
                if abs(x - self._s2x(s)) <= HIT_TOL:
                    return name
        else:
            for name, s in [("ti", pos["ti"]), ("to", pos["to"])]:
                if abs(x - self._s2x(s)) <= HIT_TOL:
                    return name
        return None

    def _on_hover(self, event):
        handle = self._hit_handle(event.x, event.y)
        self._canvas.configure(
            cursor="sb_h_double_arrow" if handle else "crosshair")
        t     = self._x2s(event.x)
        hint  = {"ti": "Trim IN", "to": "Trim OUT",
                 "fi": "Fade IN", "fo": "Fade OUT"}
        label = hint.get(handle, "")
        self._cursor_lbl.config(
            text=f"{label}  {t:.3f}s" if label else f"{t:.3f}s")

    def _on_press(self, event):
        self._drag = self._hit_handle(event.x, event.y)
        if self._drag and self.on_drag_start:
            self.on_drag_start()

    def _on_drag(self, event):
        if not self._drag: return
        s = self._x2s(event.x)
        if self._drag == "ti":
            self._trim_in_s = max(0.0, min(s, self._trim_out_s - 0.001))
            self._fade_in_s = min(self._fade_in_s,
                                  self._trim_out_s - self._trim_in_s - 0.001)
        elif self._drag == "to":
            self._trim_out_s = max(s, self._trim_in_s + 0.001)
            self._trim_out_s = min(self._trim_out_s, self._dur)
            self._fade_out_s = min(self._fade_out_s,
                                   self._trim_out_s - self._trim_in_s - 0.001)
        elif self._drag == "fi":
            max_fi = self._trim_out_s - self._trim_in_s - self._fade_out_s - 0.001
            self._fade_in_s = max(0.0, min(s - self._trim_in_s, max_fi))
        elif self._drag == "fo":
            max_fo = self._trim_out_s - self._trim_in_s - self._fade_in_s - 0.001
            self._fade_out_s = max(0.0, min(self._trim_out_s - s, max_fo))
        self._sync_entries()
        self._draw()
        if self.on_change: self.on_change()

    def _on_release(self, event):
        self._drag = None
        if self.on_drag_end:
            self.on_drag_end()


# ── Main App ───────────────────────────────────────────────────────────────────

PAD = 16

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Level Matcher")
        self.geometry("1400x1000")
        self.minsize(1100, 800)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.game_a_var = tk.StringVar()
        self.game_b_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select folders above, then click Scan & Preview.")
        self.progress   = tk.DoubleVar(value=0.0)
        self.loop_var   = tk.BooleanVar(value=False)
        self.trim_var   = tk.DoubleVar(value=0.0)

        self._running         = False
        self._row_ids         = []
        self._selected        = None
        self._trim_after_id   = None
        self._preview_playing = False

        # left-panel players (matched workflow)
        self._player_ref = Player()
        self._player_out = Player()
        self._player_ref.on_stop = lambda: self.after(0, self._on_ref_stopped)
        self._player_out.on_stop = lambda: self.after(0, self._on_out_stopped)

        # ── Session timeline state ─────────────────────────────────────────
        self._session_blocks   = []          # list[SessionBlock]
        self._sel_block        = None        # primary selected block (for editor panel)
        self._sel_blocks       = set()       # all selected blocks (multi-select)
        self._color_idx        = 0           # next color to assign
        self._pps              = DEFAULT_PPS # pixels per second (zoom)
        self._tl_start_pos     = 0.0         # where playback will start (seconds)
        self._tl_playing       = False
        self._tl_play_ts       = None        # time.monotonic() when playback started
        self._tl_procs         = []          # active Popen processes
        self._tl_timers        = []          # threading.Timer for future starts
        self._tl_muted_tracks  = set()       # set of track indices that are muted
        self._tl_soloed_tracks = set()       # set of track indices that are soloed
        self._wf_cache         = {}          # file_abs → np.array peaks | _WF_LOADING
        self._pending_bids     = set()       # block bids with unapplied audio changes
        self._undo_stack       = []          # list of session snapshots
        self._redo_stack       = []
        self._undo_committed   = True        # False while inside a continuous drag
        self._tl_lbl_drag_src  = None        # track index being reordered via label drag
        self._tl_lbl_drag_y    = 0           # current cursor y during label drag
        self._tl_drag_block    = None
        self._tl_drag_orig_sec = 0.0
        self._tl_drag_orig_trk = 0
        self._tl_drag_start_cx = 0.0
        self._tl_drag_start_cy = 0.0
        self._tl_drag_all_orig = {}   # bid → (start_sec, track) for multi-drag
        self._tl_drag_copy     = False  # True when Option+drag should clone on first motion
        self._scroll_target    = None   # canvas currently under the cursor
        self._tl_rubber        = None # (cx0,cy0,cx1,cy1) rubber-band in canvas coords
        # waveform editor state for selected block
        self._blk_trim_var     = tk.DoubleVar(value=0.0)
        self._blk_trim_after   = None

        self._build()
        self._poll_playback()

    def _setup_native_scroll(self):
        """macOS: intercept trackpad scroll via NSEvent before the OS routes it.
        tkinter's <MouseWheel> binding never fires for bare NSView (Canvas)."""
        import sys
        if sys.platform != "darwin":
            return
        try:
            from AppKit import NSEvent
        except ImportError:
            return

        def _cb(ev):
            if self._scroll_target != "tl":
                return ev
            try:
                dy = ev.scrollingDeltaY()
                dx = ev.scrollingDeltaX()
                if abs(dy) >= abs(dx) and dy != 0:
                    bbox = self._tl_cv.bbox("all")
                    total_h = max((bbox[3] - bbox[1]) if bbox else self._tl_cv.winfo_height(), 1)
                    y0, _ = self._tl_cv.yview()
                    self._tl_cv.yview_moveto(max(0.0, min(1.0, y0 - dy / total_h)))
                elif abs(dx) > abs(dy) and dx != 0:
                    bbox = self._tl_cv.bbox("all")
                    total_w = max((bbox[2] - bbox[0]) if bbox else self._tl_cv.winfo_width(), 1)
                    x0, _ = self._tl_cv.xview()
                    self._tl_cv.xview_moveto(max(0.0, min(1.0, x0 - dx / total_w)))
            except Exception:
                pass
            return ev

        self._ns_scroll_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            1 << 22, _cb)   # 1<<22 = NSEventTypeScrollWheel

    def _on_close(self):
        if getattr(self, "_ns_scroll_monitor", None) is not None:
            try:
                from AppKit import NSEvent
                NSEvent.removeMonitor_(self._ns_scroll_monitor)
            except Exception:
                pass
        self._player_ref.stop(save_position=False)
        self._player_out.stop(save_position=False)
        self._tl_stop()
        _cleanup_temp()
        self.destroy()

    # ── Layout ─────────────────────────────────────────────────────────────

    def _sep(self, parent, h=8):
        tk.Frame(parent, height=h).pack(fill="x")

    def _scrollable(self, parent):
        """Return a scrollable frame child."""
        container = tk.Frame(parent)
        container.pack(fill="both", expand=True)
        cs = tk.Canvas(container, highlightthickness=0)
        sb = ttk.Scrollbar(container, orient="vertical", command=cs.yview)
        cs.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cs.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cs, padx=PAD, pady=PAD)
        win_id = cs.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: cs.configure(scrollregion=cs.bbox("all")))
        cs.bind("<Configure>",
                lambda e: cs.itemconfig(win_id, width=e.width))
        self._left_cs = cs   # saved for global scroll routing
        return inner

    def _build(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left_host  = tk.Frame(paned)
        right_host = tk.Frame(paned)
        paned.add(left_host,  weight=1)
        paned.add(right_host, weight=3)

        left = self._scrollable(left_host)
        self._build_content(left)
        self._build_session_panel(right_host)
        self.after(50, lambda: paned.sashpos(0, 430))

    def _build_content(self, outer):
        tk.Label(outer, text="Audio Level Matcher",
                 font=("Helvetica", 17, "bold"), anchor="w").pack(fill="x")
        tk.Label(outer, text="Match Game B audio levels to Game A reference files",
                 font=("Helvetica", 11), fg="#666", anchor="w").pack(fill="x")
        self._sep(outer, 12)

        for label, var, cmd in [
            ("Game A  (reference):", self.game_a_var, self._browse_a),
            ("Game B  (to match):",  self.game_b_var, self._browse_b),
            ("Output folder:",       self.output_var, self._browse_out),
        ]:
            row = tk.Frame(outer)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, width=22, anchor="w",
                     font=("Helvetica", 11)).pack(side="left")
            tk.Button(row, text="Browse", width=9, font=("Helvetica", 11),
                      command=cmd).pack(side="right")
            tk.Entry(row, textvariable=var,
                     font=("Helvetica", 11)).pack(side="left", fill="x",
                                                   expand=True, padx=6)
        self._sep(outer, 10)

        btn_row = tk.Frame(outer)
        btn_row.pack(fill="x")
        tk.Button(btn_row, text="🔍  Scan & Preview",
                  font=("Helvetica", 12, "bold"), padx=14, pady=6,
                  command=self._scan).pack(side="left", padx=(0,10))
        self._run_btn = tk.Button(btn_row, text="▶  Process Files",
                                  font=("Helvetica", 12, "bold"),
                                  padx=14, pady=6, state="disabled",
                                  command=self._process)
        self._run_btn.pack(side="left")
        self._sep(outer, 8)

        tk.Label(outer, text="File Preview",
                 font=("Helvetica", 11, "bold"), anchor="w").pack(fill="x")
        self._sep(outer, 4)

        table_frame = tk.Frame(outer, bd=1, relief="sunken")
        table_frame.pack(fill="both", expand=True)

        cols = ("file","duration","strategy","ref_level","src_level","gain","status")
        self.tree = ttk.Treeview(table_frame, columns=cols,
                                 show="headings", height=8)
        for col, head, w, anch in [
            ("file",      "File",       210, "w"),
            ("duration",  "Duration",    72, "center"),
            ("strategy",  "Strategy",    90, "center"),
            ("ref_level", "Ref Level",  110, "center"),
            ("src_level", "Src Level",  110, "center"),
            ("gain",      "Gain (dB)",   82, "center"),
            ("status",    "Status",     110, "center"),
        ]:
            self.tree.heading(col, text=head)
            self.tree.column(col, width=w, anchor=anch, minwidth=40)

        vsb = ttk.Scrollbar(table_frame, orient="vertical",
                            command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

        for tag, col in [("ok","#1a7a1a"),("warning","#b87800"),
                         ("large","#cc3300"),("done","#1a7a1a"),("error","#cc3300")]:
            self.tree.tag_configure(tag, foreground=col)
        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)

        self._sep(outer, 8)

        # ── Player panel ──────────────────────────────────────────────────
        player_frame = tk.Frame(outer, bd=1, relief="groove", padx=10, pady=8)
        player_frame.pack(fill="x")

        hdr = tk.Frame(player_frame)
        hdr.pack(fill="x")
        tk.Label(hdr, text="▶ Playback", font=("Helvetica", 11, "bold"),
                 anchor="w").pack(side="left")
        tk.Checkbutton(hdr, text="Loop  (resumes from last position)",
                       variable=self.loop_var,
                       font=("Helvetica", 11)).pack(side="right")
        self._sep(player_frame, 6)

        ref_row = tk.Frame(player_frame)
        ref_row.pack(fill="x", pady=2)
        tk.Label(ref_row, text="Reference:", width=12, anchor="w",
                 font=("Helvetica", 11)).pack(side="left")
        self._ref_name_lbl = tk.Label(ref_row, text="—", anchor="w",
                                       font=("Helvetica", 11), fg="#444")
        self._ref_name_lbl.pack(side="left", fill="x", expand=True)
        self._ref_time_lbl = tk.Label(ref_row, text="0:00", width=6,
                                       font=("Helvetica", 10), fg="#666")
        self._ref_time_lbl.pack(side="right")
        self._ref_play_btn = tk.Button(ref_row, text="▶ Play", width=9,
                                        font=("Helvetica", 11), state="disabled",
                                        command=self._toggle_ref)
        self._ref_play_btn.pack(side="right", padx=(8,4))

        out_row = tk.Frame(player_frame)
        out_row.pack(fill="x", pady=2)
        tk.Label(out_row, text="Output:", width=12, anchor="w",
                 font=("Helvetica", 11)).pack(side="left")
        self._out_name_lbl = tk.Label(out_row, text="—", anchor="w",
                                       font=("Helvetica", 11), fg="#444")
        self._out_name_lbl.pack(side="left", fill="x", expand=True)
        self._out_time_lbl = tk.Label(out_row, text="0:00", width=6,
                                       font=("Helvetica", 10), fg="#666")
        self._out_time_lbl.pack(side="right")
        self._out_play_btn = tk.Button(out_row, text="▶ Play", width=9,
                                        font=("Helvetica", 11), state="disabled",
                                        command=self._toggle_out)
        self._out_play_btn.pack(side="right", padx=(8,4))

        self._sep(player_frame, 8)

        # Trim slider
        trim_row = tk.Frame(player_frame)
        trim_row.pack(fill="x", pady=2)
        tk.Label(trim_row, text="Gain trim:", width=12, anchor="w",
                 font=("Helvetica", 11)).pack(side="left")
        self._trim_slider = tk.Scale(trim_row, from_=-12.0, to=12.0,
                                     resolution=0.1, orient="horizontal",
                                     variable=self.trim_var,
                                     font=("Helvetica", 10), length=280,
                                     showvalue=False, state="disabled",
                                     command=self._on_trim_drag)
        self._trim_slider.pack(side="left", padx=(0,8))
        self._trim_lbl = tk.Label(trim_row, text="0.0 dB", width=8,
                                   font=("Helvetica", 11, "bold"), fg="#333")
        self._trim_lbl.pack(side="left")
        self._preview_lbl = tk.Label(trim_row, text="", width=10,
                                      font=("Helvetica", 10), fg="#888")
        self._preview_lbl.pack(side="left", padx=(8,0))

        apply_row = tk.Frame(player_frame)
        apply_row.pack(fill="x", pady=(6,2))
        tk.Label(apply_row, text="", width=12).pack(side="left")
        self._apply_btn = tk.Button(apply_row, text="✓ Apply & Save",
                                     font=("Helvetica", 11, "bold"),
                                     padx=12, state="disabled",
                                     command=self._apply_all)
        self._apply_btn.pack(side="left", padx=(0,8))
        self._reset_btn = tk.Button(apply_row, text="↺ Reset all",
                                     font=("Helvetica", 11),
                                     padx=10, state="disabled",
                                     command=self._reset_all)
        self._reset_btn.pack(side="left")

        self._sep(outer, 8)

        # ── Waveform panel ────────────────────────────────────────────────
        tk.Label(outer, text="Waveform  —  drag handles to trim/fade  (output only)",
                 font=("Helvetica", 11, "bold"), anchor="w").pack(fill="x")
        self._sep(outer, 4)

        self._wf_panel = WaveformPanel(outer, on_change=self._on_wf_change,
                                       bd=1, relief="sunken")
        self._wf_panel.pack(fill="x", pady=(0,8))

        self._sep(outer, 6)

        ttk.Progressbar(outer, variable=self.progress,
                        maximum=100, mode="determinate").pack(fill="x")
        self._sep(outer, 4)
        tk.Label(outer, textvariable=self.status_var,
                 font=("Helvetica", 10), fg="#555", anchor="w").pack(fill="x")

    # ── Folder browsing ────────────────────────────────────────────────────

    def _browse_a(self):
        d = filedialog.askdirectory(title="Select Game A audio folder")
        if d:
            self.game_a_var.set(d)
            if not self.output_var.get():
                self.output_var.set(str(Path(d).parent / "GameB_matched"))

    def _browse_b(self):
        d = filedialog.askdirectory(title="Select Game B audio folder")
        if d: self.game_b_var.set(d)

    def _browse_out(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d: self.output_var.set(d)

    # ── Scan ───────────────────────────────────────────────────────────────

    def _scan(self):
        a, b = self.game_a_var.get().strip(), self.game_b_var.get().strip()
        if not a or not b:
            messagebox.showwarning("Missing folders",
                                   "Please select both Game A and Game B folders.")
            return
        if not os.path.isdir(a) or not os.path.isdir(b):
            messagebox.showerror("Invalid folders",
                                 "One or both folders don't exist.")
            return
        self._stop_all(save_position=False)
        self.status_var.set("Scanning…"); self.update()
        pairs = match_files(a, b)
        self.tree.delete(*self.tree.get_children())
        self._row_ids = []
        self._selected = None
        self._clear_player_panel()
        self._wf_panel.clear()
        if not pairs:
            self.status_var.set("No matching filenames found.")
            self._run_btn.configure(state="disabled"); return
        total = len(pairs); self.progress.set(0)
        for i, (rel, ap, bp) in enumerate(pairs):
            self.status_var.set(f"Analysing {i+1}/{total}: {Path(rel).name}")
            self.update()
            try:
                ref, src = analyze(ap), analyze(bp)
                gain, _ = compute_gain_db(ref, src)
                tag = "ok" if abs(gain)<6 else ("warning" if abs(gain)<15 else "large")
                rid = self.tree.insert("","end", tags=(tag,), values=(
                    Path(rel).name, f"{src.duration:.2f}s", src.strategy,
                    self._fmt(ref), self._fmt(src), f"{gain:+.1f}", "ready"))
                self._row_ids.append((rid, ap, bp, rel, gain))
            except Exception as e:
                rid = self.tree.insert("","end", tags=("error",),
                    values=(Path(rel).name,"—","—","—","—","—",f"ERR: {e}"))
                self._row_ids.append((rid, ap, bp, rel, None))
            self.progress.set((i+1)/total*100); self.update()
        self.status_var.set(
            f"{total} file(s) matched.  Green < 6 dB  |  Orange 6–15 dB  |  Red > 15 dB"
            "  — review, then click Process Files.")
        self._run_btn.configure(state="normal")
        self.progress.set(0)

        # Populate session timeline with all Game B files
        out_dir = self.output_var.get().strip()
        self._tl_populate_from_scan(b, out_dir)

    def _fmt(self, s):
        if s.strategy == "peak_rms": return f"pk {s.peak_dbfs:.1f} dBFS"
        if s.strategy == "rms_lufs": return f"rms {s.rms_dbfs:.1f} dBFS"
        return f"{s.lufs_integrated:.1f} LUFS" if s.lufs_integrated > SILENCE_DBFS \
               else f"rms {s.rms_dbfs:.1f} dBFS"

    # ── Process ────────────────────────────────────────────────────────────

    def _process(self):
        out = self.output_var.get().strip()
        if not out:
            messagebox.showwarning("No output folder",
                                   "Please select an output folder."); return
        if self._running: return
        self._stop_all(save_position=False)
        self._running = True
        self._run_btn.configure(state="disabled", text="Processing…")
        threading.Thread(target=self._process_thread, args=(out,),
                         daemon=True).start()

    def _process_thread(self, out_dir):
        valid = [(r,a,b,rel,g) for r,a,b,rel,g in self._row_ids if g is not None]
        for i, (rid, ap, bp, rel, g) in enumerate(valid):
            try:
                apply_and_save(bp, str(Path(out_dir)/rel), g)
                vals = list(self.tree.item(rid,"values")); vals[-1] = "✓ written"
                self.tree.item(rid, values=vals, tags=("done",))
            except Exception as e:
                vals = list(self.tree.item(rid,"values")); vals[-1] = f"✗ {e}"
                self.tree.item(rid, values=vals, tags=("error",))
            self.progress.set((i+1)/len(valid)*100)
        self._running = False
        self.after(0, lambda: self._run_btn.configure(state="normal",
                                                       text="▶  Process Files"))
        self.after(0, lambda: self.status_var.set(
            f"Done! {len(valid)} file(s) written to: {out_dir}"))
        self.after(0, lambda: messagebox.showinfo("Complete",
            f"Done!\n{len(valid)} file(s) written to:\n{out_dir}"))
        self.after(100, self._on_row_select)
        # Auto-load output folder into timeline now files exist
        self.after(200, lambda: self._tl_load_from_folder(out_dir))

    # ── Row selection ──────────────────────────────────────────────────────

    def _on_row_select(self, event=None):
        self._stop_all(save_position=False)
        self._cancel_trim_debounce()
        sel = self.tree.selection()
        if not sel:
            self._selected = None
            self._clear_player_panel()
            self._wf_panel.clear()
            return
        rid = sel[0]
        for i, row in enumerate(self._row_ids):
            if row[0] == rid:
                self._selected = i
                _, ref_path, src_path, rel, gain = row
                fname = Path(rel).name
                self._ref_name_lbl.config(text=fname)
                out_dir   = self.output_var.get().strip()
                out_path  = str(Path(out_dir) / rel) if out_dir else None
                out_exists = out_path and os.path.isfile(out_path)
                self._out_name_lbl.config(
                    text=fname if out_exists else f"{fname}  (not yet processed)")
                self._ref_play_btn.configure(state="normal")
                self._out_play_btn.configure(
                    state="normal" if out_exists else "disabled")
                st = "normal" if out_exists else "disabled"
                self._trim_slider.configure(state=st)
                self._apply_btn.configure(state=st)
                self._reset_btn.configure(state=st)
                self.trim_var.set(0.0)
                self._trim_lbl.config(text="0.0 dB")
                self._preview_lbl.config(text="")
                self._ref_time_lbl.config(text="0:00")
                self._out_time_lbl.config(text="0:00")
                # Load waveforms
                self._wf_panel.clear()
                if out_exists:
                    try:
                        ref_dur = sf.info(ref_path).duration
                        out_dur = sf.info(out_path).duration
                    except:
                        ref_dur = out_dur = 1.0
                    self._wf_panel.set_data(ref_path, out_path, ref_dur, out_dur)
                return
        self._selected = None

    def _clear_player_panel(self):
        self._ref_name_lbl.config(text="—")
        self._out_name_lbl.config(text="—")
        self._ref_play_btn.configure(state="disabled")
        self._out_play_btn.configure(state="disabled")
        self._trim_slider.configure(state="disabled")
        self._apply_btn.configure(state="disabled")
        self._reset_btn.configure(state="disabled")
        self.trim_var.set(0.0)
        self._trim_lbl.config(text="0.0 dB")
        self._preview_lbl.config(text="")
        self._ref_time_lbl.config(text="0:00")
        self._out_time_lbl.config(text="0:00")

    # ── Playback ───────────────────────────────────────────────────────────

    def _get_paths(self):
        if self._selected is None: return None, None
        _, ref_path, _, rel, _ = self._row_ids[self._selected]
        out_dir  = self.output_var.get().strip()
        out_path = str(Path(out_dir) / rel) if out_dir else None
        return ref_path, out_path

    def _get_duration(self):
        if self._selected is None: return 1.0
        try:
            _, ref_path, _, _, _ = self._row_ids[self._selected]
            return sf.info(ref_path).duration
        except: return 1.0

    def _toggle_ref(self):
        if self._player_ref.is_playing():
            self._player_ref.stop(save_position=self.loop_var.get())
            self._ref_play_btn.config(text="▶ Play")
        else:
            ref_path, _ = self._get_paths()
            if not ref_path: return
            self._cancel_trim_debounce()
            self._player_out.stop(save_position=self.loop_var.get())
            self._out_play_btn.config(text="▶ Play")
            dur    = self._get_duration()
            resume = self._player_ref.get_resume_pos() if self.loop_var.get() else 0.0
            self._player_ref.play(ref_path, dur, start_pos=resume)
            self._ref_play_btn.config(text="■ Stop")

    def _toggle_out(self):
        if self._player_out.is_playing():
            self._player_out.stop(save_position=self.loop_var.get())
            self._out_play_btn.config(text="▶ Play")
            self._preview_playing = False
            self._preview_lbl.config(text="")
        else:
            _, out_path = self._get_paths()
            if not out_path or not os.path.isfile(out_path): return
            self._player_ref.stop(save_position=self.loop_var.get())
            self._ref_play_btn.config(text="▶ Play")
            dur    = self._get_duration()
            resume = self._player_out.get_resume_pos() if self.loop_var.get() else 0.0
            self._play_out_with_current_settings(resume, dur)

    def _play_out_with_current_settings(self, start_pos, dur):
        """Build preview file with current trim/fade/gain and play it."""
        _, out_path = self._get_paths()
        if not out_path or not os.path.isfile(out_path): return
        trim  = self.trim_var.get()
        ti    = self._wf_panel.trim_in
        to    = self._wf_panel.trim_out
        fi    = self._wf_panel.fade_in_dur
        fo    = self._wf_panel.fade_out_dur
        _, _, src_path, _, base_gain = self._row_ids[self._selected]
        has_edits = (abs(trim) > 0.05 or ti > 0.01 or
                     to < self._get_duration() - 0.01 or fi > 0.01 or fo > 0.01)
        if has_edits:
            try:
                tmp = _make_preview(src_path, base_gain+trim, ti, to, fi, fo)
                preview_dur = to - ti
                self._player_out.play(tmp, preview_dur, start_pos=start_pos)
                self._preview_playing = True
                self._preview_lbl.config(text="⚡ preview")
            except Exception as e:
                self._player_out.play(out_path, dur, start_pos=start_pos)
        else:
            self._player_out.play(out_path, dur, start_pos=start_pos)
            self._preview_playing = False
        self._out_play_btn.config(text="■ Stop")

    def _on_ref_stopped(self):
        self._ref_play_btn.config(text="▶ Play")
        if self.loop_var.get():
            ref_path, _ = self._get_paths()
            if ref_path:
                self._player_ref.play(ref_path, self._get_duration(), start_pos=0.0)
                self._ref_play_btn.config(text="■ Stop")

    def _on_out_stopped(self):
        self._out_play_btn.config(text="▶ Play")
        self._preview_playing = False
        self._preview_lbl.config(text="")
        if self.loop_var.get():
            _, out_path = self._get_paths()
            if out_path and os.path.isfile(out_path):
                self._play_out_with_current_settings(0.0, self._get_duration())

    def _stop_all(self, save_position=True):
        self._player_ref.stop(save_position=save_position)
        self._player_out.stop(save_position=save_position)
        self._ref_play_btn.config(text="▶ Play")
        self._out_play_btn.config(text="▶ Play")
        self._preview_playing = False
        self._preview_lbl.config(text="")

    def _poll_playback(self):
        def fmt(s):
            s = int(s); return f"{s//60}:{s%60:02d}"
        if self._player_ref.is_playing():
            pos = self._player_ref.get_position()
            self._ref_time_lbl.config(text=fmt(pos))
            self._wf_panel.set_ref_playhead(pos)
        if self._player_out.is_playing():
            pos = self._player_out.get_position()
            self._out_time_lbl.config(text=fmt(pos))
            self._wf_panel.set_out_playhead(pos)
        if self._tl_playing:
            self._tl_draw_playhead()
        self.after(100, self._poll_playback)

    # ── Waveform handle change ─────────────────────────────────────────────

    def _on_wf_change(self):
        """Called when trim/fade handles are dragged — restart preview if playing."""
        if self._player_out.is_playing():
            if self._trim_after_id:
                self.after_cancel(self._trim_after_id)
            self._trim_after_id = self.after(300, self._restart_out_preview)

    def _restart_out_preview(self):
        self._trim_after_id = None
        if self._player_out.is_playing():
            pos = self._player_out.get_position()
            self._player_out.stop(save_position=False)
            self._play_out_with_current_settings(pos, self._get_duration())

    # ── Gain trim drag ─────────────────────────────────────────────────────

    def _on_trim_drag(self, val):
        v = float(val)
        self._trim_lbl.config(text=f"{v:+.1f} dB" if v != 0 else "0.0 dB")
        if self._trim_after_id:
            self.after_cancel(self._trim_after_id)
        self._trim_after_id = self.after(300, self._auto_preview)

    def _auto_preview(self):
        self._trim_after_id = None
        if self._player_out.is_playing():
            pos = self._player_out.get_position()
            self._player_out.stop(save_position=False)
            self._play_out_with_current_settings(pos, self._get_duration())

    def _cancel_trim_debounce(self):
        if self._trim_after_id:
            self.after_cancel(self._trim_after_id)
            self._trim_after_id = None

    # ── Apply & Save ───────────────────────────────────────────────────────

    def _apply_all(self):
        if self._selected is None: return
        trim = self.trim_var.get()
        _, ref_path, src_path, rel, base_gain = self._row_ids[self._selected]
        out_dir = self.output_var.get().strip()
        if not out_dir:
            messagebox.showwarning("No output folder",
                                   "Please select an output folder first."); return
        out_path = str(Path(out_dir) / rel)
        if not os.path.isfile(out_path):
            messagebox.showwarning("File not found",
                                   "Run Process Files first."); return

        total_gain = base_gain + trim
        ti = self._wf_panel.trim_in
        to = self._wf_panel.trim_out
        fi = self._wf_panel.fade_in_dur
        fo = self._wf_panel.fade_out_dur

        self._stop_all(save_position=False)
        try:
            apply_and_save(src_path, out_path, total_gain, ti, to, fi, fo)

            # Update stored gain (trim baked in), reset UI
            row = list(self._row_ids[self._selected])
            row[4] = total_gain
            self._row_ids[self._selected] = tuple(row)
            rid = row[0]
            vals = list(self.tree.item(rid, "values"))
            vals[5] = f"{total_gain:+.1f}"
            vals[6] = "✓ saved"
            tag = "ok" if abs(total_gain)<6 else ("warning" if abs(total_gain)<15 else "large")
            self.tree.item(rid, values=vals, tags=(tag,))

            self.trim_var.set(0.0)
            self._trim_lbl.config(text="0.0 dB")
            self._preview_lbl.config(text="")

            # Reload waveform from newly saved file
            try:
                ref_dur = sf.info(ref_path).duration
                out_dur = sf.info(out_path).duration
            except:
                ref_dur = out_dur = self._get_duration()
            self._wf_panel.set_data(ref_path, out_path, ref_dur, out_dur)

            self.status_var.set(
                f"Saved {Path(rel).name}  gain {total_gain:+.1f} dB  "
                f"trim {ti:.2f}s–{to:.2f}s  "
                f"fade-in {fi:.2f}s  fade-out {fo:.2f}s")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save: {e}")

    def _reset_all(self):
        self._cancel_trim_debounce()
        self.trim_var.set(0.0)
        self._trim_lbl.config(text="0.0 dB")
        self._preview_lbl.config(text="")
        self._wf_panel.reset_handles()
        if self._player_out.is_playing() and self._preview_playing:
            pos = self._player_out.get_position()
            self._player_out.stop(save_position=False)
            _, out_path = self._get_paths()
            if out_path and os.path.isfile(out_path):
                self._player_out.play(out_path, self._get_duration(),
                                      start_pos=pos if self.loop_var.get() else 0.0)
                self._out_play_btn.config(text="■ Stop")
            self._preview_playing = False

    # ── Session Timeline panel ────────────────────────────────────────────

    def _build_session_panel(self, outer):
        """Build the session timeline right panel directly into outer (a plain Frame)."""
        outer.pack_propagate(False)

        # ── toolbar ───────────────────────────────────────────────────
        tb = tk.Frame(outer, padx=PAD//2, pady=6)
        tb.pack(side="top", fill="x")
        tk.Label(tb, text="Session Timeline",
                 font=("Helvetica", 13, "bold")).pack(side="left")
        tk.Button(tb, text="💾 Save Template", font=("Helvetica", 10),
                  command=self._tl_save_template).pack(side="right", padx=4)
        tk.Button(tb, text="📂 Load Template", font=("Helvetica", 10),
                  command=self._tl_load_template).pack(side="right", padx=4)
        tk.Button(tb, text="📁 Load Folder", font=("Helvetica", 10),
                  command=self._tl_load_folder_dialog).pack(side="right", padx=4)
        tk.Button(tb, text="✕ Clear", font=("Helvetica", 10),
                  command=self._tl_clear).pack(side="right", padx=4)

        # zoom
        zoom_fr = tk.Frame(outer, padx=PAD//2)
        zoom_fr.pack(side="top", fill="x")
        tk.Label(zoom_fr, text="Zoom:", font=("Helvetica", 10)).pack(side="left")
        self._zoom_var = tk.DoubleVar(value=DEFAULT_PPS)
        zoom_sl = tk.Scale(zoom_fr, from_=20, to=300, resolution=5,
                           orient="horizontal", variable=self._zoom_var,
                           showvalue=False, length=160,
                           command=self._tl_on_zoom)
        zoom_sl.pack(side="left", padx=4)
        self._zoom_lbl = tk.Label(zoom_fr, text=f"{DEFAULT_PPS:.0f} px/s",
                                  font=("Helvetica", 9), fg="#555", width=8)
        self._zoom_lbl.pack(side="left")

        # ── bottom section: pack from bottom up so tl_outer gets remaining space ──
        self._blk_wf = SingleFilePanel(outer, on_change=self._tl_on_wf_change,
                                       on_drag_start=self._tl_wf_drag_start,
                                       on_drag_end=self._commit_undo,
                                       bd=1, relief="sunken")
        self._blk_wf.pack(side="bottom", fill="x", padx=PAD//2, pady=(0, 8))

        tk.Label(outer, text="Waveform  —  drag handles to trim/fade",
                 font=("Helvetica", 10, "bold"), padx=PAD//2,
                 anchor="w").pack(side="bottom", fill="x")

        sep2 = tk.Frame(outer, height=1, bg="#ccc")
        sep2.pack(side="bottom", fill="x", padx=PAD//2, pady=4)

        apply_row = tk.Frame(outer, padx=PAD//2, pady=4)
        apply_row.pack(side="bottom", fill="x")
        self._blk_apply_btn = tk.Button(apply_row, text="✓ Apply & Save All",
                                        font=("Helvetica", 10, "bold"),
                                        padx=10, state="disabled",
                                        command=self._tl_apply_all)
        self._blk_apply_btn.pack(side="left", padx=(0, 6))
        self._blk_reset_btn = tk.Button(apply_row, text="↺ Reset",
                                        font=("Helvetica", 10),
                                        padx=8, state="disabled",
                                        command=self._tl_reset_block)
        self._blk_reset_btn.pack(side="left")

        gain_row = tk.Frame(outer, padx=PAD//2)
        gain_row.pack(side="bottom", fill="x")
        tk.Label(gain_row, text="Gain trim:", width=10, anchor="w",
                 font=("Helvetica", 10)).pack(side="left")
        self._blk_trim_slider = tk.Scale(gain_row, from_=-24.0, to=24.0,
                                         resolution=0.5, orient="horizontal",
                                         variable=self._blk_trim_var,
                                         font=("Helvetica", 9), length=180,
                                         showvalue=False, state="disabled",
                                         command=self._tl_on_gain_drag)
        self._blk_trim_slider.pack(side="left", padx=(0, 6))
        self._blk_trim_slider.bind("<ButtonPress-1>",   lambda e: self._push_undo())
        self._blk_trim_slider.bind("<ButtonRelease-1>", lambda e: self._commit_undo())
        self._blk_gain_lbl = tk.Label(gain_row, text="—", width=8,
                                      font=("Helvetica", 10, "bold"), fg="#333")
        self._blk_gain_lbl.pack(side="left")

        blk_fr = tk.Frame(outer, padx=PAD//2, pady=4)
        blk_fr.pack(side="bottom", fill="x")
        tk.Label(blk_fr, text="Selected block:",
                 font=("Helvetica", 10, "bold")).pack(side="left")
        self._blk_name_lbl = tk.Label(blk_fr, text="—",
                                      font=("Helvetica", 10), fg="#555")
        self._blk_name_lbl.pack(side="left", padx=6)
        self._blk_loop_var = tk.BooleanVar(value=False)
        self._blk_loop_chk = tk.Checkbutton(blk_fr, text="Loop",
                                             variable=self._blk_loop_var,
                                             font=("Helvetica", 10),
                                             command=self._tl_toggle_loop)
        self._blk_loop_chk.pack(side="right")

        sep = tk.Frame(outer, height=1, bg="#ccc")
        sep.pack(side="bottom", fill="x", padx=PAD//2, pady=4)

        # ── transport bar ──────────────────────────────────────────────
        tp = tk.Frame(outer, padx=PAD//2, pady=4)
        tp.pack(side="bottom", fill="x")
        self._tl_play_btn = tk.Button(tp, text="▶ Play", width=10,
                                      font=("Helvetica", 11, "bold"),
                                      command=self._tl_toggle_play_btn)
        self._tl_play_btn.pack(side="left", padx=(0, 8))
        self._tl_pos_lbl = tk.Label(tp, text="0.00s", width=8,
                                    font=("Helvetica", 11), fg="#333")
        self._tl_pos_lbl.pack(side="left")
        tk.Label(tp, text="  Spacebar to play/stop",
                 font=("Helvetica", 9), fg="#888").pack(side="left")

        # ── timeline canvas area (fills remaining vertical space) ──────
        tl_outer = tk.Frame(outer)
        tl_outer.pack(side="top", fill="both", expand=True, padx=PAD//2)

        # header row: non-scrolling corner + ruler (always visible)
        header_row = tk.Frame(tl_outer, height=RULER_H)
        header_row.pack(side="top", fill="x")
        header_row.pack_propagate(False)

        corner_cv = tk.Canvas(header_row, width=LABEL_W, height=RULER_H,
                              bg="#d0d0d0", highlightthickness=0)
        corner_cv.pack(side="left")
        corner_cv.create_text(LABEL_W // 2, RULER_H // 2, text="Track",
                              fill="#666", font=("Helvetica", 8))

        self._ruler_cv = tk.Canvas(header_row, height=RULER_H, bg="#d8d8d8",
                                   highlightthickness=0)
        self._ruler_cv.pack(side="left", fill="x", expand=True)
        self._ruler_cv.bind("<ButtonPress-1>", self._tl_ruler_press)

        # tracks row: scrollable labels + timeline
        tracks_row = tk.Frame(tl_outer)
        tracks_row.pack(side="top", fill="both", expand=True)

        # vertical scrollbar (pack right first so it claims space)
        self._tl_vscroll = ttk.Scrollbar(tracks_row, orient="vertical",
                                         command=self._tl_yview)
        self._tl_vscroll.pack(side="right", fill="y")

        # left: track labels (no ruler header — handled by corner_cv above)
        self._lbl_cv = tk.Canvas(tracks_row, width=LABEL_W,
                                 bg="#e8e8e8", highlightthickness=1,
                                 highlightbackground="#ccc")
        self._lbl_cv.pack(side="left", fill="y")
        self._lbl_cv.bind("<ButtonPress-1>",   self._tl_lbl_press)
        self._lbl_cv.bind("<B1-Motion>",       self._tl_lbl_motion)
        self._lbl_cv.bind("<ButtonRelease-1>", self._tl_lbl_release)
        self._lbl_cv.bind("<Enter>",      lambda e: (self._lbl_cv.focus_set(),   setattr(self, "_scroll_target", "tl")))
        self._lbl_cv.bind("<Leave>",      lambda e: setattr(self, "_scroll_target", None))
        self._lbl_cv.bind("<MouseWheel>", self._tl_on_wheel)
        self._ruler_cv.bind("<Enter>",      lambda e: (self._ruler_cv.focus_set(), setattr(self, "_scroll_target", "tl")))
        self._ruler_cv.bind("<Leave>",      lambda e: setattr(self, "_scroll_target", None))
        self._ruler_cv.bind("<MouseWheel>", self._tl_on_wheel)

        # right: scrollable timeline (no ruler — handled by _ruler_cv above)
        # _tl_cv sits directly in tracks_row so it has the same height as _lbl_cv.
        # The horizontal scrollbar lives in a separate bottom row to avoid shrinking _tl_cv.
        self._tl_cv = tk.Canvas(tracks_row, bg="#f8f8f8",
                                highlightthickness=1, highlightbackground="#ccc",
                                xscrollcommand=self._tl_xscroll_set,
                                yscrollcommand=self._tl_yscroll_set)
        self._tl_cv.pack(side="left", fill="both", expand=True)

        # bottom row: corner spacer + horizontal scrollbar (keeps _tl_cv full-height)
        bottom_row = tk.Frame(tl_outer)
        bottom_row.pack(side="top", fill="x")
        tk.Frame(bottom_row, width=LABEL_W).pack(side="left")
        self._tl_hscroll = ttk.Scrollbar(bottom_row, orient="horizontal",
                                          command=lambda *a: self._tl_cv.xview(*a))
        self._tl_hscroll.pack(side="left", fill="x", expand=True)

        self._tl_cv.bind("<ButtonPress-1>",   self._tl_on_press)
        self._tl_cv.bind("<B1-Motion>",        self._tl_on_motion)
        self._tl_cv.bind("<ButtonRelease-1>",  self._tl_on_release)
        self._tl_cv.bind("<Configure>",        lambda e: self._tl_draw())
        self._tl_cv.bind("<Enter>",      lambda e: (self._tl_cv.focus_set(), setattr(self, "_scroll_target", "tl")))
        self._tl_cv.bind("<Leave>",      lambda e: setattr(self, "_scroll_target", None))
        self._tl_cv.bind("<MouseWheel>", self._tl_on_wheel)
        self.bind("<space>",          self._tl_toggle_play)
        self.bind("<KeyPress-space>", self._tl_toggle_play)
        self.bind("<Delete>",         self._tl_on_delete)
        self.bind("<BackSpace>",      self._tl_on_delete)
        self.bind("<Command-z>",      self._undo)
        self.bind("<Command-Z>",      self._redo)

        self._setup_native_scroll()
        self._tl_draw_labels()

    # ── timeline helpers ───────────────────────────────────────────────

    def _tl_yview(self, *args):
        self._tl_cv.yview(*args)

    def _tl_yscroll_set(self, first, last):
        self._tl_vscroll.set(first, last)
        self._lbl_cv.yview_moveto(float(first))

    def _tl_xscroll_set(self, first, last):
        self._tl_hscroll.set(first, last)
        self._ruler_cv.xview_moveto(float(first))

    def _tl_on_wheel(self, event):
        amount    = max(1, abs(event.delta))
        direction = 1 if event.delta > 0 else -1   # natural-scroll direction
        if event.state & 0x1:   # Shift state = horizontal on macOS
            self._tl_cv.xview("scroll", direction * amount, "units")
        else:
            self._tl_yview("scroll", direction * amount, "units")
        return "break"

    def _tl_x(self, sec):
        return int(sec * self._pps)

    def _tl_sec(self, cx):
        return max(0.0, cx / self._pps)

    def _tl_n_tracks(self):
        if not self._session_blocks:
            return 1
        return max(b.track for b in self._session_blocks) + 1

    def _tl_height(self):
        return self._tl_n_tracks() * TRACK_H

    def _tl_track_y(self, track):
        return track * TRACK_H

    def _tl_canvas_track(self, cy):
        t = int(cy / TRACK_H)
        return max(0, min(self._tl_n_tracks() - 1, t))

    def _tl_is_muted_effective(self, track):
        """True if track should be silent: explicitly muted, or solo-isolated."""
        if track in self._tl_muted_tracks:
            return True
        if self._tl_soloed_tracks and track not in self._tl_soloed_tracks:
            return True
        return False

    def _tl_scroll_width(self):
        latest = max((b.start_sec + b.dur for b in self._session_blocks),
                     default=0.0)
        return max(TL_MIN_SEC, latest + 4.0)

    # ── waveform cache ─────────────────────────────────────────────────

    def _load_wf(self, file_abs):
        try:
            data, sr = sf.read(file_abs, always_2d=True)
            mono = np.abs(data.mean(axis=1))
            n_total = len(mono)
            n_chunks = min(2000, max(50, n_total // 128))
            trim = (n_total // n_chunks) * n_chunks
            chunks = mono[:trim].reshape(n_chunks, -1)
            peaks = chunks.max(axis=1).astype(np.float32)
            if trim < n_total:
                peaks[-1] = max(peaks[-1], float(mono[trim:].max()))
            mx = peaks.max()
            if mx > 0:
                peaks /= mx
            self._wf_cache[file_abs] = peaks
        except Exception:
            self._wf_cache[file_abs] = np.array([], dtype=np.float32)
        self.after(0, self._tl_draw)

    # ── draw ───────────────────────────────────────────────────────────

    def _tl_draw(self):
        cv = self._tl_cv
        cv.delete("all")
        w = max(cv.winfo_width(), 400)

        n_tracks  = self._tl_n_tracks()
        tl_h      = self._tl_height()   # tracks only, no ruler
        total_sec = self._tl_scroll_width()
        total_px  = self._tl_x(total_sec)
        scroll_w  = max(total_px, w)
        cv.configure(scrollregion=(0, 0, scroll_w, tl_h))
        self._lbl_cv.configure(scrollregion=(0, 0, LABEL_W, tl_h))
        self._ruler_cv.configure(scrollregion=(0, 0, scroll_w, RULER_H))

        # Draw ruler in the fixed header canvas
        rcv = self._ruler_cv
        rcv.delete("all")
        rcv.create_rectangle(0, 0, scroll_w, RULER_H, fill="#d8d8d8", outline="")
        for interval in [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60]:
            if total_sec / interval <= 60: break
        t = 0.0
        while t <= total_sec + 1e-6:
            x = self._tl_x(t)
            rcv.create_line(x, 0, x, RULER_H, fill="#888")
            lbl = (f"{t:.2f}" if total_sec < 4
                   else f"{t:.1f}" if total_sec < 20
                   else f"{int(t)}s")
            rcv.create_text(x + 2, RULER_H // 2, text=lbl, anchor="w",
                            fill="#444", font=("Helvetica", 8))
            t = round(t + interval, 6)

        # Track backgrounds
        for i in range(n_tracks):
            y0 = self._tl_track_y(i)
            if self._tl_is_muted_effective(i):
                bg = "#d8d0d0" if i % 2 == 0 else "#d0c8c8"
            else:
                bg = "#f0f0f0" if i % 2 == 0 else "#e8e8f0"
            cv.create_rectangle(0, y0, scroll_w, y0 + TRACK_H,
                                fill=bg, outline="")
            cv.create_line(0, y0, scroll_w, y0, fill="#d0d0d0")

        # Blocks
        for b in self._session_blocks:
            x0 = self._tl_x(b.start_sec)
            x1 = self._tl_x(b.start_sec + max(b.dur, 0.3))
            y0 = self._tl_track_y(b.track) + TRACK_PAD
            y1 = self._tl_track_y(b.track) + TRACK_H - TRACK_PAD
            selected  = b in self._sel_blocks
            muted     = self._tl_is_muted_effective(b.track)
            pending   = b.bid in self._pending_bids
            if selected:
                outline_col, outline_w, dash = "#ffffff", 2, None
            elif pending:
                outline_col, outline_w, dash = "#f0c000", 2, (4, 2)
            else:
                outline_col, outline_w, dash = "#333333", 1, None
            stipple = "gray50" if muted else ""
            kw = dict(fill=b.color, outline=outline_col, width=outline_w,
                      stipple=stipple, tags=("block", b.bid))
            if dash:
                kw["dash"] = dash
            cv.create_rectangle(x0, y0, x1, y1, **kw)

            # Waveform overlay — lazy-load in background thread
            if b.file_abs not in self._wf_cache:
                self._wf_cache[b.file_abs] = _WF_LOADING
                threading.Thread(target=self._load_wf,
                                 args=(b.file_abs,), daemon=True).start()
            wf = self._wf_cache.get(b.file_abs)
            if isinstance(wf, np.ndarray) and len(wf) > 1:
                bw = x1 - x0
                if bw >= 4:
                    cy_f   = (y0 + y1) / 2.0
                    half_h = ((y1 - y0) / 2.0) * 0.85
                    n = len(wf)
                    n_px = int(bw)
                    indices = np.linspace(0, n - 1, n_px).astype(int)
                    samples = wf[indices]
                    # Polygon: top edge left→right then bottom edge right→left
                    pts = []
                    for px in range(n_px):
                        pts += [x0 + px, cy_f - samples[px] * half_h]
                    for px in range(n_px - 1, -1, -1):
                        pts += [x0 + px, cy_f + samples[px] * half_h]
                    wf_col = _lighter(b.color, 0.55)
                    cv.create_polygon(pts, fill=wf_col, outline="",
                                      stipple=stipple,
                                      tags=("block", b.bid))

            if b.loop:
                cv.create_text(x0 + 4, y0 + 3, text="∞",
                               anchor="nw", fill="white",
                               font=("Helvetica", 9, "bold"))
            label = Path(b.file_abs).stem
            max_chars = max(1, (x1 - x0 - 8) // 7)
            if len(label) > max_chars:
                label = label[:max_chars - 1] + "…"
            cv.create_text(x0 + 4, (y0 + y1) // 2, text=label,
                           anchor="w", fill="white",
                           font=("Helvetica", 9, "bold"))

        # Rubber-band selection rect
        if self._tl_rubber:
            rx0, ry0, rx1, ry1 = self._tl_rubber
            cv.create_rectangle(rx0, ry0, rx1, ry1,
                                outline="#4477ee", width=1,
                                fill="#4477ee", stipple="gray25",
                                tags="rubber")

        # Start marker
        sm_x = self._tl_x(self._tl_start_pos)
        cv.create_line(sm_x, 0, sm_x, tl_h, fill="#e08000", width=2,
                       tags="start_marker")
        cv.create_polygon(sm_x - 7, 0, sm_x + 7, 0, sm_x, 12,
                          fill="#e08000", outline="", tags="start_marker")

        self._tl_draw_labels()
        self._tl_draw_playhead()

    def _tl_draw_playhead(self):
        cv = self._tl_cv
        cv.delete("playhead")
        if self._tl_playing and self._tl_play_ts is not None:
            pos = self._tl_start_pos + (time.monotonic() - self._tl_play_ts)
            x   = self._tl_x(pos)
            cv.create_line(x, 0, x, self._tl_height(), fill="#dd2020", width=2,
                           tags="playhead")
            self._tl_pos_lbl.config(text=f"{pos:.2f}s")
        else:
            self._tl_pos_lbl.config(
                text=f"{self._tl_start_pos:.2f}s")

    def _tl_draw_labels(self):
        cv = self._lbl_cv
        cv.delete("all")
        n_tracks  = self._tl_n_tracks()
        drag_src  = self._tl_lbl_drag_src
        drag_y    = self._tl_lbl_drag_y
        raw_dst   = int(drag_y / TRACK_H) if drag_src is not None else -1
        drag_dst  = max(0, min(n_tracks - 1, raw_dst)) if drag_src is not None else -1

        # Build track-index → display name from blocks
        track_names = {}
        for b in self._session_blocks:
            if b.track not in track_names:
                track_names[b.track] = b.slot_name

        def _short(name, max_ch=11):
            return name if len(name) <= max_ch else name[:max_ch - 1] + "…"

        for i in range(n_tracks):
            y0  = self._tl_track_y(i)
            yc  = y0 + TRACK_H // 2
            muted  = i in self._tl_muted_tracks
            soloed = i in self._tl_soloed_tracks
            is_drag = (drag_src is not None and i == drag_src)
            bg = ("#d8c8c8" if muted else ("#e0e0e0" if i % 2 == 0 else "#d8d8e8"))
            alpha_bg = "#c0c8d8" if is_drag else bg
            cv.create_rectangle(0, y0, LABEL_W, y0 + TRACK_H,
                               fill=alpha_bg, outline="")
            cv.create_line(0, y0, LABEL_W, y0, fill="#c0c0c0")
            fg_col = "#aaa" if is_drag else "#555"
            label = _short(track_names.get(i, f"T{i + 1}"))
            cv.create_text(6, yc, text=label,
                           anchor="w", fill=fg_col, font=("Helvetica", 9))
            # Mute button
            bx0 = LABEL_W - MUTE_BTN_W - 4
            bx1 = LABEL_W - 4
            btn_bg = "#cc4444" if muted else "#d0d0d0"
            btn_fg = "white"   if muted else "#555"
            cv.create_rectangle(bx0, yc - 9, bx1, yc + 9,
                                fill=btn_bg, outline="#888", tags=f"mute_{i}")
            cv.create_text((bx0 + bx1) // 2, yc, text="M",
                           fill=btn_fg, font=("Helvetica", 8, "bold"),
                           tags=f"mute_{i}")
            # Solo button (left of mute)
            sx0 = bx0 - SOLO_BTN_W - 2
            sx1 = bx0 - 2
            sol_bg = "#c8a020" if soloed else "#d0d0d0"
            sol_fg = "white"   if soloed else "#555"
            cv.create_rectangle(sx0, yc - 9, sx1, yc + 9,
                                fill=sol_bg, outline="#888", tags=f"solo_{i}")
            cv.create_text((sx0 + sx1) // 2, yc, text="S",
                           fill=sol_fg, font=("Helvetica", 8, "bold"),
                           tags=f"solo_{i}")

        # Drag ghost — draw floating label at cursor
        if drag_src is not None:
            gy = max(0, min(drag_y, n_tracks * TRACK_H - TRACK_H))
            gyc = gy + TRACK_H // 2
            cv.create_rectangle(2, gy + 2, LABEL_W - 2, gy + TRACK_H - 2,
                                fill="#4477cc", outline="#224488", width=2)
            drag_label = _short(track_names.get(drag_src, f"T{drag_src + 1}"))
            cv.create_text(6, gyc, text=drag_label,
                           anchor="w", fill="white", font=("Helvetica", 9, "bold"))
            # Drop indicator line
            if drag_dst >= drag_src:
                drop_y = self._tl_track_y(drag_dst) + TRACK_H
            else:
                drop_y = self._tl_track_y(drag_dst)
            cv.create_line(0, drop_y, LABEL_W, drop_y, fill="#224488", width=2)

    # ── interaction ────────────────────────────────────────────────────

    def _tl_ruler_press(self, event):
        cx = self._ruler_cv.canvasx(event.x)
        self._tl_start_pos = max(0.0, self._tl_sec(cx))
        self._tl_draw()

    def _tl_lbl_press(self, event):
        x  = event.x
        cy = self._lbl_cv.canvasy(event.y)
        track = int(cy / TRACK_H)
        if track >= self._tl_n_tracks():
            return
        bx0 = LABEL_W - MUTE_BTN_W - 4  # mute left edge
        sx0 = bx0 - SOLO_BTN_W - 2      # solo left edge
        if x >= bx0:
            # Mute button
            self._push_undo(); self._commit_undo()
            if track in self._tl_muted_tracks:
                self._tl_muted_tracks.discard(track)
            else:
                self._tl_muted_tracks.add(track)
            self._tl_draw_labels()
            self._tl_draw()
        elif x >= sx0:
            # Solo button
            self._push_undo(); self._commit_undo()
            if track in self._tl_soloed_tracks:
                self._tl_soloed_tracks.discard(track)
            else:
                self._tl_soloed_tracks.add(track)
            self._tl_draw_labels()
            self._tl_draw()
        else:
            # Start track reorder drag
            self._push_undo()
            self._tl_lbl_drag_src = track
            self._tl_lbl_drag_y   = cy
            self._tl_draw_labels()

    def _tl_lbl_motion(self, event):
        if self._tl_lbl_drag_src is None:
            return
        self._tl_lbl_drag_y = self._lbl_cv.canvasy(event.y)
        self._tl_draw_labels()

    def _tl_lbl_release(self, event):
        self._commit_undo()
        if self._tl_lbl_drag_src is None:
            return
        src = self._tl_lbl_drag_src
        cy  = self._lbl_cv.canvasy(event.y)
        raw = int(cy / TRACK_H)
        dst = max(0, min(self._tl_n_tracks() - 1, raw))
        self._tl_lbl_drag_src = None
        if src != dst:
            self._tl_reorder_tracks(src, dst)
        self._tl_draw_labels()
        self._tl_draw()

    def _tl_reorder_tracks(self, src, dst):
        for b in self._session_blocks:
            if b.track == src:
                b.track = dst
            elif src < dst and src < b.track <= dst:
                b.track -= 1
            elif src > dst and dst <= b.track < src:
                b.track += 1
        def _remap(track_set):
            new = set()
            for t in track_set:
                if t == src:
                    new.add(dst)
                elif src < dst and src < t <= dst:
                    new.add(t - 1)
                elif src > dst and dst <= t < src:
                    new.add(t + 1)
                else:
                    new.add(t)
            return new
        self._tl_muted_tracks  = _remap(self._tl_muted_tracks)
        self._tl_soloed_tracks = _remap(self._tl_soloed_tracks)

    def _tl_on_press(self, event):
        cx  = self._tl_cv.canvasx(event.x)
        cy  = self._tl_cv.canvasy(event.y)
        add = bool(event.state & 0x0001) or bool(event.state & 0x0010)  # Shift or Cmd
        opt = bool(event.state & 0x0008)                                 # Option/Alt
        hit = self._tl_hit_block(cx, cy)
        if hit:
            if add:
                if hit in self._sel_blocks:
                    self._sel_blocks.discard(hit)
                    self._sel_block = next(iter(self._sel_blocks), None)
                    if self._sel_block:
                        self._tl_select_block(self._sel_block)
                    else:
                        self._tl_clear_editor()
                else:
                    self._sel_blocks.add(hit)
                    self._tl_select_block(hit)
            else:
                if hit not in self._sel_blocks:
                    self._sel_blocks = {hit}
                    self._tl_select_block(hit)
            # Push undo before drag starts (gated by _undo_committed)
            self._push_undo()
            # Start drag — record original positions of all selected blocks
            self._tl_drag_block    = hit
            self._tl_drag_orig_sec = hit.start_sec
            self._tl_drag_orig_trk = hit.track
            self._tl_drag_start_cx = cx
            self._tl_drag_start_cy = cy
            self._tl_drag_all_orig = {b.bid: (b.start_sec, b.track)
                                      for b in self._sel_blocks}
            self._tl_drag_copy     = opt and not add  # Option+drag = clone on first motion
        else:
            if not add:
                self._sel_blocks = set()
                self._sel_block  = None
                self._tl_clear_editor()
            # Start rubber-band; set start marker at click position
            self._tl_rubber      = (cx, cy, cx, cy)
            self._tl_start_pos   = max(0.0, self._tl_sec(cx))
            self._tl_draw()

    def _tl_on_motion(self, event):
        cx = self._tl_cv.canvasx(event.x)
        cy = self._tl_cv.canvasy(event.y)
        if self._tl_drag_block is not None:
            # Option+drag: clone selected blocks on the first real motion
            if self._tl_drag_copy:
                self._tl_drag_copy = False
                copies = []
                bid_map = {}
                for b in self._sel_blocks:
                    new_b = SessionBlock(
                        bid=str(uuid.uuid4())[:8],
                        slot_name=b.slot_name, file_abs=b.file_abs,
                        file_rel=b.file_rel, track=b.track,
                        start_sec=b.start_sec, dur=b.dur,
                        gain_db=b.gain_db, loop=b.loop,
                        trim_in=b.trim_in, trim_out=b.trim_out,
                        fade_in=b.fade_in, fade_out=b.fade_out,
                        color=b.color,
                    )
                    copies.append(new_b)
                    bid_map[b.bid] = new_b
                    if b.bid in self._pending_bids:
                        self._pending_bids.add(new_b.bid)
                self._session_blocks.extend(copies)
                new_drag = bid_map[self._tl_drag_block.bid]
                self._tl_drag_block    = new_drag
                self._sel_blocks       = set(copies)
                self._sel_block        = new_drag
                self._tl_drag_all_orig = {c.bid: (c.start_sec, c.track) for c in copies}

            dx_sec    = (cx - self._tl_drag_start_cx) / self._pps
            drag_orig_trk = self._tl_drag_all_orig[self._tl_drag_block.bid][1]
            delta_trk = self._tl_canvas_track(cy) - drag_orig_trk
            for b in self._sel_blocks:
                orig_sec, orig_trk = self._tl_drag_all_orig[b.bid]
                b.start_sec = max(0.0, round(orig_sec + dx_sec, 3))
                b.track     = max(0, orig_trk + delta_trk)
            self._tl_draw()
        elif self._tl_rubber is not None:
            rx0, ry0 = self._tl_rubber[0], self._tl_rubber[1]
            self._tl_rubber = (rx0, ry0, cx, cy)
            # Update selection to blocks that overlap the rubber-band rect
            lx, rx = min(rx0, cx), max(rx0, cx)
            ty, by = min(ry0, cy), max(ry0, cy)
            self._sel_blocks = set()
            for b in self._session_blocks:
                bx0 = self._tl_x(b.start_sec)
                bx1 = self._tl_x(b.start_sec + max(b.dur, 0.3))
                by0 = self._tl_track_y(b.track) + TRACK_PAD
                by1 = self._tl_track_y(b.track) + TRACK_H - TRACK_PAD
                if bx0 < rx and bx1 > lx and by0 < by and by1 > ty:
                    self._sel_blocks.add(b)
            self._sel_block = next(iter(self._sel_blocks), None)
            self._tl_draw()

    def _tl_on_release(self, event):
        self._commit_undo()
        self._tl_drag_copy = False
        if self._tl_drag_block is not None:
            self._tl_drag_block    = None
            self._tl_drag_all_orig = {}
        elif self._tl_rubber is not None:
            self._tl_rubber = None
            if self._sel_block:
                self._tl_select_block(self._sel_block)
            self._tl_draw()

    def _tl_hit_block(self, cx, cy):
        for b in reversed(self._session_blocks):
            x0 = self._tl_x(b.start_sec)
            x1 = self._tl_x(b.start_sec + max(b.dur, 0.3))
            y0 = self._tl_track_y(b.track) + TRACK_PAD
            y1 = self._tl_track_y(b.track) + TRACK_H - TRACK_PAD
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return b
        return None

    def _tl_select_block(self, block):
        """Set primary selection and update the editor panel. Does NOT change _sel_blocks."""
        self._sel_block = block
        n = len(self._sel_blocks)
        suffix = f"  (+{n-1} more)" if n > 1 else ""
        self._blk_name_lbl.config(text=Path(block.file_abs).name + suffix)
        self._blk_trim_var.set(block.gain_db)
        self._blk_gain_lbl.config(
            text=f"{block.gain_db:+.1f} dB" if block.gain_db != 0 else "0.0 dB")
        self._blk_loop_var.set(block.loop)
        self._blk_trim_slider.configure(state="normal")
        self._blk_reset_btn.configure(state="normal")
        self._update_apply_btn()
        self._blk_wf.clear()
        try:    dur = sf.info(block.file_abs).duration
        except: dur = block.dur
        self._blk_wf.set_data(block.file_abs, dur)
        self._blk_wf._trim_in_s  = block.trim_in
        self._blk_wf._trim_out_s = block.trim_out if block.trim_out > 0 else dur
        self._blk_wf._fade_in_s  = block.fade_in
        self._blk_wf._fade_out_s = block.fade_out
        self._blk_wf._sync_entries()
        self._tl_draw()

    def _tl_clear_editor(self):
        self._blk_name_lbl.config(text="—")
        self._blk_trim_slider.configure(state="disabled")
        self._blk_reset_btn.configure(state="disabled")
        self._blk_wf.clear()
        self._update_apply_btn()
        self._tl_draw()

    def _tl_toggle_loop(self):
        if self._sel_block:
            self._push_undo(); self._commit_undo()
            self._sel_block.loop = self._blk_loop_var.get()
            self._mark_pending(self._sel_block)
            self._tl_draw()

    def _tl_on_zoom(self, val):
        self._pps = float(val)
        self._zoom_lbl.config(text=f"{self._pps:.0f} px/s")
        self._tl_draw()

    # ── spacebar / transport ───────────────────────────────────────────

    def _tl_toggle_play(self, event=None):
        if self._tl_playing:
            self._tl_stop()
        else:
            self._tl_play()
        return "break"  # prevent default spacebar scroll

    def _tl_toggle_play_btn(self):
        self._tl_toggle_play()

    def _tl_play(self):
        if not self._session_blocks: return
        self._tl_stop()
        sox = SOX if os.path.isfile(SOX) else shutil.which("sox")
        if not sox: return
        start = self._tl_start_pos
        self._tl_procs  = []
        self._tl_timers = []
        self._tl_playing    = True
        self._tl_play_ts    = time.monotonic()
        for b in self._session_blocks:
            if self._tl_is_muted_effective(b.track):
                continue
            region = (b.trim_out - b.trim_in) if b.trim_out > b.trim_in else b.dur
            if region <= 0: region = b.dur
            gain_lin = _from_db(b.gain_db)
            if b.loop:
                # Playing at start? figure out phase offset
                delay = b.start_sec - start
                if delay < 0:
                    phase = (-delay) % region
                    offset = b.trim_in + phase
                    self._tl_launch_sox(sox, b.file_abs, gain_lin,
                                        offset, region, loop=True)
                else:
                    t = threading.Timer(delay,
                        lambda bfa=b.file_abs, g=gain_lin,
                               tri=b.trim_in, reg=region:
                        self._tl_launch_sox(sox, bfa, g, tri, reg, loop=True))
                    t.daemon = True
                    t.start()
                    self._tl_timers.append(t)
            else:
                # One-shot
                end_sec = b.start_sec + region
                if start >= end_sec:
                    continue  # already passed
                if start >= b.start_sec:
                    elapsed = start - b.start_sec
                    offset  = b.trim_in + elapsed
                    remaining = region - elapsed
                    self._tl_launch_sox(sox, b.file_abs, gain_lin,
                                        offset, remaining, loop=False)
                else:
                    delay = b.start_sec - start
                    t = threading.Timer(delay,
                        lambda bfa=b.file_abs, g=gain_lin,
                               tri=b.trim_in, reg=region:
                        self._tl_launch_sox(sox, bfa, g, tri, reg, loop=False))
                    t.daemon = True
                    t.start()
                    self._tl_timers.append(t)
        self._tl_play_btn.config(text="■ Stop")
        self._tl_cv.focus_set()

    def _tl_launch_sox(self, sox, path, gain_lin, trim_in, duration, loop):
        cmd = [sox, "-v", str(gain_lin), path, "-d", "trim", str(trim_in)]
        if loop:
            cmd += ["repeat", "-1"]
        else:
            cmd += [str(duration)]
        try:
            p = subprocess.Popen(cmd,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            self._tl_procs.append(p)
        except Exception:
            pass

    def _tl_stop(self):
        self._tl_playing = False
        self._tl_play_ts = None
        for t in self._tl_timers:
            try: t.cancel()
            except: pass
        self._tl_timers = []
        for p in self._tl_procs:
            try: p.kill()
            except: pass
        self._tl_procs = []
        self.after(0, lambda: self._tl_play_btn.config(text="▶ Play"))
        self.after(0, self._tl_draw)

    # ── gain drag (live) ───────────────────────────────────────────────

    def _tl_on_gain_drag(self, val):
        v = float(val)
        self._blk_gain_lbl.config(
            text=f"{v:+.1f} dB" if v != 0 else "0.0 dB")
        if self._sel_block:
            self._sel_block.gain_db = v
            self._mark_pending(self._sel_block)

    # ── waveform editor callbacks ──────────────────────────────────────

    def _tl_wf_drag_start(self):
        self._push_undo()

    def _tl_on_wf_change(self):
        if self._sel_block:
            b = self._sel_block
            b.trim_in   = self._blk_wf.trim_in
            b.trim_out  = self._blk_wf.trim_out
            b.fade_in   = self._blk_wf.fade_in_dur
            b.fade_out  = self._blk_wf.fade_out_dur
            self._mark_pending(b)

    def _tl_apply_all(self):
        if not self._pending_bids: return
        out_dir = self.output_var.get().strip()
        if not out_dir:
            messagebox.showwarning("No output folder",
                                   "Please select an output folder first.")
            return
        pending_blocks = [b for b in self._session_blocks
                          if b.bid in self._pending_bids]
        errors, saved = [], 0
        for b in pending_blocks:
            out_path = str(Path(out_dir) / b.file_rel)
            try:
                apply_and_save(b.file_abs, out_path,
                               b.gain_db, b.trim_in, b.trim_out,
                               b.fade_in, b.fade_out)
                b.file_abs = out_path
                try:    b.dur = sf.info(out_path).duration
                except: pass
                self._wf_cache.pop(out_path, None)  # force waveform refresh
                self._pending_bids.discard(b.bid)
                saved += 1
            except Exception as e:
                errors.append(f"{Path(b.file_rel).name}: {e}")
        if self._sel_block and self._sel_block.bid not in self._pending_bids:
            self._blk_wf.set_data(self._sel_block.file_abs, self._sel_block.dur)
        if errors:
            messagebox.showerror("Errors during apply",
                                 f"{saved} saved, {len(errors)} failed:\n" +
                                 "\n".join(errors))
        else:
            self.status_var.set(f"Applied & saved {saved} file(s) to {out_dir}")
        self._update_apply_btn()
        self._tl_draw()

    def _tl_reset_block(self):
        if not self._sel_block: return
        self._push_undo(); self._commit_undo()
        b = self._sel_block
        b.gain_db = 0.0
        b.trim_in = 0.0
        b.trim_out = b.dur
        b.fade_in = 0.0
        b.fade_out = 0.0
        self._blk_trim_var.set(0.0)
        self._blk_gain_lbl.config(text="0.0 dB")
        self._blk_wf.reset_handles()
        self._mark_pending(b)

    # ── load files into timeline ───────────────────────────────────────

    def _tl_load_from_folder(self, folder):
        """Load all audio files from folder. Update existing blocks, add new ones."""
        root = Path(folder)
        files = sorted(
            (str(p.relative_to(root)), str(p))
            for p in root.rglob("*")
            if p.suffix.lower() in AUDIO_EXTENSIONS
        )
        # keyed by filename (case-insensitive) for matching
        existing = {Path(b.file_rel).name.lower(): b
                    for b in self._session_blocks}
        added = updated = 0
        for rel, abs_path in files:
            key = Path(rel).name.lower()
            if key in existing:
                b = existing[key]
                b.file_abs = abs_path
                b.file_rel = rel
                try:    b.dur = sf.info(abs_path).duration
                except: pass
                updated += 1
            else:
                try:    dur = sf.info(abs_path).duration
                except: dur = 1.0
                color = BLOCK_COLS[self._color_idx % len(BLOCK_COLS)]
                self._color_idx += 1
                track = len(self._session_blocks)
                self._session_blocks.append(SessionBlock(
                    bid=str(uuid.uuid4())[:8],
                    slot_name=Path(rel).stem,
                    file_abs=abs_path, file_rel=rel,
                    track=track, start_sec=0.0, dur=dur,
                    gain_db=0.0, loop=False,
                    trim_in=0.0, trim_out=dur,
                    fade_in=0.0, fade_out=0.0,
                    color=color,
                ))
                added += 1
        self._tl_draw()
        return added, updated

    def _tl_load_folder_dialog(self):
        folder = filedialog.askdirectory(title="Load folder into timeline")
        if not folder: return
        added, updated = self._tl_load_from_folder(folder)
        self.status_var.set(
            f"Timeline: loaded {added} new + updated {updated} existing blocks "
            f"from {Path(folder).name}/")

    def _tl_populate_from_scan(self, b_dir, out_dir):
        """Called after Scan — load Game B files, prefer output versions if they exist."""
        added, updated = self._tl_load_from_folder(b_dir)
        if out_dir:
            self._tl_refresh_file_paths(out_dir)

    def _tl_refresh_file_paths(self, out_dir):
        """Switch existing blocks to their output versions where available."""
        if not out_dir: return
        for b in self._session_blocks:
            op = str(Path(out_dir) / b.file_rel)
            if os.path.isfile(op):
                b.file_abs = op
                try:    b.dur = sf.info(op).duration
                except: pass
        self._tl_draw()

    def _tl_on_delete(self, event=None):
        if not self._sel_blocks: return
        self._push_undo(); self._commit_undo()
        self._tl_stop()
        for b in self._sel_blocks:
            self._pending_bids.discard(b.bid)
            try: self._session_blocks.remove(b)
            except ValueError: pass
        self._sel_blocks = set()
        self._sel_block  = None
        self._tl_clear_editor()

    # ── undo / redo ────────────────────────────────────────────────────

    def _make_snapshot(self):
        return {
            'blocks': [
                {'bid': b.bid, 'slot_name': b.slot_name,
                 'file_abs': b.file_abs, 'file_rel': b.file_rel,
                 'track': b.track, 'start_sec': b.start_sec, 'dur': b.dur,
                 'gain_db': b.gain_db, 'loop': b.loop,
                 'trim_in': b.trim_in, 'trim_out': b.trim_out,
                 'fade_in': b.fade_in, 'fade_out': b.fade_out, 'color': b.color}
                for b in self._session_blocks
            ],
            'muted':   frozenset(self._tl_muted_tracks),
            'soloed':  frozenset(self._tl_soloed_tracks),
            'pending': frozenset(self._pending_bids),
        }

    def _push_undo(self):
        if not self._undo_committed:
            return
        self._undo_committed = False
        self._undo_stack.append(self._make_snapshot())
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _commit_undo(self):
        self._undo_committed = True

    def _restore_snapshot(self, snap):
        bid_map = {b.bid: b for b in self._session_blocks}
        new_blocks = []
        for bd in snap['blocks']:
            b = bid_map.get(bd['bid'])
            if b is None:
                b = SessionBlock(**bd)
            else:
                for k, v in bd.items():
                    setattr(b, k, v)
            new_blocks.append(b)
        self._session_blocks   = new_blocks
        self._tl_muted_tracks  = set(snap['muted'])
        self._tl_soloed_tracks = set(snap['soloed'])
        self._pending_bids     = set(snap.get('pending', set()))
        new_bid_map = {b.bid: b for b in self._session_blocks}
        if self._sel_block:
            self._sel_block  = new_bid_map.get(self._sel_block.bid)
            self._sel_blocks = {new_bid_map[b.bid] for b in self._sel_blocks
                                if b.bid in new_bid_map}
            if self._sel_block:
                self._tl_select_block(self._sel_block)
            else:
                self._sel_blocks = set()
                self._tl_clear_editor()

    def _undo(self, event=None):
        if not self._undo_stack: return
        self._commit_undo()
        self._redo_stack.append(self._make_snapshot())
        self._restore_snapshot(self._undo_stack.pop())
        self._update_apply_btn()
        self._tl_draw()

    def _redo(self, event=None):
        if not self._redo_stack: return
        self._commit_undo()
        self._undo_stack.append(self._make_snapshot())
        self._restore_snapshot(self._redo_stack.pop())
        self._update_apply_btn()
        self._tl_draw()

    def _mark_pending(self, block):
        self._pending_bids.add(block.bid)
        self._update_apply_btn()

    def _update_apply_btn(self):
        n = len(self._pending_bids)
        if n == 0:
            self._blk_apply_btn.configure(state="disabled", text="✓ Apply & Save All")
        elif n == 1:
            self._blk_apply_btn.configure(state="normal", text="✓ Apply & Save (1)")
        else:
            self._blk_apply_btn.configure(state="normal", text=f"✓ Apply All ({n})")

    def _tl_clear(self):
        if not self._session_blocks: return
        if not messagebox.askyesno("Clear session",
                                   "Remove all blocks from the timeline?"):
            return
        self._tl_stop()
        self._session_blocks = []
        self._sel_block      = None
        self._sel_blocks     = set()
        self._color_idx      = 0
        self._pending_bids   = set()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._tl_clear_editor()

    # ── save / load template ───────────────────────────────────────────

    def _tl_save_template(self):
        path = filedialog.asksaveasfilename(
            title="Save Session Template",
            defaultextension=".json",
            filetypes=[("JSON template", "*.json"), ("All files", "*.*")])
        if not path: return
        data = {
            "version": 1,
            "muted_tracks":  sorted(self._tl_muted_tracks),
            "soloed_tracks": sorted(self._tl_soloed_tracks),
            "blocks": [],
        }
        for b in sorted(self._session_blocks, key=lambda b: (b.track, b.start_sec)):
            data["blocks"].append({
                "slot_name": b.slot_name,
                "filename":  Path(b.file_rel).name,
                "file_rel":  b.file_rel,
                "track":     b.track,
                "start_sec": round(b.start_sec, 3),
                "dur":       round(b.dur, 3),
                "gain_db":   round(b.gain_db, 2),
                "loop":      b.loop,
                "trim_in":   round(b.trim_in, 3),
                "trim_out":  round(b.trim_out, 3),
                "fade_in":   round(b.fade_in, 3),
                "fade_out":  round(b.fade_out, 3),
                "color":     b.color,
            })
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.status_var.set(f"Template saved: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _tl_load_template(self):
        path = filedialog.askopenfilename(
            title="Load Session Template",
            filetypes=[("JSON template", "*.json"), ("All files", "*.*")])
        if not path: return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Load failed", str(e)); return

        # Build lookup: filename (lower) → current block (for resolving file_abs/file_rel)
        file_lookup = {}
        for b in self._session_blocks:
            file_lookup[Path(b.file_rel).name.lower()] = b

        self._push_undo(); self._commit_undo()

        new_blocks = []
        matched = 0
        for entry in data.get("blocks", []):
            fname_key = Path(entry.get("file_rel",
                             entry.get("filename", ""))).name.lower()
            orig = file_lookup.get(fname_key)
            if orig is None:
                continue
            saved_dur = entry.get("dur")
            if saved_dur:
                dur = float(saved_dur)
            else:
                try:    dur = sf.info(orig.file_abs).duration
                except: dur = orig.dur
            new_blocks.append(SessionBlock(
                bid       = str(uuid.uuid4())[:8],
                slot_name = entry.get("slot_name", orig.slot_name),
                file_abs  = orig.file_abs,
                file_rel  = orig.file_rel,
                track     = entry.get("track", 0),
                start_sec = entry.get("start_sec", 0.0),
                dur       = dur,
                gain_db   = entry.get("gain_db", 0.0),
                loop      = entry.get("loop", False),
                trim_in   = entry.get("trim_in", 0.0),
                trim_out  = entry.get("trim_out", dur),
                fade_in   = entry.get("fade_in", 0.0),
                fade_out  = entry.get("fade_out", 0.0),
                color     = entry.get("color", BLOCK_COLS[0]),
            ))
            matched += 1

        self._session_blocks = new_blocks
        self._sel_blocks     = set()
        self._sel_block      = None
        self._pending_bids   = set()
        self._tl_clear_editor()
        if "muted_tracks" in data:
            self._tl_muted_tracks = set(data["muted_tracks"])
        if "soloed_tracks" in data:
            self._tl_soloed_tracks = set(data["soloed_tracks"])
        self._tl_draw()
        self.status_var.set(
            f"Template loaded: {Path(path).name}  "
            f"({matched}/{len(data.get('blocks', []))} blocks)")


if __name__ == "__main__":
    app = App()
    app.mainloop()
