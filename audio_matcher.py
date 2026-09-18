#!/usr/bin/env python3
"""
Audio Level Matcher — Game A → Game B  (MVP3)
Adds: stacked waveform display, trim in/out handles, S-curve fade in/out.
"""

import os, sys, threading, subprocess, time, tempfile, shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from dataclasses import dataclass
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

def _to_dbfs(v):
    return max(20.0 * np.log10(v), SILENCE_DBFS) if v > 0 else SILENCE_DBFS

def _from_db(db):
    return 10.0 ** (db / 20.0)

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
    return {str(p.relative_to(root)).lower(): str(p)
            for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS}

def match_files(a_dir, b_dir):
    a, b = find_audio_files(a_dir), find_audio_files(b_dir)
    return sorted([(rel, a[rel], b[rel]) for rel in b if rel in a])

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

    def __init__(self, parent, on_change=None,
                 external_ref_canvas=None, external_out_canvas=None,
                 single_canvas=False, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_change = on_change
        self._ext_ref_cv   = external_ref_canvas
        self._ext_out_cv   = external_out_canvas
        self._single_canvas = single_canvas

        self._ref_dur    = 1.0
        self._out_dur    = 1.0
        self._total_dur  = 1.0
        self._ref_wf     = None
        self._out_wf     = None
        self._trim_in_s  = 0.0
        self._trim_out_s = 1.0
        self._fade_in_s  = 0.0
        self._fade_out_s = 0.0
        self._ref_ph_s   = 0.0
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

        # Use external canvases if provided, else create own
        if self._ext_ref_cv is not None:
            self._ref_canvas = self._ext_ref_cv
        else:
            self._ref_canvas = tk.Canvas(self, height=CANVAS_H, bg="#f5f5f5",
                                         highlightthickness=1,
                                         highlightbackground="#bbb")
            self._ref_canvas.pack(fill="x", pady=(0,1))

        if self._ext_out_cv is not None:
            self._out_canvas = self._ext_out_cv
        else:
            self._out_canvas = tk.Canvas(self, height=CANVAS_H, bg="#eef3fb",
                                         highlightthickness=1,
                                         highlightbackground="#bbb")
            if not self._single_canvas:
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

    def set_out_only(self, out_path, out_dur):
        """Load only the output (blue) waveform; leave reference (green) blank."""
        self._ref_wf     = None
        self._out_dur    = max(out_dur, 0.001)
        self._ref_dur    = self._out_dur
        self._total_dur  = self._out_dur
        self._trim_in_s  = 0.0
        self._trim_out_s = self._out_dur
        self._fade_in_s  = 0.0
        self._fade_out_s = 0.0
        self._ref_ph_s   = 0.0
        self._out_ph_s   = 0.0
        self._sync_entries()
        def _load():
            try:    self._out_wf = build_waveform(out_path)
            except: self._out_wf = None
            self.after(0, self._draw)
        threading.Thread(target=_load, daemon=True).start()

    def set_ref_only(self, ref_path, ref_dur):
        """Load only the reference (green) waveform; keep output (blue) as-is."""
        self._ref_dur   = max(ref_dur, 0.001)
        self._total_dur = max(self._ref_dur, self._out_dur)
        self._ref_ph_s  = 0.0
        def _load():
            try:    self._ref_wf = build_waveform(ref_path)
            except: self._ref_wf = None
            self.after(0, self._draw)
        threading.Thread(target=_load, daemon=True).start()

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

# ── Main App ───────────────────────────────────────────────────────────────────

PAD = 16

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Level Matcher")
        self.geometry("980x1000")
        self.minsize(820, 800)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.game_a_var = tk.StringVar()
        self.game_b_var = tk.StringVar()
        self.output_var = tk.StringVar()

        self._prefs_path = os.path.expanduser(
            "~/Library/Application Support/AudioLevelMatcher/prefs.json")
        self._load_prefs()
        self.status_var = tk.StringVar(value="Select folders above, then click Scan & Preview.")
        self.progress   = tk.DoubleVar(value=0.0)
        self.loop_var    = tk.BooleanVar(value=False)
        self.trim_var    = tk.DoubleVar(value=0.0)
        self.um_trim_var = tk.DoubleVar(value=0.0)

        self._running         = False
        self._row_ids         = []        # matched pairs
        self._selected        = None      # index into _row_ids OR None
        self._selected_unmatched = None   # index into _unmatched
        self._active_source   = None      # "matched" | "unmatched"
        self._unmatched       = []        # [(src_path, rel)]
        self._ref_paths       = []        # paths for reference listbox
        self._trim_after_id   = None
        self._preview_playing = False
        self._unmatched_gains = {}        # rel → gain_db (manual)

        self._player_ref = Player()
        self._player_out = Player()
        self._player_um  = Player()
        self._player_ref.on_stop = lambda: self.after(0, self._on_ref_stopped)
        self._player_out.on_stop = lambda: self.after(0, self._on_out_stopped)
        self._player_um.on_stop  = lambda: self.after(0, self._on_um_stopped)
        self._um_preview_playing = False
        self._um_trim_after_id   = None

        self._build()
        self._poll_playback()

    def _on_close(self):
        self._player_ref.stop(save_position=False)
        self._player_out.stop(save_position=False)
        self._player_um.stop(save_position=False)
        _cleanup_temp()
        self.destroy()

    # ── Layout ─────────────────────────────────────────────────────────────

    def _load_prefs(self):
        try:
            import json
            with open(self._prefs_path) as f:
                p = json.load(f)
            if p.get("game_a"): self.game_a_var.set(p["game_a"])
            if p.get("game_b"): self.game_b_var.set(p["game_b"])
            if p.get("output"): self.output_var.set(p["output"])
        except Exception:
            pass

    def _save_prefs(self):
        try:
            import json
            os.makedirs(os.path.dirname(self._prefs_path), exist_ok=True)
            with open(self._prefs_path, "w") as f:
                json.dump({
                    "game_a": self.game_a_var.get(),
                    "game_b": self.game_b_var.get(),
                    "output": self.output_var.get(),
                }, f)
        except Exception:
            pass

    def _sep(self, parent, h=8):
        tk.Frame(parent, height=h).pack(fill="x")

    def _build(self):
        # Scrollable outer frame
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        canvas_scroll = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical",
                                   command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas_scroll.pack(side="left", fill="both", expand=True)

        outer = tk.Frame(canvas_scroll, padx=PAD, pady=PAD)
        win_id = canvas_scroll.create_window((0,0), window=outer, anchor="nw")

        def _on_frame_configure(e):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        def _on_canvas_configure(e):
            canvas_scroll.itemconfig(win_id, width=e.width)
        outer.bind("<Configure>", _on_frame_configure)
        canvas_scroll.bind("<Configure>", _on_canvas_configure)

        # Two-finger scroll — bound to all widgets
        def _on_mousewheel(e):
            canvas_scroll.yview_scroll(int(-1*(e.delta/120)), "units")
        def _on_trackpad(e):
            canvas_scroll.yview_scroll(int(-1*(e.delta/10)), "units")
        # Bind to root window so works anywhere on screen
        def _scroll(e):
            delta = e.delta
            if abs(delta) >= 120:
                units = int(-1 * delta / 120)
            else:
                units = int(-1 * delta / 8) or (-1 if delta > 0 else 1)
            canvas_scroll.yview_scroll(units, "units")
            return "break"

        # Recursively bind scroll to every widget, including those added later
        def _bind_all_scroll(widget):
            try:
                widget.bind("<MouseWheel>", _scroll)
            except Exception:
                pass
            for child in widget.winfo_children():
                _bind_all_scroll(child)

        # Bind now and again after 200ms (after all widgets are rendered)
        self.bind_all("<MouseWheel>", _scroll)
        self.after(200,  lambda: _bind_all_scroll(self))
        self.after(1000, lambda: _bind_all_scroll(self))

        self._build_content(outer)

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

        # ═══════════════════════════════════════════════════════════════
        # ── MATCHED PAIR SECTION ─────────────────────────────────────────
        # ═══════════════════════════════════════════════════════════════

        matched_frame = tk.Frame(outer, bd=1, relief="groove", padx=10, pady=8)
        matched_frame.pack(fill="x")

        # Header with Loop toggle
        hdr = tk.Frame(matched_frame)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Matched Pair",
                 font=("Helvetica", 11, "bold"), anchor="w").pack(side="left")
        tk.Checkbutton(hdr, text="Loop",
                       variable=self.loop_var,
                       font=("Helvetica", 11)).pack(side="right")
        self._sep(matched_frame, 4)

        # Reference waveform + Play button
        tk.Label(matched_frame, text="Reference",
                 font=("Helvetica", 10, "bold"), fg="#3a7a3a",
                 anchor="w").pack(fill="x")
        ref_wf_row = tk.Frame(matched_frame)
        ref_wf_row.pack(fill="x")
        self._ref_wf_canvas = tk.Canvas(ref_wf_row, height=90, bg="#f5f5f5",
                                         highlightthickness=1,
                                         highlightbackground="#bbb")
        self._ref_wf_canvas.pack(side="left", fill="x", expand=True)
        ref_right = tk.Frame(ref_wf_row, width=90)
        ref_right.pack(side="right", fill="y", padx=(4,0))
        ref_right.pack_propagate(False)
        self._ref_name_lbl = tk.Label(ref_right, text="—", anchor="w",
                                       font=("Helvetica", 9), fg="#444",
                                       wraplength=80)
        self._ref_name_lbl.pack(fill="x")
        self._ref_time_lbl = tk.Label(ref_right, text="0:00",
                                       font=("Helvetica", 9), fg="#666")
        self._ref_time_lbl.pack()
        self._ref_play_btn = tk.Button(ref_right, text="▶ Play",
                                        font=("Helvetica", 10), state="disabled",
                                        command=self._toggle_ref)
        self._ref_play_btn.pack(fill="x", pady=(4,0))

        self._sep(matched_frame, 4)

        # Output waveform + Play button
        tk.Label(matched_frame, text="Output",
                 font=("Helvetica", 10, "bold"), fg="#1a5fa8",
                 anchor="w").pack(fill="x")
        out_wf_row = tk.Frame(matched_frame)
        out_wf_row.pack(fill="x")
        self._out_wf_canvas = tk.Canvas(out_wf_row, height=90, bg="#eef3fb",
                                         highlightthickness=1,
                                         highlightbackground="#bbb")
        self._out_wf_canvas.pack(side="left", fill="x", expand=True)
        out_right = tk.Frame(out_wf_row, width=90)
        out_right.pack(side="right", fill="y", padx=(4,0))
        out_right.pack_propagate(False)
        self._out_name_lbl = tk.Label(out_right, text="—", anchor="w",
                                       font=("Helvetica", 9), fg="#444",
                                       wraplength=80)
        self._out_name_lbl.pack(fill="x")
        self._out_time_lbl = tk.Label(out_right, text="0:00",
                                       font=("Helvetica", 9), fg="#666")
        self._out_time_lbl.pack()
        self._out_play_btn = tk.Button(out_right, text="▶ Play",
                                        font=("Helvetica", 10), state="disabled",
                                        command=self._toggle_out)
        self._out_play_btn.pack(fill="x", pady=(4,0))

        # Shared ruler + handles strip — embed WaveformPanel but use its canvases
        self._sep(matched_frame, 2)
        self._wf_panel = WaveformPanel(matched_frame, on_change=self._on_wf_change,
                                        bd=0, relief="flat",
                                        external_ref_canvas=self._ref_wf_canvas,
                                        external_out_canvas=self._out_wf_canvas)
        self._wf_panel.pack(fill="x")

        self._sep(matched_frame, 6)

        # Gain trim row
        trim_row = tk.Frame(matched_frame)
        trim_row.pack(fill="x", pady=2)
        tk.Label(trim_row, text="Gain trim:", width=10, anchor="w",
                 font=("Helvetica", 11)).pack(side="left")
        self._trim_slider = tk.Scale(trim_row, from_=-12.0, to=12.0,
                                     resolution=0.1, orient="horizontal",
                                     variable=self.trim_var,
                                     font=("Helvetica", 10), length=260,
                                     showvalue=False, state="disabled",
                                     command=self._on_trim_drag)
        self._trim_slider.pack(side="left", padx=(0,6))
        self._trim_lbl = tk.Label(trim_row, text="0.0 dB", width=7,
                                   font=("Helvetica", 11, "bold"), fg="#333")
        self._trim_lbl.pack(side="left")
        self._preview_lbl = tk.Label(trim_row, text="", width=8,
                                      font=("Helvetica", 10), fg="#888")
        self._preview_lbl.pack(side="left", padx=(6,0))
        self._apply_btn = tk.Button(trim_row, text="✓ Apply & Save",
                                     font=("Helvetica", 10, "bold"),
                                     padx=10, state="disabled",
                                     command=self._apply_all)
        self._apply_btn.pack(side="right", padx=(6,0))
        self._reset_btn = tk.Button(trim_row, text="↺ Reset",
                                     font=("Helvetica", 10),
                                     padx=8, state="disabled",
                                     command=self._reset_all)
        self._reset_btn.pack(side="right", padx=(6,0))

        self._sep(outer, 12)

        # ═══════════════════════════════════════════════════════════════
        # ── UNMATCHED FILE SECTION ────────────────────────────────────
        # ═══════════════════════════════════════════════════════════════

        tk.Label(outer, text="Unmatched Game B Files  —  no Game A reference found",
                 font=("Helvetica", 11, "bold"), fg="#884400", anchor="w").pack(fill="x")
        self._sep(outer, 4)

        unmatched_frame = tk.Frame(outer, bd=1, relief="groove", padx=10, pady=8)
        unmatched_frame.pack(fill="x")

        # Unmatched waveform + Play button
        tk.Label(unmatched_frame, text="Unmatched file",
                 font=("Helvetica", 10, "bold"), fg="#884400",
                 anchor="w").pack(fill="x")
        um_wf_row = tk.Frame(unmatched_frame)
        um_wf_row.pack(fill="x")
        self._um_wf_canvas = tk.Canvas(um_wf_row, height=90, bg="#fff8f0",
                                        highlightthickness=1,
                                        highlightbackground="#cc9966")
        self._um_wf_canvas.pack(side="left", fill="x", expand=True)
        um_right = tk.Frame(um_wf_row, width=90)
        um_right.pack(side="right", fill="y", padx=(4,0))
        um_right.pack_propagate(False)
        self._um_file_lbl = tk.Label(um_right, text="—", anchor="w",
                                      font=("Helvetica", 9), fg="#444",
                                      wraplength=80)
        self._um_file_lbl.pack(fill="x")
        self._um_time_lbl = tk.Label(um_right, text="0:00",
                                      font=("Helvetica", 9), fg="#666")
        self._um_time_lbl.pack()
        self._um_play_btn = tk.Button(um_right, text="▶ Play",
                                       font=("Helvetica", 10), state="disabled",
                                       command=self._toggle_um)
        self._um_play_btn.pack(fill="x", pady=(4,0))

        self._sep(unmatched_frame, 2)
        self._um_wf_panel = WaveformPanel(unmatched_frame,
                                           on_change=self._on_um_wf_change,
                                           bd=0, relief="flat",
                                           external_ref_canvas=self._um_wf_canvas,
                                           external_out_canvas=self._um_wf_canvas,
                                           single_canvas=True)
        self._um_wf_panel.pack(fill="x")

        self._sep(unmatched_frame, 6)

        # Unmatched gain trim row
        um_trim_row = tk.Frame(unmatched_frame)
        um_trim_row.pack(fill="x", pady=2)
        tk.Label(um_trim_row, text="Gain trim:", width=10, anchor="w",
                 font=("Helvetica", 11)).pack(side="left")
        self._um_trim_slider = tk.Scale(um_trim_row, from_=-12.0, to=12.0,
                                         resolution=0.1, orient="horizontal",
                                         variable=self.um_trim_var,
                                         font=("Helvetica", 10), length=260,
                                         showvalue=False, state="disabled",
                                         command=self._on_um_trim_drag)
        self._um_trim_slider.pack(side="left", padx=(0,6))
        self._um_trim_lbl = tk.Label(um_trim_row, text="0.0 dB", width=7,
                                      font=("Helvetica", 11, "bold"), fg="#333")
        self._um_trim_lbl.pack(side="left")
        self._um_preview_lbl = tk.Label(um_trim_row, text="", width=8,
                                         font=("Helvetica", 10), fg="#888")
        self._um_preview_lbl.pack(side="left", padx=(6,0))
        self._um_apply_btn = tk.Button(um_trim_row, text="✓ Apply & Save",
                                        font=("Helvetica", 10, "bold"),
                                        padx=10, state="disabled",
                                        command=self._apply_unmatched)
        self._um_apply_btn.pack(side="right", padx=(6,0))
        self._um_reset_btn = tk.Button(um_trim_row, text="↺ Reset",
                                        font=("Helvetica", 10),
                                        padx=8, state="disabled",
                                        command=self._reset_unmatched)
        self._um_reset_btn.pack(side="right", padx=(6,0))

        self._sep(outer, 10)

        # File lists
        lists_frame = tk.Frame(outer)
        lists_frame.pack(fill="both", expand=False)

        # Left: unmatched files list
        left = tk.Frame(lists_frame)
        left.pack(side="left", fill="both", expand=True, padx=(0,6))
        tk.Label(left, text="Game B files (no match):",
                 font=("Helvetica", 10, "bold"), anchor="w").pack(fill="x")
        um_frame = tk.Frame(left, bd=1, relief="sunken")
        um_frame.pack(fill="both", expand=True)
        self._um_list = tk.Listbox(um_frame, font=("Helvetica", 11),
                                    selectmode="single", height=6,
                                    activestyle="none")
        um_vsb = ttk.Scrollbar(um_frame, orient="vertical",
                                command=self._um_list.yview)
        self._um_list.configure(yscrollcommand=um_vsb.set)
        um_vsb.pack(side="right", fill="y")
        self._um_list.pack(fill="both", expand=True)
        self._um_list.bind("<<ListboxSelect>>", self._on_unmatched_select)

        # Right: processed output files (reference)
        right = tk.Frame(lists_frame)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="Processed output files (play for reference):",
                 font=("Helvetica", 10, "bold"), anchor="w").pack(fill="x")
        ref_frame = tk.Frame(right, bd=1, relief="sunken")
        ref_frame.pack(fill="both", expand=True)
        self._ref_list = tk.Listbox(ref_frame, font=("Helvetica", 11),
                                     selectmode="single", height=6,
                                     activestyle="none")
        ref_vsb = ttk.Scrollbar(ref_frame, orient="vertical",
                                  command=self._ref_list.yview)
        self._ref_list.configure(yscrollcommand=ref_vsb.set)
        ref_vsb.pack(side="right", fill="y")
        self._ref_list.pack(fill="both", expand=True)
        self._ref_list.bind("<<ListboxSelect>>", self._on_ref_list_select)

        self._sep(outer, 6)

        ttk.Progressbar(outer, variable=self.progress,
                        maximum=100, mode="determinate").pack(fill="x")
        self._sep(outer, 4)
        tk.Label(outer, textvariable=self.status_var,
                 font=("Helvetica", 10), fg="#555", anchor="w").pack(fill="x")

    # ── Folder browsing ────────────────────────────────────────────────────

    def _browse_a(self):
        initial = self.game_a_var.get().strip()
        initial = initial if os.path.isdir(initial) else os.path.expanduser("~")
        d = filedialog.askdirectory(title="Select Game A audio folder",
                                    initialdir=initial)
        if d:
            self.game_a_var.set(d)
            self._save_prefs()

    def _browse_b(self):
        initial = self.game_b_var.get().strip()
        initial = initial if os.path.isdir(initial) else os.path.expanduser("~")
        d = filedialog.askdirectory(title="Select Game B audio folder",
                                    initialdir=initial)
        if d:
            self.game_b_var.set(d)
            out = str(Path(d).parent / "01 Master" / "wav")
            self.output_var.set(out)
            self._save_prefs()

    def _browse_out(self):
        initial = self.output_var.get().strip()
        p = Path(initial)
        while str(p) != p.root and not p.exists():
            p = p.parent
        initial = str(p) if p.exists() else os.path.expanduser("~")
        d = filedialog.askdirectory(title="Select output folder",
                                    initialdir=initial)
        if d:
            self.output_var.set(d)
            self._save_prefs()

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
        self._save_prefs()
        self._stop_all(save_position=False)
        self.status_var.set("Scanning…"); self.update()
        pairs = match_files(a, b)
        self.tree.delete(*self.tree.get_children())
        self._row_ids = []
        self._selected = None
        self._selected_unmatched = None
        self._active_source = None
        self._clear_player_panel()
        self._wf_panel.clear()
        self._um_list.delete(0, "end")
        self._ref_list.delete(0, "end")
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

        # Find unmatched Game B files
        a_files = find_audio_files(a)
        b_files = find_audio_files(b)
        self._unmatched = [(b_files[rel], rel)
                           for rel in b_files if rel not in a_files]
        self._unmatched.sort(key=lambda x: x[1])
        self._um_list.delete(0, "end")
        self._unmatched_gains = {}
        for _, rel in self._unmatched:
            self._um_list.insert("end", Path(rel).name)
        if self._unmatched:
            self.status_var.set(
                self.status_var.get() +
                f"  |  {len(self._unmatched)} unmatched file(s) below.")

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
        self.after(150, self._refresh_ref_list)

    # ── Row selection ──────────────────────────────────────────────────────

    def _on_row_select(self, event=None):
        self._stop_all(save_position=False)
        self._cancel_trim_debounce()
        sel = self.tree.selection()
        if not sel:
            self._selected = None
            self._active_source = None
            self._clear_player_panel()
            self._wf_panel.clear()
            return
        self._selected_unmatched = None
        self._active_source = "matched"
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
        if self._active_source == "unmatched" and self._selected_unmatched is not None:
            src_path, rel = self._unmatched[self._selected_unmatched]
            out_dir  = self.output_var.get().strip()
            out_path = str(Path(out_dir) / rel) if out_dir else src_path
            return src_path, out_path
        if self._selected is None: return None, None
        _, ref_path, _, rel, _ = self._row_ids[self._selected]
        out_dir  = self.output_var.get().strip()
        out_path = str(Path(out_dir) / rel) if out_dir else None
        return ref_path, out_path

    def _get_duration(self):
        if self._active_source == "unmatched" and self._selected_unmatched is not None:
            try:
                src_path, _ = self._unmatched[self._selected_unmatched]
                return sf.info(src_path).duration
            except: return 1.0
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
        trim = self.trim_var.get()
        ti   = self._wf_panel.trim_in
        to   = self._wf_panel.trim_out
        fi   = self._wf_panel.fade_in_dur
        fo   = self._wf_panel.fade_out_dur

        # Get source path and base gain depending on active mode
        if self._active_source == "unmatched" and self._selected_unmatched is not None:
            src_path, rel = self._unmatched[self._selected_unmatched]
            base_gain = 0.0
            # For unmatched, always build a preview from source
            try:
                tmp = _make_preview(src_path, base_gain + trim, ti, to, fi, fo)
                preview_dur = to - ti
                self._player_out.play(tmp, preview_dur, start_pos=start_pos)
                self._preview_playing = True
                self._preview_lbl.config(text="⚡ preview")
            except Exception as e:
                self._player_out.play(src_path, dur, start_pos=start_pos)
                self._preview_playing = False
            self._out_play_btn.config(text="■ Stop")
            return

        # Matched file path
        _, out_path = self._get_paths()
        if not out_path or not os.path.isfile(out_path): return
        if self._selected is None: return
        _, _, src_path, _, base_gain = self._row_ids[self._selected]

        has_edits = (abs(trim) > 0.05 or ti > 0.01 or
                     to < self._get_duration() - 0.01 or fi > 0.01 or fo > 0.01)
        if has_edits:
            try:
                tmp = _make_preview(src_path, base_gain + trim, ti, to, fi, fo)
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

    def _stop_um(self, save_position=False):
        self._player_um.stop(save_position=save_position)
        self._um_play_btn.config(text="▶ Play")
        self._um_preview_playing = False
        self._um_preview_lbl.config(text="")

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
        if self._player_um.is_playing():
            pos = self._player_um.get_position()
            self._um_time_lbl.config(text=fmt(pos))
            self._um_wf_panel.set_out_playhead(pos)
        self.after(100, self._poll_playback)

    # ── Waveform handle change ─────────────────────────────────────────────

    def _on_wf_change(self):
        """Called when trim/fade handles are dragged — restart preview."""
        if self._trim_after_id:
            self.after_cancel(self._trim_after_id)
        self._trim_after_id = self.after(300, self._restart_out_preview)

    def _restart_out_preview(self):
        self._trim_after_id = None
        is_unmatched = (self._active_source == "unmatched" and
                        self._selected_unmatched is not None)
        if self._player_out.is_playing() or is_unmatched:
            pos = self._player_out.get_position() if self._player_out.is_playing() else 0.0
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
        is_unmatched = (self._active_source == "unmatched" and
                        self._selected_unmatched is not None)
        if self._player_out.is_playing() or is_unmatched:
            pos = self._player_out.get_position() if self._player_out.is_playing() else 0.0
            self._player_out.stop(save_position=False)
            self._play_out_with_current_settings(pos, self._get_duration())

    def _cancel_trim_debounce(self):
        if self._trim_after_id:
            self.after_cancel(self._trim_after_id)
            self._trim_after_id = None

    # ── Unmatched player controls ─────────────────────────────────────

    def _toggle_um(self):
        if self._player_um.is_playing():
            self._player_um.stop(save_position=self.loop_var.get())
            self._um_play_btn.config(text="▶ Play")
        else:
            if self._selected_unmatched is None: return
            src_path, _ = self._unmatched[self._selected_unmatched]
            try: dur = sf.info(src_path).duration
            except: dur = 1.0
            resume = self._player_um.get_resume_pos() if self.loop_var.get() else 0.0
            self._play_um_with_settings(resume, dur)

    def _on_um_stopped(self):
        self._um_play_btn.config(text="▶ Play")
        self._um_preview_playing = False
        self._um_preview_lbl.config(text="")
        if self.loop_var.get() and self._selected_unmatched is not None:
            src_path, _ = self._unmatched[self._selected_unmatched]
            try: dur = sf.info(src_path).duration
            except: dur = 1.0
            self._play_um_with_settings(0.0, dur)

    def _play_um_with_settings(self, start_pos, dur):
        if self._selected_unmatched is None: return
        src_path, _ = self._unmatched[self._selected_unmatched]
        trim = self.um_trim_var.get()
        ti   = self._um_wf_panel.trim_in
        to   = self._um_wf_panel.trim_out
        fi   = self._um_wf_panel.fade_in_dur
        fo   = self._um_wf_panel.fade_out_dur
        has_edits = (abs(trim) > 0.05 or ti > 0.01 or
                     to < dur - 0.01 or fi > 0.01 or fo > 0.01)
        if has_edits:
            try:
                tmp = _make_preview(src_path, trim, ti, to, fi, fo)
                self._player_um.play(tmp, to - ti, start_pos=start_pos)
                self._um_preview_playing = True
                self._um_preview_lbl.config(text="⚡ preview")
            except:
                self._player_um.play(src_path, dur, start_pos=start_pos)
                self._um_preview_playing = False
        else:
            self._player_um.play(src_path, dur, start_pos=start_pos)
            self._um_preview_playing = False
        self._um_play_btn.config(text="■ Stop")

    def _on_um_wf_change(self):
        if self._um_trim_after_id:
            self.after_cancel(self._um_trim_after_id)
        self._um_trim_after_id = self.after(300, self._restart_um_preview)

    def _restart_um_preview(self):
        self._um_trim_after_id = None
        if self._selected_unmatched is None: return
        src_path, _ = self._unmatched[self._selected_unmatched]
        try: dur = sf.info(src_path).duration
        except: dur = 1.0
        pos = self._player_um.get_position() if self._player_um.is_playing() else 0.0
        self._player_um.stop(save_position=False)
        self._play_um_with_settings(pos, dur)

    def _on_um_trim_drag(self, val):
        v = float(val)
        self._um_trim_lbl.config(text=f"{v:+.1f} dB" if v != 0 else "0.0 dB")
        if self._um_trim_after_id:
            self.after_cancel(self._um_trim_after_id)
        self._um_trim_after_id = self.after(300, self._restart_um_preview)

    def _apply_unmatched(self):
        if self._selected_unmatched is None: return
        src_path, rel = self._unmatched[self._selected_unmatched]
        out_dir = self.output_var.get().strip()
        if not out_dir:
            messagebox.showwarning("No output folder", "Please select an output folder."); return
        out_path = str(Path(out_dir) / rel)
        trim = self.um_trim_var.get()
        ti = self._um_wf_panel.trim_in
        to = self._um_wf_panel.trim_out
        fi = self._um_wf_panel.fade_in_dur
        fo = self._um_wf_panel.fade_out_dur
        self._stop_um()
        try:
            apply_and_save(src_path, out_path, trim, ti, to, fi, fo)
            self._refresh_ref_list()
            # Promote to matched table
            rid = self.tree.insert("", "end", tags=("done",), values=(
                Path(rel).name,
                f"{sf.info(out_path).duration:.2f}s",
                "manual", "—", "manual",
                f"{trim:+.1f}", "✓ saved",
            ))
            self._row_ids.append((rid, src_path, src_path, rel, trim))
            self._unmatched.pop(self._selected_unmatched)
            self._um_list.delete(self._selected_unmatched)
            self._selected_unmatched = None
            self._um_file_lbl.config(text="—")
            self._um_time_lbl.config(text="0:00")
            self._um_play_btn.configure(state="disabled")
            self._um_trim_slider.configure(state="disabled")
            self._um_apply_btn.configure(state="disabled")
            self._um_reset_btn.configure(state="disabled")
            self.um_trim_var.set(0.0)
            self._um_trim_lbl.config(text="0.0 dB")
            self._um_wf_panel.clear()
            self.status_var.set(
                f"Saved {Path(rel).name}  gain {trim:+.1f} dB  "
                f"trim {ti:.2f}s–{to:.2f}s  fade-in {fi:.2f}s  fade-out {fo:.2f}s")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save: {e}")

    def _reset_unmatched(self):
        if self._um_trim_after_id:
            self.after_cancel(self._um_trim_after_id)
        self.um_trim_var.set(0.0)
        self._um_trim_lbl.config(text="0.0 dB")
        self._um_preview_lbl.config(text="")
        self._um_wf_panel.reset_handles()
        if self._selected_unmatched is not None:
            src_path, _ = self._unmatched[self._selected_unmatched]
            try: dur = sf.info(src_path).duration
            except: dur = 1.0
            if self._player_um.is_playing():
                self._player_um.stop(save_position=False)
                self._player_um.play(src_path, dur, start_pos=0.0)
                self._um_play_btn.config(text="■ Stop")

    # ── Unmatched file handlers ───────────────────────────────────────────

    def _refresh_ref_list(self):
        """Rebuild the processed output files reference list."""
        out_dir = self.output_var.get().strip()
        self._ref_list.delete(0, "end")
        self._ref_paths = []
        if not out_dir or not os.path.isdir(out_dir):
            return
        for p in sorted(Path(out_dir).rglob("*")):
            if p.suffix.lower() in AUDIO_EXTENSIONS:
                self._ref_list.insert("end", p.name)
                self._ref_paths.append(str(p))

    def _on_unmatched_select(self, event=None):
        sel = self._um_list.curselection()
        if not sel:
            return
        idx = sel[0]
        src_path, rel = self._unmatched[idx]
        self._selected_unmatched = idx

        # Stop unmatched player only (leave matched pair playing)
        self._stop_um()

        fname = Path(rel).name
        self._um_file_lbl.config(text=fname)
        self._um_time_lbl.config(text="0:00")
        self._um_play_btn.configure(state="normal")
        self._um_trim_slider.configure(state="normal")
        self._um_apply_btn.configure(state="normal")
        self._um_reset_btn.configure(state="normal")

        # Restore saved gain
        gain = self._unmatched_gains.get(rel, 0.0)
        self.um_trim_var.set(gain)
        self._um_trim_lbl.config(text=f"{gain:+.1f} dB" if gain != 0 else "0.0 dB")
        self._um_preview_lbl.config(text="")

        # Load waveform into unmatched panel
        try:
            dur = sf.info(src_path).duration
        except:
            dur = 1.0
        self._um_wf_panel.set_out_only(src_path, dur)

        # Auto-play
        self._player_um.play(src_path, dur, start_pos=0.0)
        self._um_play_btn.config(text="■ Stop")

    def _on_ref_list_select(self, event=None):
        """Play a processed output file in the matched ref player for comparison."""
        sel = self._ref_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._ref_paths):
            return
        ref_path = self._ref_paths[idx]
        try:
            dur = sf.info(ref_path).duration
        except:
            dur = 1.0
        # Load into green (reference) waveform of matched panel
        self._wf_panel.set_ref_only(ref_path, dur)
        self._ref_name_lbl.config(text=Path(ref_path).name)
        self._ref_play_btn.configure(state="normal")
        # Play in ref player
        self._player_out.stop(save_position=False)
        self._out_play_btn.config(text="▶ Play")
        self._player_ref.stop(save_position=False)
        self._player_ref.play(ref_path, dur, start_pos=0.0)
        self._ref_play_btn.config(text="■ Stop")

    def _get_unmatched_paths(self):
        if self._selected_unmatched is None:
            return None
        return self._unmatched[self._selected_unmatched]  # (src_path, rel)

    # ── Apply & Save ───────────────────────────────────────────────────────

    def _apply_all(self):
        trim = self.trim_var.get()
        out_dir = self.output_var.get().strip()
        if not out_dir:
            messagebox.showwarning("No output folder",
                                   "Please select an output folder first."); return

        # Determine source — matched or unmatched
        if self._active_source == "unmatched" and self._selected_unmatched is not None:
            src_path, rel = self._get_unmatched_paths()
            ref_path = src_path
            base_gain = 0.0
            total_gain = trim
            out_path = str(Path(out_dir) / rel)
            is_unmatched = True
        elif self._selected is not None:
            _, ref_path, src_path, rel, base_gain = self._row_ids[self._selected]
            out_path = str(Path(out_dir) / rel)
            if not os.path.isfile(out_path):
                messagebox.showwarning("File not found",
                                       "Run Process Files first."); return
            total_gain = base_gain + trim
            is_unmatched = False
        else:
            return
        ti = self._wf_panel.trim_in
        to = self._wf_panel.trim_out
        fi = self._wf_panel.fade_in_dur
        fo = self._wf_panel.fade_out_dur

        self._stop_all(save_position=False)
        try:
            apply_and_save(src_path, out_path, total_gain, ti, to, fi, fo)

            self._refresh_ref_list()

            if is_unmatched:
                # Promote to matched table
                self._unmatched_gains[rel] = total_gain
                rid = self.tree.insert("", "end", tags=("done",), values=(
                    Path(rel).name,
                    f"{sf.info(out_path).duration:.2f}s",
                    "manual", "—", "manual",
                    f"{total_gain:+.1f}", "✓ saved",
                ))
                self._row_ids.append((rid, src_path, src_path, rel, total_gain))
                self._unmatched.pop(self._selected_unmatched)
                self._um_list.delete(self._selected_unmatched)
                self._selected_unmatched = None
                self._active_source = None
                self._clear_player_panel()
                self._wf_panel.clear()
            else:
                # Update matched table row
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


if __name__ == "__main__":
    app = App()
    app.mainloop()
