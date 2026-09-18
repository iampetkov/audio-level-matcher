# Audio Level Matcher

A macOS desktop tool for batch-matching audio asset loudness between 
two game projects.

Analyses LUFS and peak levels across one game's audio library and 
applies matched levels to the corresponding assets of a second game.

## What it does

- Analyses source audio files using adaptive loudness strategies:
  - Peak + RMS for short files
  - RMS + LUFS blend for mid-length files
  - Integrated LUFS (EBU R128) for longer files
- Applies matched levels to target files in bulk
- Per-file manual controls: level, trim, fade in, fade out (draggable handles)
- Waveform display per file
- Per-file playback preview
- Folder path persistence between sessions
- Packaged as a standalone macOS .app

## Why I built it

Manually matching loudness across two game audio libraries is time consuming 
and tedious. This tool does it in bulk with the option to 
fine-tune individual files before export.

## Requirements

- macOS
- Python 3.x / Tkinter
- sox (for playback)

## Screenshot

<img width="982" height="1027" alt="image" src="https://github.com/user-attachments/assets/377f2f14-7f10-4b18-aac4-5fa9eb9d5117" />
<img width="975" height="937" alt="image" src="https://github.com/user-attachments/assets/3c0a443f-b253-45e8-bdd2-0ad8582d4c25" />
