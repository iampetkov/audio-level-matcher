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
- DAW-like timeline for chaining audio files to audition them in playback scenarios close to how they would be heard in-game.
- Ability to save the playback scenarios as presets for future use with new sets of files.
- Undo/Redo
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

<img width="1999" height="1142" alt="image" src="https://github.com/user-attachments/assets/40ee67ce-a5da-427f-932a-6977630ce646" />
