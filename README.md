# tone_generator

Generates a short tone with clean
raised-cosine on/off ramps and saves it as WAV or MP3.

## Features

- Sine, square, triangle (band-limited, alias-free), or noise waveform
- Multi-tone / harmonic complex (sum several carriers)
- Sinusoidal amplitude modulation (SAM)
- Hann (raised-cosine) onset/offset ramps, independently settable
- Level in dBFS, targeted as peak or RMS
- Mono or stereo, optional second sync/trigger channel
- Leading/trailing silence padding
- WAV (16/24-bit) always available; MP3 export via `pydub`
- 10 s calibration tone at the same frequency/level
- Optional JSON file with parameters/timing
- Optional envelope + spectrum plot

## Requirements

- Python 3.10+
- `numpy`
- `pydub` + `ffmpeg` (only needed for `--format mp3`)
- `matplotlib` (only needed for `--plot`)

## Install

```bash
pip install numpy
python3 tone_generator.py
```

## Usage

```bash
# Default tone: 750 Hz, 75 ms, 10 ms ramps, -12 dBFS
python3 tone_generator.py

# Custom frequency and duration
python3 tone_generator.py --freq 1200 --duration 100

# Square wave, loudness-matched to a sine via RMS
python3 tone_generator.py --waveform square --level-mode rms --amplitude -18

# Multi-tone / harmonic complex (sum of carriers)
python3 tone_generator.py --freq 440 554 659

# 40 Hz amplitude modulation (SAM)
python3 tone_generator.py --freq 750 --am-freq 40 --am-depth 0.8

# MP3 output
python3 tone_generator.py --format mp3
```

Run `python3 tone_generator.py --help` for the full option list.

## Output

Each run writes an audio file. The auto-generated filename encodes frequency,
duration, and ramp (e.g. `tone_750hz_75ms_10msramp.wav`); use
`--out` to name it yourself, and `--force` to overwrite an existing file.
Pass `--metadata` to also write a `<file>.json` recording every parameter used.

`--calibration-tone` additionally writes a `<file>_calib.wav` with a 10 s
steady tone at the same frequency/level/waveform, for metering playback level.

## License

MIT