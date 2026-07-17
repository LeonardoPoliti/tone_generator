#!/usr/bin/env python3
"""
tone_generator.py -- command-line tone generator.

Generates a short tone with raised-cosine on/off ramps and saves it as WAV
(16/24-bit PCM) or MP3.

Features:
  - sine, square, triangle (band-limited, alias-free), or noise waveform
  - multi-tone / harmonic complex: sum several carriers (--freq 440 554 659)
  - sinusoidal amplitude modulation (SAM): --am-freq / --am-depth
  - Hann (raised-cosine) onset/offset ramps, independently settable
  - level in dBFS, targeted as peak or RMS
  - mono or stereo, optional second sync/trigger channel
  - leading/trailing silence padding
  - WAV (16/24-bit) always available; MP3 export via pydub (lazy import)
  - 10 s calibration tone at the same freq/level
  - optional JSON file with parameters/timing (--metadata)
  - optional envelope + spectrum plot

Defaults: 750 Hz, 75 ms, 10 ms ramps, -12 dBFS peak, mono, 44.1 kHz, 16-bit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import wave
from datetime import datetime, timezone

import numpy as np

#----------------------------------------------------------------

PCM_MAX = {16: 32767, 24: 8388607}


def db_to_lin(db: float) -> float:
    """dBFS -> linear amplitude."""
    return 10 ** (db / 20.0)


def lin_to_db(lin: float) -> float:
    """Linear amplitude -> dBFS, floored to avoid log(0)."""
    return 20.0 * float(np.log10(max(float(lin), 1e-12)))


def fmt_num(x: float) -> str:
    """Filename-safe number: 750 -> '750', 12.5 -> '12p5'."""
    s = str(int(x)) if float(x).is_integer() else str(float(x))
    return s.replace(".", "p").replace("-", "neg")


def generate_carrier(freqs: list[float], n: int, sr: int, waveform: str,
                     rng: np.random.Generator, am_freq: float | None = None,
                     am_depth: float = 1.0) -> np.ndarray:
    """Unit-peak carrier. For tonal waveforms, sums one component per entry in
    `freqs` (multi-tone / harmonic complex). Square/triangle are band-limited
    (additive, harmonics below Nyquist only) so they contain no aliased
    inharmonic energy. Optional sinusoidal amplitude modulation (SAM),
    (1 + am_depth * sin(2*pi*am_freq*t)), is applied before peak normalization."""
    t = np.arange(n, dtype=np.float64) / sr
    nyquist = sr / 2.0

    if waveform == "noise":
        x = rng.standard_normal(n)
    else:
        x = np.zeros(n, dtype=np.float64)
        for freq in freqs:
            if waveform == "sine":
                x += np.sin(2 * np.pi * freq * t)
            elif waveform in ("square", "triangle"):
                if freq >= nyquist:
                    sys.exit(f"Error: --freq {freq:g} Hz is at or above Nyquist ({nyquist:g} Hz).")
                k = 1
                while k * freq < nyquist:
                    if waveform == "square":
                        x += np.sin(2 * np.pi * k * freq * t) / k
                    else:
                        x += ((-1) ** ((k - 1) // 2)) * np.sin(2 * np.pi * k * freq * t) / (k * k)
                    k += 2
            else:
                raise ValueError(f"Unknown waveform: {waveform}")

    if am_freq is not None:
        x = x * (1.0 + am_depth * np.sin(2 * np.pi * am_freq * t))

    peak = np.max(np.abs(x))
    return x / peak if peak > 0 else x


def make_envelope(n_total: int, n_in: int, n_out: int) -> np.ndarray:
    """Raised-cosine (Hann half-window) onset/offset envelope."""
    env = np.ones(n_total, dtype=np.float64)
    n_in = min(n_in, n_total)
    n_out = min(n_out, n_total - n_in)

    if n_in > 1:
        env[:n_in] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_in)))
    if n_out > 1:
        env[n_total - n_out:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, n_out)))
    return env


def build_tone(freqs: list[float], n_total: int, n_in: int, n_out: int, sr: int,
               waveform: str, amplitude_db: float, level_mode: str,
               rng: np.random.Generator, am_freq: float | None = None,
               am_depth: float = 1.0) -> np.ndarray:
    """Ramped tone scaled so that its peak (or RMS) equals amplitude_db dBFS."""
    shaped = generate_carrier(freqs, n_total, sr, waveform, rng, am_freq,
                              am_depth) * make_envelope(n_total, n_in, n_out)
    target = db_to_lin(amplitude_db)

    if level_mode == "peak":
        ref = np.max(np.abs(shaped))
    else:
        ref = np.sqrt(np.mean(shaped ** 2))
    if ref <= 0:
        return shaped

    out = shaped * (target / ref)
    if np.max(np.abs(out)) > 1.0:
        sys.exit(
            f"Error: --level-mode rms at {amplitude_db:g} dBFS RMS drives the peak to "
            f"{lin_to_db(np.max(np.abs(out))):.2f} dBFS, which clips. Lower --amplitude."
        )
    return out


def quantize(matrix: np.ndarray, bit_depth: int) -> tuple[bytes, np.ndarray]:
    """Interleaved float (n, ch) in [-1,1] -> (PCM bytes, int array actually written).
    24-bit is packed manually as 3-byte little-endian two's complement."""
    max_val = PCM_MAX[bit_depth]
    flat = matrix.reshape(-1)
    ints = np.clip(np.round(flat * max_val), -max_val - 1, max_val)

    if bit_depth == 16:
        arr = ints.astype("<i2")
        return arr.tobytes(), arr.astype(np.int64)

    arr = ints.astype("<i4")
    as_u8 = np.frombuffer(arr.tobytes(), dtype=np.uint8).reshape(-1, 4)
    return np.ascontiguousarray(as_u8[:, :3]).tobytes(), arr.astype(np.int64)


def write_wav(path: str, pcm: bytes, sr: int, channels: int, bit_depth: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2 if bit_depth == 16 else 3)
        wf.setframerate(sr)
        wf.writeframes(pcm)


def write_mp3(wav_path: str, mp3_path: str, bitrate: str) -> None:
    try:
        from pydub import AudioSegment
    except ImportError as e:
        sys.exit(
            "MP3 export needs 'pydub' plus ffmpeg on PATH.\n"
            "  pip install pydub\n"
            "  install ffmpeg via your package manager or https://ffmpeg.org\n"
            f"import error: {e}"
        )
    try:
        AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate=bitrate)
    except Exception as e:
        sys.exit(f"MP3 export failed (is ffmpeg installed and on PATH?): {e}")


def plot_diagnostics(mono: np.ndarray, sr: int, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"--plot needs matplotlib (pip install matplotlib): {e}", file=sys.stderr)
        return

    n = len(mono)
    spec = np.fft.rfft(mono * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mag = 20 * np.log10(np.maximum(np.abs(spec), 1e-12))
    mag -= mag.max()

    peak_idx = np.argmax(mag)
    peak_freq = freqs[peak_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(np.arange(n) / sr * 1000.0, mono, linewidth=0.8)
    ax1.set(title="Waveform / envelope", xlabel="Time (ms)", ylabel="Amplitude")
    ax2.plot(freqs, mag, linewidth=0.8)
    ax2.axvline(peak_freq, color="red", linewidth=0.8, linestyle="--")
    ax2.annotate(f"{peak_freq:.1f} Hz", xy=(peak_freq, 0), xytext=(5, -8),
                textcoords="offset points", color="red", fontsize=9)
    ax2.set(title="Magnitude spectrum", xlabel="Frequency (Hz)", ylabel="dB (rel. peak)")
    ax2.set_xlim(0, min(sr / 2, 6000))
    ax2.set_ylim(-100, 5)
    fig.suptitle(f"{title}")
    fig.tight_layout()
    plt.show()

def default_filename(a: argparse.Namespace, dur: float, r_in: float, r_out: float) -> str:
    ramp = (f"{fmt_num(r_in)}msramp" if r_in == r_out
            else f"{fmt_num(r_in)}in{fmt_num(r_out)}outmsramp")
    ext = "mp3" if a.format == "mp3" else "wav"
    freq_tag = "-".join(fmt_num(f) for f in a.freq)
    am_tag = f"_am{fmt_num(a.am_freq)}hz" if a.am_freq is not None else ""
    return f"tone_{freq_tag}hz_{fmt_num(dur)}ms_{ramp}{am_tag}.{ext}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a tone and save it as WAV or MP3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "--duration is the TOTAL tone length INCLUDING both ramps, not the sustain "
            "alone. --pad-pre/--pad-post add silence OUTSIDE that duration."
        ),
    )
    p.add_argument("--freq", type=float, nargs="+", default=[750.0],
                   help="Carrier frequency in Hz; give several to sum a multi-tone / "
                        "harmonic complex (e.g. --freq 440 554 659)")
    p.add_argument("--duration", type=float, default=75.0,
                   help="TOTAL tone length in ms, including ramps")
    p.add_argument("--ramp", type=float, default=10.0, help="Onset AND offset ramp in ms")
    p.add_argument("--ramp-in", type=float, default=None, help="Override onset ramp in ms")
    p.add_argument("--ramp-out", type=float, default=None, help="Override offset ramp in ms")
    p.add_argument("--waveform", choices=["sine", "square", "triangle", "noise"],
                   default="sine", help="Carrier shape (square/triangle are band-limited)")
    p.add_argument("--amplitude", type=float, default=-12.0,
                   help="Level in dBFS, negative; peak or RMS per --level-mode")
    p.add_argument("--level-mode", choices=["peak", "rms"], default="peak",
                   help="Interpret --amplitude as peak or RMS; use rms to loudness-match waveforms")
    p.add_argument("--am-freq", type=float, default=None,
                   help="Sinusoidal amplitude-modulation (SAM) rate in Hz; omit for no AM")
    p.add_argument("--am-depth", type=float, default=1.0,
                   help="AM modulation depth 0..1 (1 = full modulation)")
    p.add_argument("--sr", type=int, default=44100, help="Sample rate in Hz")
    p.add_argument("--bit-depth", type=int, choices=[16, 24], default=16, help="PCM bit depth")
    p.add_argument("--channels", type=int, choices=[1, 2], default=1,
                   help="Output channels (duplicated/diotic, not panned)")
    p.add_argument("--pad-pre", type=float, default=0.0, help="Leading silence in ms")
    p.add_argument("--pad-post", type=float, default=0.0, help="Trailing silence in ms")
    p.add_argument("--sync-channel", action="store_true",
                   help="Force stereo: L=stimulus, R=full-scale hard-gated sync burst for an "
                        "external timing-capture input. NEVER route R to a listener's ear")
    p.add_argument("--sync-level", type=float, default=0.0, help="Sync burst level in dBFS")
    p.add_argument("--format", choices=["wav", "mp3"], default="wav", help="Output format")
    p.add_argument("--mp3-bitrate", type=str, default="320k", help="MP3 bitrate if --format mp3")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for --waveform noise")
    p.add_argument("--calibration-tone", action="store_true",
                   help="Also emit a 10 s steady tone (same freq/level/waveform) alongside "
                        "the normal output, named <output>_calib, for SPL/level metering")
    p.add_argument("--metadata", action="store_true",
                   help="Also write a .json file with parameters and timing")
    p.add_argument("--plot", action="store_true", help="Show envelope + spectrum after saving")
    p.add_argument("--out", type=str, default=None,
                   help="Output path (default: auto-named from freq/duration/ramp)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing output file instead of refusing")
    return p.parse_args()


def main() -> None:
    a = parse_args()

    if a.amplitude > 0:
        sys.exit("Error: --amplitude must be <= 0 dBFS (0 dBFS is full scale).")
    if any(f <= 0 for f in a.freq):
        sys.exit("Error: every --freq must be positive.")
    for f in a.freq:
        if f >= a.sr / 2:
            sys.exit(f"Error: --freq {f:g} Hz is at or above Nyquist ({a.sr / 2:g} Hz).")
    if a.am_freq is not None:
        if a.am_freq <= 0:
            sys.exit("Error: --am-freq must be positive.")
        if not 0.0 <= a.am_depth <= 1.0:
            sys.exit("Error: --am-depth must be in [0, 1].")
    if min(a.ramp, a.ramp_in or 0, a.ramp_out or 0, a.pad_pre, a.pad_post) < 0:
        sys.exit("Error: ramp and pad values must be >= 0.")

    r_in = a.ramp_in if a.ramp_in is not None else a.ramp
    r_out = a.ramp_out if a.ramp_out is not None else a.ramp
    dur = 10000.0 if a.calibration_tone else a.duration

    if dur <= 0:
        sys.exit("Error: --duration must be positive.")
    if r_in + r_out > dur:
        sys.exit(
            f"Error: ramp-in + ramp-out ({r_in:g} + {r_out:g} = {r_in + r_out:g} ms) exceeds "
            f"total duration ({dur:g} ms). Shorten the ramps or lengthen the duration."
        )

    seed = int(np.random.SeedSequence().entropy % (2 ** 32)) if a.seed is None else a.seed
    rng = np.random.default_rng(seed)

    ms = lambda x: int(round(x * a.sr / 1000.0))
    n_total, n_in, n_out = ms(dur), ms(r_in), ms(r_out)
    n_pre, n_post = ms(a.pad_pre), ms(a.pad_post)
    if n_total < 1:
        sys.exit(f"Error: --duration {dur:g} ms is under one sample at {a.sr} Hz.")

    tone = build_tone(a.freq, n_total, n_in, n_out, a.sr, a.waveform,
                      a.amplitude, a.level_mode, rng, a.am_freq, a.am_depth)
    mono = np.concatenate([np.zeros(n_pre), tone, np.zeros(n_post)])

    channels = 2 if a.sync_channel else a.channels
    if a.sync_channel:
        sync = np.concatenate([
            np.zeros(n_pre),
            generate_carrier([a.freq[0]], n_total, a.sr, "sine", rng) * db_to_lin(a.sync_level),
            np.zeros(n_post),
        ])
        matrix = np.column_stack([mono, sync])
    elif channels == 2:
        matrix = np.column_stack([mono, mono])
    else:
        matrix = mono.reshape(-1, 1)

    if a.waveform != "sine":
        print(
            f"Warning: --waveform {a.waveform} carries far more harmonic energy than a sine "
            "at the same peak level. Consider --level-mode rms to loudness-match it.",
            file=sys.stderr,
        )
    if a.format == "mp3":
        print(
            "Warning: MP3 is lossy. Encoder-dependent leading padding (~576-1152 samples at "
            "44.1 kHz) shifts onset relative to any external trigger, and the codec smears "
            "transients. Avoid for timing-critical use -- prefer WAV.",
            file=sys.stderr,
        )
    if a.pad_pre > 0:
        print(
            f"Warning: --pad-pre {a.pad_pre:g} ms delays tone onset relative to the START of "
            "the file. Account for this offset if you time-lock to playback start.",
            file=sys.stderr,
        )
    if a.sync_channel:
        print(
            "Warning: --sync-channel writes a full-scale burst on the RIGHT channel. Route it "
            "to your capture device ONLY. It is not a listening signal.",
            file=sys.stderr,
        )

    out_path = a.out or default_filename(a, dur, r_in, r_out)
    if os.path.exists(out_path) and not a.force:
        sys.exit(
            f"Error: {out_path} already exists. Refusing to overwrite a stimulus file.\n"
            "Pass --force to overwrite, or --out to write elsewhere."
        )
    pcm, ints = quantize(matrix, a.bit_depth)

    if a.format == "mp3":
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            write_wav(tmp, pcm, a.sr, channels, a.bit_depth)
            write_mp3(tmp, out_path, a.mp3_bitrate)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        write_wav(out_path, pcm, a.sr, channels, a.bit_depth)

    max_val = PCM_MAX[a.bit_depth]
    stim_ints = ints.reshape(-1, channels)[n_pre:n_pre + n_total, 0] / max_val
    peak_db = lin_to_db(np.max(np.abs(stim_ints)))
    rms_db = lin_to_db(np.sqrt(np.mean(stim_ints ** 2)))
    actual_dur_ms = n_total / a.sr * 1000.0
    half_amp_ms = r_in / 2.0

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": out_path,
        "parameters": {
            "freq_hz": a.freq,
            "am_freq_hz": a.am_freq,
            "am_depth": a.am_depth if a.am_freq is not None else None,
            "duration_ms_requested": dur,
            "duration_ms_actual": actual_dur_ms,
            "duration_samples": n_total,
            "ramp_in_ms": r_in,
            "ramp_out_ms": r_out,
            "ramp_shape": "raised_cosine_hann",
            "waveform": a.waveform,
            "band_limited": a.waveform in ("square", "triangle"),
            "level_mode": a.level_mode,
            "amplitude_dBFS_requested": a.amplitude,
            "amplitude_dBFS_peak_written": peak_db,
            "amplitude_dBFS_rms_written": rms_db,
            "sample_rate_hz": a.sr,
            "bit_depth": a.bit_depth,
            "channels": channels,
            "sync_channel": a.sync_channel,
            "sync_level_dBFS": a.sync_level if a.sync_channel else None,
            "pad_pre_ms": a.pad_pre,
            "pad_post_ms": a.pad_post,
            "format": a.format,
            "mp3_bitrate": a.mp3_bitrate if a.format == "mp3" else None,
            "seed": seed,
            "calibration_tone": a.calibration_tone,
        },
        "timing": {
            "nominal_onset_ms_from_file_start": a.pad_pre,
            "half_amplitude_onset_ms_from_nominal": half_amp_ms,
            "note": "Hann ramp reaches half amplitude (-6 dB) at ramp_in/2.",
        },
        "note": "dB SPL/loudness is not set by this script -- it depends on playback "
                "hardware and gain, and must be measured/calibrated externally.",
    }
    if a.metadata:
        sidecar = out_path.split(".")[0] + ".json"
        with open(sidecar, "w") as f:
            json.dump(meta, f, indent=2)

    print("=" * 72)
    print("Tone generated")
    print(f"  output              : {out_path}")
    freq_str = ", ".join(f"{f:g}" for f in a.freq)
    print(f"  frequency           : {freq_str} Hz"
          + (f" ({len(a.freq)}-tone complex)" if len(a.freq) > 1 else ""))
    if a.am_freq is not None:
        print(f"  amplitude mod       : {a.am_freq:g} Hz SAM, depth {a.am_depth:g}")
    print(f"  waveform            : {a.waveform}"
          + (" (band-limited)" if a.waveform in ("square", "triangle") else ""))
    print(f"  duration            : {actual_dur_ms:.4f} ms actual / {dur:g} ms requested "
          f"({n_total} samples, ramps included)")
    print(f"  ramp in / out       : {r_in:g} / {r_out:g} ms raised cosine (Hann)")
    print(f"  pad pre / post      : {a.pad_pre:g} / {a.pad_post:g} ms")
    print(f"  level mode          : {a.level_mode}")
    print(f"  requested level     : {a.amplitude:g} dBFS ({a.level_mode})")
    print(f"  written peak        : {peak_db:.2f} dBFS")
    print(f"  written RMS         : {rms_db:.2f} dBFS")
    print(f"  sample rate         : {a.sr} Hz")
    print(f"  bit depth           : {a.bit_depth}-bit PCM")
    print(f"  channels            : {channels}"
          + (" (L=stimulus, R=sync burst)" if a.sync_channel else ""))
    print(f"  format              : {a.format}")
    if a.waveform == "noise":
        print(f"  noise seed          : {seed}")
    print(f"  onset               : nominal at {a.pad_pre:g} ms from file start; "
          f"envelope hits half amplitude (-6 dB) {half_amp_ms:g} ms later")
    print("  NOTE: dB SPL/loudness is not set here -- calibrate in hardware")
    print("        (--calibration-tone + an SPL meter or loudness meter).")
    print("=" * 72)

    if a.calibration_tone:
        calib_channels = a.channels
        n_calib = ms(10000.0)
        calib_tone = build_tone(a.freq, n_calib, n_in, n_out, a.sr, a.waveform,
                                a.amplitude, a.level_mode, rng, a.am_freq, a.am_depth)
        calib_matrix = (np.column_stack([calib_tone, calib_tone]) if calib_channels == 2
                        else calib_tone.reshape(-1, 1))

        stem, ext = os.path.splitext(out_path)
        calib_path = f"{stem}_calib{ext}"
        if os.path.exists(calib_path) and not a.force:
            sys.exit(f"Error: {calib_path} already exists. Pass --force to overwrite.")

        calib_pcm, _ = quantize(calib_matrix, a.bit_depth)
        if a.format == "mp3":
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                write_wav(tmp, calib_pcm, a.sr, calib_channels, a.bit_depth)
                write_mp3(tmp, calib_path, a.mp3_bitrate)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        else:
            write_wav(calib_path, calib_pcm, a.sr, calib_channels, a.bit_depth)

        print(f"Calibration tone (10 s, same freq/level/waveform): {calib_path}")

    if a.plot:
        plot_diagnostics(mono, a.sr, out_path)


if __name__ == "__main__":
    main()