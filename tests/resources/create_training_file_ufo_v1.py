#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 Mikko Honkala
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import scipy.io.wavfile as wavfile
import random
import math
import warnings
import soundfile as sf  # For writing 24-bit PCM output

# Optionally suppress non-critical warnings from scipy.wavfile.
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

# ==================== CONFIGURATION ====================
SEED = 142
random.seed(SEED)
np.random.seed(SEED)

SAMPLE_RATE = 48000            # Hz
FINAL_DURATION = 404.0         # seconds (6:44 total)
FIRST_PART_DURATION = 18.0     # seconds: portion from input.wav to preserve at the beginning
NEW_SECTION_DURATION = 44.0    # seconds for the new long sweeps section (C)
NEW_SECTION_D_DURATION = 30.0  # seconds for the new section D (10 linear sweeps, each 3 sec)

# Available time for the random sweeps block (S) + tail (T):
available_S_T = FINAL_DURATION - FIRST_PART_DURATION - NEW_SECTION_DURATION  # 404 - 18 - 44 = 342 sec

# For long input files we want a default random sweeps block duration:
DEFAULT_SWEEPS_DURATION = 240.0   # seconds, if possible

# (The old approach used a density based on 160 sweeps over 240 s.)
# -- The new S block will be built from contiguous 2-second segments.
# Other parameters for generating individual sweeps (within each segment):
MIN_FREQ = 20                # Hz (for random sweeps)
MAX_FREQ = 24000             # Hz
MIN_AMP = 0.1
MAX_AMP = 1.0
MIX_PROB = 0.5               # probability to choose a linear sweep instead of logarithmic

# For the new long sweeps section (C), we sweep from near 0 to NEW_SWEEP_END:
NEW_SWEEP_END = 23900.0

# New scaling factor to lower the maximum amplitude of section C to 95%
NEW_SWEEPS_SCALE = 0.95

INPUT_FILENAME = 'input.wav'
FINAL_OUTPUT_FILENAME = 'input_sweeps.wav'
# ========================================================

# ----------------- Read and Process input.wav -----------------
print("Reading input file:", INPUT_FILENAME)
input_rate, input_data = wavfile.read(INPUT_FILENAME)
if input_rate != SAMPLE_RATE:
    raise ValueError(f"Input sample rate ({input_rate}) does not match expected {SAMPLE_RATE} Hz.")

# Convert input_data to float32 in the range [-1, 1]:
# For 16-bit PCM: divide by 32768.0.
# For 24-bit PCM (stored as int32), we assume the lower 8 bits are unused.
if input_data.dtype == np.int16:
    input_data = input_data.astype(np.float32) / 32768.0
elif input_data.dtype == np.int32:
    # Right-shift by 8 bits to extract the original 24-bit data,
    # then divide by 2^23 = 8388608 to scale to [-1, 1].
    input_data = (input_data >> 8).astype(np.float32) / 8388608.0
elif input_data.dtype == np.float32:
    pass
else:
    input_data = input_data.astype(np.float32)

# Determine number of channels and duration.
if input_data.ndim == 1:
    channels = 1
else:
    channels = input_data.shape[1]
input_samples = input_data.shape[0]
input_duration = input_samples / SAMPLE_RATE
print(f"Input duration: {input_duration:.2f} s, Channels: {channels}")

if input_duration < FIRST_PART_DURATION:
    raise ValueError(f"Input file must be at least {FIRST_PART_DURATION} seconds long.")

# ----------------- Decide Durations for Sections -----------------
# We want the final file to be: A + S + C + D + T = 404 s, where:
#   A = FIRST_PART_DURATION (18 s)
#   C = new long sweeps section = NEW_SECTION_DURATION (44 s)
#   D = new section of 10 linear sweeps = NEW_SECTION_D_DURATION (30 s)
#   S + T = available_S_T = 342 s  (from the input tail and random sweeps)
#
# For shorter input files, let T = min(input_duration - FIRST_PART_DURATION, 342),
# and then S_duration_old = available_S_T - T_duration.
# Finally, we subtract 30 sec from S (to make room for section D).
if input_duration < FINAL_DURATION:
    T_duration = min(input_duration - FIRST_PART_DURATION, available_S_T)
    S_duration_old = available_S_T - T_duration
else:
    S_duration_old = DEFAULT_SWEEPS_DURATION
    T_duration = available_S_T - S_duration_old

# Now subtract 30 seconds from the random sweeps section for the new section D.
S_duration = S_duration_old - NEW_SECTION_D_DURATION

print("Section durations:")
print(f" - A (first part from input): {FIRST_PART_DURATION} s")
print(f" - S (random sweeps block, reduced): {S_duration:.2f} s")
print(f" - C (new long sweeps): {NEW_SECTION_DURATION} s")
print(f" - D (10 linear sweeps): {NEW_SECTION_D_DURATION} s")
print(f" - T (tail from input): {T_duration:.2f} s")
# Total = 18 + S_duration + 44 + 30 + T_duration = 404 s

# ----------------- New Random Sweeps Block (S) – New Version -----------------
# In this version, S is built from contiguous segments of fixed duration (2 seconds)
# (with one possible segment for any leftover time).
# In each segment, we randomly choose between 1 and 3 overlapping sweeps.
def sample_frequency(min_freq, max_freq, mix_prob):
    """Sample a frequency using a mix of linear and logarithmic scaling."""
    if random.random() < mix_prob:
        return random.uniform(min_freq, max_freq)
    else:
        return math.exp(random.uniform(math.log(min_freq), math.log(max_freq)))

def generate_sweep_segment(duration):
    """Generate a segment of length 'duration' seconds composed of 1-3 overlapping sweeps."""
    seg_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, seg_samples, endpoint=False)
    # Choose a random number (1-3) of sweeps to overlay in this segment.
    n_sweeps = random.randint(1, 3)
    segment = np.zeros(seg_samples, dtype=np.float32)
    for _ in range(n_sweeps):
        f_start = sample_frequency(MIN_FREQ, MAX_FREQ, MIX_PROB)
        f_end = sample_frequency(MIN_FREQ, MAX_FREQ, MIX_PROB)
        amplitude = random.uniform(MIN_AMP, MAX_AMP)
        is_log = random.choice([True, False])
        if not is_log:
            # Linear sweep over the full segment duration.
            phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * (t**2) / duration)
            sweep = np.sin(phase)
        else:
            # Logarithmic sweep.
            if abs(f_end - f_start) < 1e-6:
                phase = 2 * np.pi * f_start * t
                sweep = np.sin(phase)
            else:
                K = math.log(f_end / f_start)
                phase = 2 * np.pi * f_start * duration / K * (np.exp((t / duration) * K) - 1)
                sweep = np.sin(phase)
        segment += amplitude * sweep
    return segment

# Build S_signal from contiguous 2-second segments.
segments = []
full_seg_duration = 2.0
n_full_segments = int(S_duration // full_seg_duration)
leftover = S_duration - (n_full_segments * full_seg_duration)
for i in range(n_full_segments):
    seg = generate_sweep_segment(full_seg_duration)
    segments.append(seg)
    print(f"Generated S segment {i+1} of {full_seg_duration:.1f} s with {len(seg)} samples.")
if leftover > 0:
    seg = generate_sweep_segment(leftover)
    segments.append(seg)
    print(f"Generated leftover S segment of {leftover:.2f} s with {len(seg)} samples.")

S_signal = np.concatenate(segments, axis=0)
print(f"Total S section duration: {S_signal.shape[0] / SAMPLE_RATE:.2f} s.")

# Normalize the entire S_signal to 90% of full scale.
max_val = np.max(np.abs(S_signal))
if max_val > 0:
    S_signal = (S_signal / max_val) * 0.9

# If the input is multi-channel but S_signal is mono, duplicate the channel.
if channels > 1 and S_signal.ndim == 1:
    S_signal = np.tile(S_signal[:, np.newaxis], (1, channels))

# ----------------- Generate the New Long Sweeps Section (C) -----------------
# Section C lasts NEW_SECTION_DURATION seconds (44 s) and is built from 4 concatenated sweeps:
#   C1: 8 sec linear sweep from 20 Hz to NEW_SWEEP_END.
#   C2: 8 sec logarithmic sweep from 1 Hz to NEW_SWEEP_END.
#   C3: 14 sec linear sweep from 20 Hz to NEW_SWEEP_END with amplitude modulation.
#   C4: 14 sec logarithmic sweep from 1 Hz to NEW_SWEEP_END with amplitude modulation.
def generate_linear_sweep(duration, f_start, f_end, envelope=None):
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * (t**2) / duration)
    sweep = np.sin(phase)
    if envelope is not None:
        sweep *= envelope(t)
    return sweep

def generate_log_sweep(duration, f_start, f_end, envelope=None):
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    if f_start <= 0:
        f_start = 1  # Avoid log(0)
    if abs(f_end - f_start) < 1e-6:
        phase = 2 * np.pi * f_start * t
    else:
        K = math.log(f_end / f_start)
        phase = 2 * np.pi * f_start * duration / K * (np.exp((t / duration) * K) - 1)
    sweep = np.sin(phase)
    if envelope is not None:
        sweep *= envelope(t)
    return sweep

# New envelope: period 0.25 sec and amplitude ranges from 2% to 98%.
def sine_envelope(t):
    return 0.48 * np.sin(2 * np.pi * t / 0.25) + 0.5

# Generate the four sweeps and scale them via NEW_SWEEPS_SCALE:
# Note: The linear sweeps now start at 20 Hz.
C1 = NEW_SWEEPS_SCALE * generate_linear_sweep(8.0, 20.0, NEW_SWEEP_END)
C2 = NEW_SWEEPS_SCALE * generate_log_sweep(8.0, 1.0, NEW_SWEEP_END)
C3 = NEW_SWEEPS_SCALE * generate_linear_sweep(14.0, 20.0, NEW_SWEEP_END, envelope=sine_envelope)
C4 = NEW_SWEEPS_SCALE * generate_log_sweep(14.0, 1.0, NEW_SWEEP_END, envelope=sine_envelope)

# Concatenate the four sweeps to form section C.
C_signal = np.concatenate((C1, C2, C3, C4))
print(f"Generated new long sweeps section (C): {C_signal.shape[0] / SAMPLE_RATE:.2f} s total.")

# If multi-channel, duplicate C_signal.
if channels > 1 and C_signal.ndim == 1:
    C_signal = np.tile(C_signal[:, np.newaxis], (1, channels))

# ----------------- Generate the New Section D -----------------
# This section consists of 10 linear sweeps from 1 kHz to 23.0 kHz,
# each 3 seconds long, with decreasing volume.
# The first sweep is at 99% amplitude and the last at about 10%.
def generate_linear_sweep_custom(duration, f_start, f_end):
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * (t**2) / duration)
    return np.sin(phase)

D_segments = []
num_sweeps_D = 10
for i in range(num_sweeps_D):
    # Linear interpolation of volume: from 0.99 (first) down to ~0.10 (last)
    vol = 0.99 - (0.89 * (i / (num_sweeps_D - 1)))
    sweep = generate_linear_sweep_custom(3.0, 1000.0, 23000.0) * vol
    D_segments.append(sweep)
D_signal = np.concatenate(D_segments, axis=0)
print(f"Generated new section D: {D_signal.shape[0] / SAMPLE_RATE:.2f} s total.")

# If multi-channel, duplicate D_signal.
if channels > 1 and D_signal.ndim == 1:
    D_signal = np.tile(D_signal[:, np.newaxis], (1, channels))

# ----------------- Extract Sections from input.wav -----------------
# Section A: The first FIRST_PART_DURATION seconds from the input.
first_part_samples = int(FIRST_PART_DURATION * SAMPLE_RATE)
A_signal = input_data[:first_part_samples]

# Section T: Tail from input, starting at FIRST_PART_DURATION and lasting T_duration.
tail_samples = int(T_duration * SAMPLE_RATE)
T_signal = input_data[first_part_samples:first_part_samples + tail_samples]

# Ensure A and T have the correct number of channels.
def ensure_channels(signal, target_channels):
    if signal.ndim == 1 and target_channels > 1:
        return np.tile(signal[:, np.newaxis], (1, target_channels))
    elif signal.ndim > 1 and signal.shape[1] != target_channels:
        return signal[:, :target_channels]
    return signal

A_signal = ensure_channels(A_signal, channels)
T_signal = ensure_channels(T_signal, channels)

# ----------------- Concatenate All Sections -----------------
# Final structure: A + S + C + D + T
final_signal = np.concatenate((A_signal, S_signal, C_signal, D_signal, T_signal), axis=0)

# Force final_signal to be exactly FINAL_DURATION seconds.
final_total_samples = int(FINAL_DURATION * SAMPLE_RATE)
current_samples = final_signal.shape[0]
if current_samples > final_total_samples:
    final_signal = final_signal[:final_total_samples]
elif current_samples < final_total_samples:
    pad_samples = final_total_samples - current_samples
    if channels > 1:
        pad = np.zeros((pad_samples, channels), dtype=np.float32)
    else:
        pad = np.zeros(pad_samples, dtype=np.float32)
    final_signal = np.concatenate((final_signal, pad), axis=0)

print(f"Final signal: {final_signal.shape[0]} samples, {final_signal.shape[0] / SAMPLE_RATE:.2f} s total.")

# ----------------- Save Final Output as 24-bit PCM -----------------
print(f"Saving final output as 24-bit PCM to '{FINAL_OUTPUT_FILENAME}'...")
sf.write(FINAL_OUTPUT_FILENAME, final_signal, SAMPLE_RATE, subtype='PCM_24')
print(f"Saved final output to '{FINAL_OUTPUT_FILENAME}'.")
