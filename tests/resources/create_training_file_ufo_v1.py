#!/usr/bin/env python3
"""
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""

import numpy as np
import scipy.io.wavfile as wavfile
import random
import math
import warnings
import soundfile as sf  # For writing 24-bit PCM output

# Optionally suppress non‐critical warnings from scipy.wavfile.
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

# ==================== CONFIGURATION ====================
SEED = 142
random.seed(SEED)
np.random.seed(SEED)

SAMPLE_RATE = 48000           # Hz
FINAL_DURATION = 404.0        # seconds (6:44)
FIRST_PART_DURATION = 18.0    # seconds: from input.wav to preserve at the beginning

# Default sweeps parameters (if input.wav is long enough)
DEFAULT_SWEEPS_DURATION = 240.0  # seconds (if possible)

# Original sweeps density (used to decide how many sweeps to generate)
ORIGINAL_SWEEPS_NUM = 160          # originally 160 sweeps over 240 s
ORIGINAL_SWEEPS_TOTAL = 240.0      # seconds
SWEEPS_DENSITY = ORIGINAL_SWEEPS_NUM / ORIGINAL_SWEEPS_TOTAL  # sweeps per second

SWEEP_DURATION = 5.0        # each sweep lasts 5 seconds
MIN_FREQ = 20               # Hz
MAX_FREQ = 24000            # Hz
MIN_AMP = 0.1
MAX_AMP = 1.0
MIX_PROB = 0.5              # probability for linear frequency selection

INPUT_FILENAME = 'input.wav'
FINAL_OUTPUT_FILENAME = 'input_sweeps.wav'
# ========================================================

# ----------------- Read and Process input.wav -----------------
print("Reading input file:", INPUT_FILENAME)
input_rate, input_data = wavfile.read(INPUT_FILENAME)
if input_rate != SAMPLE_RATE:
    raise ValueError(f"Input sample rate ({input_rate}) does not match expected {SAMPLE_RATE} Hz.")

# Convert input_data to float32 in the range [-1, 1]:
# For 16-bit: divide by 32768.0.
# For 24-bit (stored as int32), we assume the lower 8 bits are not used.
if input_data.dtype == np.int16:
    input_data = input_data.astype(np.float32) / 32768.0
elif input_data.dtype == np.int32:
    # Assuming 24-bit PCM stored in 32-bit integers.
    # Right-shift by 8 bits to extract the original 24-bit data, then scale.
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

# ----------------- Determine Durations for Sweeps and Tail -----------------
# Final output will be: [First part (A)] + [Sweeps block] + [Tail (B)] = 404 s.
if input_duration < FINAL_DURATION:
    # If input is short, use all available tail and fill missing time with sweeps.
    tail_duration = input_duration - FIRST_PART_DURATION
    sweeps_duration = FINAL_DURATION - input_duration
else:
    # Input is long: try to use the default sweeps duration.
    desired_tail_duration = FINAL_DURATION - FIRST_PART_DURATION - DEFAULT_SWEEPS_DURATION
    available_tail = input_duration - FIRST_PART_DURATION
    if available_tail < desired_tail_duration:
        # Not enough tail available: use full tail and adjust sweeps duration.
        tail_duration = available_tail
        sweeps_duration = FINAL_DURATION - FIRST_PART_DURATION - tail_duration
    else:
        tail_duration = desired_tail_duration
        sweeps_duration = DEFAULT_SWEEPS_DURATION

print(f"Final output: {FINAL_DURATION} s")
print(f" - First part (from input): {FIRST_PART_DURATION} s")
print(f" - Sweeps block: {sweeps_duration:.2f} s")
print(f" - Tail (from input): {tail_duration:.2f} s")

# ----------------- Generate the Sweeps Block -----------------
def sample_frequency(min_freq, max_freq, mix_prob):
    """Sample a frequency using a mix of linear and logarithmic scaling."""
    if random.random() < mix_prob:
        return random.uniform(min_freq, max_freq)
    else:
        return math.exp(random.uniform(math.log(min_freq), math.log(max_freq)))

sweeps_total_samples = int(sweeps_duration * SAMPLE_RATE)
sweeps_signal = np.zeros(sweeps_total_samples, dtype=np.float32)

# Decide number of sweeps based on density.
NUM_SWEEPS = int(sweeps_duration * SWEEPS_DENSITY)
print(f"Generating {NUM_SWEEPS} sweeps...")

for i in range(NUM_SWEEPS):
    # Choose a random start time (ensure the sweep fits in the block)
    max_start_time = sweeps_duration - SWEEP_DURATION
    if max_start_time <= 0:
        break
    start_time = random.uniform(0, max_start_time)
    start_sample = int(start_time * SAMPLE_RATE)

    # Create time vector for this sweep.
    t = np.linspace(0, SWEEP_DURATION, int(SWEEP_DURATION * SAMPLE_RATE), endpoint=False)

    # Randomly choose start and end frequencies.
    f_start = sample_frequency(MIN_FREQ, MAX_FREQ, MIX_PROB)
    f_end = sample_frequency(MIN_FREQ, MAX_FREQ, MIX_PROB)
    amplitude = random.uniform(MIN_AMP, MAX_AMP)
    is_logarithmic = random.choice([True, False])

    if not is_logarithmic:
        phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * (t**2) / SWEEP_DURATION)
        sweep = np.sin(phase)
    else:
        if abs(f_end - f_start) < 1e-6:
            phase = 2 * np.pi * f_start * t
            sweep = np.sin(phase)
        else:
            K = math.log(f_end / f_start)
            phase = 2 * np.pi * f_start * SWEEP_DURATION / K * (np.exp((t / SWEEP_DURATION) * K) - 1)
            sweep = np.sin(phase)
    sweep *= amplitude

    # Add the sweep into the sweeps block.
    end_sample = start_sample + len(sweep)
    if end_sample > sweeps_total_samples:
        sweep = sweep[:sweeps_total_samples - start_sample]
        end_sample = sweeps_total_samples
    sweeps_signal[start_sample:end_sample] += sweep

    print(f"Sweep {i+1}: {'Logarithmic' if is_logarithmic else 'Linear'} from {f_start:.2f} Hz to {f_end:.2f} Hz, "
          f"amp {amplitude:.2f}, starting at {start_time:.2f} s.")

# Normalize the sweeps block independently (to 90% of full scale).
max_val = np.max(np.abs(sweeps_signal))
if max_val > 0:
    sweeps_signal = (sweeps_signal / max_val) * 0.9

# If input is multi‐channel but sweeps_signal is mono, duplicate sweeps across channels.
if channels > 1 and sweeps_signal.ndim == 1:
    sweeps_signal = np.tile(sweeps_signal[:, np.newaxis], (1, channels))

# ----------------- Extract Parts from input.wav -----------------
# Part A: First 18 seconds.
first_part_samples = int(FIRST_PART_DURATION * SAMPLE_RATE)
A = input_data[:first_part_samples]

# Part B: The tail from input, starting at 18 s and lasting tail_duration.
tail_samples = int(tail_duration * SAMPLE_RATE)
B = input_data[first_part_samples:first_part_samples + tail_samples]

# Helper to ensure a signal has the correct number of channels.
def ensure_channels(signal, target_channels):
    if signal.ndim == 1 and target_channels > 1:
        return np.tile(signal[:, np.newaxis], (1, target_channels))
    elif signal.ndim > 1 and signal.shape[1] != target_channels:
        return signal[:, :target_channels]
    return signal

A = ensure_channels(A, channels)
B = ensure_channels(B, channels)

# ----------------- Concatenate the Final Signal -----------------
# Final signal = [First part] + [Sweeps block] + [Tail]
final_signal = np.concatenate((A, sweeps_signal, B), axis=0)

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
# Using SoundFile to write a proper 24-bit PCM WAV.
print(f"Saving final output as 24-bit PCM to '{FINAL_OUTPUT_FILENAME}'...")
sf.write(FINAL_OUTPUT_FILENAME, final_signal, SAMPLE_RATE, subtype='PCM_24')
print(f"Saved final output to '{FINAL_OUTPUT_FILENAME}'.")
