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

import numpy as np
import scipy.io.wavfile as wavfile
import random
import math

# ==================== CONFIGURATION ====================
# Seed for reproducibility.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Sample rate in Hz.
SAMPLE_RATE = 48000

# Number of sweeps to generate.
NUM_SWEEPS = 40  # Increased number of sweeps due to shorter sweep duration.

# Duration of each sweep in seconds (shorter than before).
SWEEP_DURATION = 5.0

# Total duration of the output file in seconds.
TOTAL_DURATION = 60.0

# Minimum and maximum frequencies (in Hz) for the sweeps.
MIN_FREQ = 20       # Include low-frequency content.
MAX_FREQ = 24000    # Up to 24 kHz.

# Amplitude range for each sweep (scaling factor).
MIN_AMP = 0.1
MAX_AMP = 1.0

# Output filename.
OUTPUT_FILENAME = 'random_sine_sweeps_float.wav'
# ========================================================

# Total number of samples for the final output.
total_samples = int(TOTAL_DURATION * SAMPLE_RATE)
output_signal = np.zeros(total_samples)

# Generate each sweep and add it into the overall output at a random start time.
for i in range(NUM_SWEEPS):
    # Choose a random start time such that the sweep fits entirely within the output.
    max_start_time = TOTAL_DURATION - SWEEP_DURATION
    start_time = random.uniform(0, max_start_time)
    start_sample = int(start_time * SAMPLE_RATE)

    # Time vector for the sweep.
    t = np.linspace(0, SWEEP_DURATION, int(SWEEP_DURATION * SAMPLE_RATE), endpoint=False)

    # Randomize sweep parameters:
    # Frequencies are chosen uniformly in logarithmic space for a balanced representation
    # across octaves (more low-frequency resolution).
    f_start = math.exp(random.uniform(math.log(MIN_FREQ), math.log(MAX_FREQ)))
    f_end   = math.exp(random.uniform(math.log(MIN_FREQ), math.log(MAX_FREQ)))

    # Randomize amplitude for this sweep.
    amplitude = random.uniform(MIN_AMP, MAX_AMP)

    # Randomly choose sweep type: True for logarithmic, False for linear.
    is_logarithmic = random.choice([True, False])

    # --- Generate the sweep ---
    if not is_logarithmic:
        # Linear sweep:
        #   Instantaneous frequency: f(t) = f_start + (f_end - f_start) * t / SWEEP_DURATION
        #   Phase: φ(t) = 2π [ f_start*t + (f_end-f_start)*t²/(2*SWEEP_DURATION) ]
        phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * (t**2) / SWEEP_DURATION)
        sweep = np.sin(phase)
    else:
        # Logarithmic sweep:
        #   Instantaneous frequency: f(t) = f_start * (f_end / f_start)^(t/SWEEP_DURATION)
        #   Phase: φ(t) = 2π * f_start * SWEEP_DURATION / ln(f_end/f_start) * ( (f_end/f_start)^(t/SWEEP_DURATION) - 1 )
        # If f_start and f_end are nearly equal, revert to a constant frequency.
        if abs(f_end - f_start) < 1e-6:
            phase = 2 * np.pi * f_start * t
            sweep = np.sin(phase)
        else:
            K = np.log(f_end / f_start)
            phase = 2 * np.pi * f_start * SWEEP_DURATION / K * (np.exp((t / SWEEP_DURATION) * K) - 1)
            sweep = np.sin(phase)

    # Apply randomized amplitude.
    sweep *= amplitude

    # Add the generated sweep to the output signal at the chosen start time.
    end_sample = start_sample + len(sweep)
    if end_sample > total_samples:
        # Truncate the sweep if it overruns the total duration.
        sweep = sweep[:total_samples - start_sample]
        end_sample = total_samples

    output_signal[start_sample:end_sample] += sweep

    # Optionally, print out the parameters for each sweep.
    sweep_type = 'Logarithmic' if is_logarithmic else 'Linear'
    print(f"Sweep {i+1}: {sweep_type} sweep from {f_start:.2f} Hz to {f_end:.2f} Hz, amplitude {amplitude:.2f}, starting at {start_time:.2f} s.")

# Normalize the final signal to prevent clipping (values will be in [-1, 1]).
max_val = np.max(np.abs(output_signal))
if max_val > 0:
    output_signal /= max_val
    output_signal *= 0.9  # Scale to 90% of the full range.

# Convert the signal to 32-bit float for float based WAV output.
output_signal = output_signal.astype(np.float32)

# Save the generated signal to disk as a WAV file with float format.
wavfile.write(OUTPUT_FILENAME, SAMPLE_RATE, output_signal)
print(f"\nSaved generated sweeps to '{OUTPUT_FILENAME}'.")
