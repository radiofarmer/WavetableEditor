from WavetableEditor.Wavetable import *
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt


def play(samples, samprate=48000, s_format=pyaudio.paInt16):
    if s_format == pyaudio.paInt16 and samples.dtype != np.int16:
        samples = (2 ** 10 * samples).astype(np.int16)

    pa = pyaudio.PyAudio()
    stream = pa.open(format=s_format, channels=1, rate=samprate, output=True)
    stream.write(samples)
    stream.stop_stream()
    stream.close()


def formant_shift_test(wave: Waveform, shift_func, samples_per_cycle=440, cycles=1):
    x = wave.generate_series(samples_per_cycle, cycles=cycles)
    play(np.tile(x, 100 // cycles))
    x_fft = np.abs(fft.fft(x)[:int(samples_per_cycle * cycles) // 2])

    phase_new = np.array([shift_func((i % samples_per_cycle) / samples_per_cycle, 2.5)
                          for i in range(x.shape[0])])
    x_shifted = np.array([x[int(i * samples_per_cycle)] for i in phase_new])
    play(np.tile(x_shifted, 100 // cycles))
    x_shifted_fft = np.abs(fft.fft(x_shifted)[:int(samples_per_cycle * cycles) // 2])

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(x_fft)
    ax[1, 0].plot(x_shifted_fft)
    ax[0, 1].plot(x)
    ax[1, 1].plot(x_shifted)
    plt.show()


if __name__ == "__main__":
    shift = lambda p, f = 2: (f * p) if p < 1/f else 0.
    plt.show()
    cycles = 2
    saw_series = HarmonicSeries(0, 200, 1, saw_coeffs)
    wave = Waveform()
    wave.append_series(saw_series)
    formant_shift_test(wave, shift, samples_per_cycle=400, cycles=2)
