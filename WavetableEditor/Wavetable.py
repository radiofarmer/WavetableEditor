import numpy as np
from WavetableEditor import IO
from scipy import fft
from scipy import interpolate
import matplotlib.pyplot as plt
import math


def quantize_to_fraction(x, f):
    return np.floor(f * x) / f


def make_shift_function(x, shift_step, max_shift=1., noise=0.):
    func = interpolate.interp1d(np.linspace(0, 1, x.shape[0]), x / np.max(x))
    return lambda h: h + max_shift * quantize_to_fraction(
        func(np.linspace(0, 1., h.shape[0])) + np.random.random(h.shape[0]) * noise,
        shift_step
    )


def window(k, length):
    t = np.linspace(0, 1, length)
    fact = -np.abs((k - np.floor(k)) - 0.5) + 0.5
    return (-np.cos(2 * np.pi * t) / 2 + 0.5) ** fact


def oscillating_signs(harmonics):
    signs = np.ones(harmonics.shape[0])
    for i, h in enumerate(harmonics):
        if h <= 1.:
            continue
        else:
            num = h / (h - np.floor(h)) if h != int(h) else h
            signs[i] *= 1. if num % 2 else -1.
    return signs


def square_coeffs(harmonics):
    coeffs = np.zeros(harmonics.shape[0])
    for i, h in enumerate(harmonics):
        if h == 0:
            coeffs[i] = 0
        elif h % 2 == 0:
            coeffs[i] = 0
        else:
            coeffs[i] = 1 / h
    return coeffs


def saw_coeffs(harmonics):
    coeffs = np.zeros(harmonics.shape[0])
    for i, h in enumerate(harmonics):
        if h == 0:
            coeffs[i] = 0
        else:
            coeffs[i] = 1 / h
    return coeffs


def zero_phase(h, *args, **kwargs):
    return np.zeros(h.shape[0])


class HarmonicSeries:

    def __init__(self, start, end, period, amp_func, phase_func=None, h_shift_func=None, **kwargs):
        if phase_func is None:
            phase_func = zero_phase
        if h_shift_func is None:
            h_shift_func = lambda x: x
        # Array of harmonic numbers
        self.harmonics = np.arange(start, end, period)
        h = h_shift_func(self.harmonics)
        self.harmonics = h
        # Array of amplitudes
        self._amplitudes = amp_func(self.harmonics)
        self._scale = 1.
        # Array of phases
        self._phases = phase_func(self.harmonics)
        self._step_size = period

        if "normalize" in kwargs:
            self.normalize(kwargs["normalize"])

    def __mul__(self, other):
        """Multiplying the HarmonicSeries object multiplies the amplitudes"""
        self._scale = other
        return self

    def normalize(self, h_target):
        """Normalizes all amplitudes to the indicated harmonic"""
        if np.max(np.abs(self._amplitudes)) != 0.:
            self._amplitudes /= self._amplitudes[int(h_target - 1)]
        if np.max(np.abs(self._phases)) != 0.:
            self._phases /= self._phases[int(h_target - 1)]
            self._phases *= 2 * np.pi

    def evaluate(self, samprate, cycles=1, os_level=8, bandlimit=None, window=True):

        # t = np.arange(0, samprate * cycles)
        t = np.linspace(0., cycles * 2 * np.pi, samprate * cycles, endpoint=False)
        series = np.zeros(t.shape[0])
        # adj = np.cos(self.harmonics - 1) ** 2 * (np.pi / (2 * self.harmonics[-1]))  # gibbs
        adj = np.sinc(self.harmonics * np.pi / (2 * np.max(self.harmonics)))  # sigma factor
        for a, p, h, g in zip(self.amplitudes, self.phases, self.harmonics, adj):
            if h <= bandlimit / os_level if bandlimit is not None else samprate / (4 * os_level):
                # Harmonics whose waveforms do not have an integer-number of cycles within
                # the rendered region are windowed to prevent aliasing. Increasing the number
                # of cycles allows more inharmonic frequencies to fit evenly into the wavetable.
                if not window or np.abs(h * cycles - np.round(h * cycles)) < 1e-3:
                    partial = a * np.sin(float(h) * t + p)
                else:
                    print("Harmonic {} does not fit into {} cycles".format(h, cycles))
                    wnd1 = np.concatenate([(np.cos(np.pi * t[:int(samprate)] / samprate) + 1) / 2,
                                           np.zeros(int(samprate * (cycles - 1)))])
                    wnd2 = np.concatenate([np.ones(int(samprate * (cycles - 1))),
                                           wnd1[int(samprate) - 1::-1]])
                    t_ext = np.arange(0, np.floor(samprate * cycles * (1 + h - np.floor(h))))
                    wave_full = np.sin(2 * np.pi * h * t_ext / samprate + p)
                    wave1 = wave_full[:int(samprate * cycles)]
                    wave2 = wave_full[len(wave_full) - int(samprate * cycles):]
                    partial = a * (wave1 * wnd1 + wave2 * wnd2)
                series += partial * g
        return series

    @property
    def amplitudes(self):
        return self._amplitudes

    @property
    def phases(self):
        return self._phases

    @property
    def scale(self):
        return self._scale

    @property
    def max_harmonic(self):
        return self.harmonics[-1]

    @property
    def num_harmonics(self):
        return self.harmonics.shape[0]

    @property
    def step_size(self):
        return self._step_size


class Waveform():

    def __init__(self):
        self.series_ = []

    def add_series(self, *args, **kwargs):
        new_series = HarmonicSeries(*args, **kwargs)
        self.append_series(new_series)

    def append_series(self, new_series):
        self.series_.append(new_series)

    def normalize(self):
        """Normalizes all series so that the (summed) maximum harmonic (not necessarily the fundamental) is one"""
        fund_sum = np.sum([np.max(s.amplitudes) for s in self.series_])

        for s in self.series_:
            s.normalize(fund_sum)

    def generate_series(self, samprate, **kwargs):
        if "cycles" in kwargs:
            num_cycles = kwargs['cycles']
        else:
            num_cycles = 1

        length = int(samprate * num_cycles)
        sum_sines = np.zeros(length)
        # Sum sinusoids of all harmonic series
        for s in self.series_:
            s_wave = s.evaluate(samprate, **kwargs)
            s_wave /= np.max(np.abs(s_wave)) if np.max(s_wave) else 1.
            sum_sines += s_wave * s.scale

        # Normalize to the highest value
        if np.max(np.abs(sum_sines)) != 0.:
            self.waveform = sum_sines / np.max(np.abs(sum_sines))
            # self.waveform = sum_sines / (sum_sines ** 2).sum() * num_cycles
        else:
            self.waveform = sum_sines
        return self.waveform

    def generate_ifft(self, samprate):

        freq_domain = np.zeros(samprate // 2)

        for s in self.series_:
            offset = s.harmonics[0]
            top_harmonic = s.max_harmonic + offset

            amp_interp_func = interpolate.interp1d(s.harmonics, s.amplitudes)
            phase_interp_func = interpolate.interp1d(s.harmonics, s.phases)

            a_resampled = amp_interp_func(np.arange(offset, top_harmonic, s.step_size))
            p_resampled = phase_interp_func(np.arange(offset, top_harmonic, s.step_size))

            # Pad the amplitude and phase arrays with zeros if the fundamental frequency of
            # the series is not the wavetable fundamental
            if top_harmonic >= samprate // 2:
                a = a_resampled[:math.floor(samprate // 2 - np.ceil(offset))]
                p = p_resampled[:math.floor(samprate // 2 - np.ceil(offset))]
            else:
                a = np.concatenate([a_resampled, np.zeros(math.floor(samprate // 2 - top_harmonic))])
                p = np.concatenate([p_resampled, np.zeros(math.floor(samprate // 2 - top_harmonic))])

            if offset > 1:
                a = np.concatenate([np.zeros(offset), a])
                p = np.concatenate([np.zeros(offset), p])

            # Interpolate amplitude and phase values
            s_complex = a + 1.0j * p
            # Pad with the DC offset value
            s_complex = np.concatenate([[0], s_complex])

            # Add to the frequency-domain representation
            freq_domain = np.add(freq_domain, s_complex[:samprate // 2])

        # Add negative frequencies
        freq_domain = np.concatenate([freq_domain, np.conj(freq_domain[::-1])]) * samprate

        self.waveform = fft.ifft(freq_domain)

        return self.waveform

    def from_sample(self, samples, samprate, cycles=1):
        fft_length = min(samprate * cycles, samples.shape[0])
        transform = fft.fft(samples[:fft_length])
        amps = np.real(transform)
        phases = np.imag(transform)

        # Shift the fundamental frequency to bin [cycles]
        clip_region = np.argmax(np.abs(transform))
        transform = np.concatenate([np.zeros(cycles), transform[clip_region:], np.zeros(clip_region)])
        self.waveform = transform
        return self.waveform


class Wavetable():

    def __init__(self):
        self.waves_ = []


def freq_spec(x):
    return np.abs(fft.fft(x))


def plot_freqs(x, freq_max=None):
    if freq_max is None:
        freq_max = x.shape[0]
    fourier_transform = fft.fft(x)
    plt.plot(np.abs(fourier_transform[:freq_max]))
    plt.show()


def plot_fft(x, freq_max=None):
    if freq_max is None:
        freq_max = x.shape[0]
    fourier_transform = fft.fft(x)
    plt.plot(np.real(fourier_transform[:freq_max]))
    plt.plot(np.imag(fourier_transform[:freq_max]))
    plt.show()


if __name__ == "__main__":
    wt1 = Waveform()
    wt1.add_series(1, 2, 1, saw_coeffs)
    # wave = wt1.generate_series(48000, cycles=3)
    # plt.plot(np.abs(fft.fft(wave)[:100]))

    wt2 = Waveform()
    wt2.add_series(1, 200, 1, saw_coeffs)
    # wave = wt2.generate_series(48000, cycles=3)
    # wave = np.tile(wave, 10)
    # plt.plot(np.abs(fft.fft(wave)[:100]))
    # plt.show()
    IO.export_mipmap([wt1, wt2], "", "Sine-Saw", 2 ** 14, cycles_per_level=1)
