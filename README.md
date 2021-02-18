# WavetableEditor
Python tools for generating wavetables for use in audio plugins

The file `TablitsaSynthesizer/PeriodicTable.py` contains the script used to generate the wavetables for my project [Tablitsa Wavetable Synthesizer](https://github.com/radiofarmer/Tablitsa-Synthesizer).

# User Guide
This repository is still a work in progress. The current functionality is essentially the minimum required for its current use, namely producing audio data in the format required for the Tablitsa synthesizer. The core data-generation functions are rather simple at the moment and probably not of much use to others, so the primary purpose of making this repository available is to allow anyone who is interested in using Tablitsa to make their own wavetables. For this, I recommend using the `PeriodicTable.py` script, which generates all 118 wavetables in the proper format. By editing the global variables of this script, you can adjust the harmonic content and relative amplitudes of the wavetables' frequency components, generating new timbres. The `wt_io.py` module takes care of formating the files into so that they are usable for Tablitsa, but for reference, the `.wt` format is as following:

*Header*: (24 bytes total)
* Length of audio data, in samples: 32-bit unsigned integer
* Largest mipmap level\*, in samples: 32-bit unsigned integer
* Smallest mipmap level, in samples: 32-bit unsigned integer
* Number of unique waveforms (timbres): 32-bit unsigned integer
* Oversampling, as ratio of the number of samples in a mipmap level to maximum number of samples to be read during a single cycle of that level: 32-bit unsigned integer
*Data*
* Samples, of arbitrary number: 64-bit float

\* To decrease aliasing, wavetables are stored as bandlimited mipmaps, each a factor of two smaller than the last. "Waveform", "wavetable position", or "timbre" refers to the prototypical signal of arbitrary length. "Mipmap" refers to a series of waveforms sampled at defined frequencies and bandlimited to a frequency defined by a certain multiple of the fundamental frequency (one cycle of the waveform at the given sample rate). "Mipmap level" refers to one of the sampled waveforms in this list, and is the buffer from which audio data is read in a synthesizer program. The correct mipmap level---that which provides maximum harmonic content with minimal aliasing---for a given frequency (in Hz) can be calculated as follows, where `n` is the number of samples in the mipmap level:

`n = 2 ^ ceil(log2(SAMPLE_RATE / NOTE_FREQUENCY * OVERSAMPLING_RATIO))`

For example, to play A440 at 44100 samples/second with a 8x oversampling ratio:

  1. Divide the sample rate by the note frequency to get the number of samples spanned by one cycle of this frequency: `44100 samples s^-1 * 1 / (440 cycles s^-1) = 100.2272 samples/cycle`
  2. Multiply by the oversampling ratio to get the required oversampled mipmap size, assuming you had a mipmap level for every possible frequency: `100.2272 * 8 = 801.8182`
  3. Since you don't have a mipmap level for every possible frequency, get the next largest power of two: 
      1. Take the base-two logarithm of the theoretical mipmap size: `log2(801.8182) = 9.6471`
      2. Round up: `ceil(9.6471) = 10`
      3. Raise 2 to this power: `2^10 = 1024`.
  4. Therefore, the mipmap of size 1024 is the largest available adequately bandlimited mipmap level for this frequency. With 8x oversampling, this sampled waveform can be used to play notes with between 64 and 128 samples per cycle (at 44100 samples/s, that's 689.06 and 344.53 Hz, respectively) and is therefore band-limited to `64 / 2 = 1024 samples / (8 * 4) = 32` times the fundamental frequency. In other words: the 64-sample-per-cycle note represents a frequency 32 times smaller than the master Nyquist frequency, and so it can contain no harmonics higher than 32, as these would be played back at a frequency greater than the Nyquist frequency and would get aliased.
