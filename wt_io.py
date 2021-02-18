import numpy as np
import pickle
import wave
import os


def format_mipmap(waveforms, max_fps, cycles_per_level, oversampling, **kwargs):
    # Round maxFPS up to the nearest power of 2
    max_fps = int(2 ** np.ceil(np.log(max_fps) / np.log(2)))
    if "levels" in kwargs and kwargs["levels"] is not None:
        sizes = kwargs["levels"]
    else:
        sizes = [max_fps, ]
        while sizes[-1] / 2 >= oversampling * 4:
            sizes.append(sizes[-1] / 2)

    mipmaps = np.ndarray([0, ])
    print("Oversampling Level:", oversampling)

    # Check for band-limit specification
    if "limits" in kwargs:
        limit = kwargs["limits"]
    else:
        limit = [None for w in waveforms]

    # Make sure the list of band limits matches the list of waveforms in length.
    # If it is shorter, append None-values to the BEGINNING of the list (so that
    # explicit band limits only apply to the higher-frequency tables)]
    while len(limit) < len(waveforms):
        limit.insert(0, None)

    # List of the maximum number of harmonics for each mipmap level
    max_harmonics = [s for s in sizes]
    # For each wavetable (i.e. premix):
    for wt_i, wt in enumerate(waveforms):
        # Generate full_size wavetable
        # For each sample rate:
        for lim, framerate in zip(limit, sizes):
            # Downsample wavetable
            wt_wav = wt.generate_series(framerate, cycles=cycles_per_level, bandlimit=lim)

            # Obtain proper data format
            if "format" in kwargs and kwargs["format"] == np.int16:
                # Scale the waveform to the range [-32767, 32767]
                wt_wav *= (2 ** 14 - 1) / np.max(np.abs(wt_wav.astype(np.float64)))
                mipmap = wt_wav.astype(np.int16)
            else:
                mipmap = wt_wav.astype(np.float64)

            # Scale loud waveforms, if requested
            if "scale_to" in kwargs and kwargs["scale_to"]:
                mipmap *= 1. - (np.sum(np.abs(mipmap)) / mipmap.shape[0] - kwargs["scale_to"])

            # Add mipmap to the array
            mipmaps = np.concatenate((mipmaps, mipmap), axis=0)
            # print("Added {}-sample segment of wavefrom {} to wavetable".format(framerate * cycles_per_level, wt_i + 1))
    print("Full wavetables size: {} samples, {} cycles".format(mipmaps.shape[0], cycles_per_level))

    return mipmaps, sizes, len(waveforms)


def export_mipmap_wav(waveforms, directory, filename, max_fps, cycles_per_level=1, oversampling=8, **kwargs):
    mipmaps, _, _ = format_mipmap(waveforms, max_fps, cycles_per_level, oversampling)
    # NOTE: Array datatype MUST be specified as "16-bit integer" (again) upon performing this step
    mipmaps = np.ascontiguousarray(mipmaps * (2 ** 15 - 1), dtype=np.int16)

    with open('mipmap_array.pickle', 'wb') as f:
        pickle.dump(mipmaps, f)

    # Possible BUG: unknown behavior if Reaper/Asio4All is open
    # Write Wavetable
    full_path = os.path.join(directory, filename)
    if not any([full_path[-4:] == w for w in [".wav", ".WAV"]]):
        full_path += ".wav"
    file_out = wave.open(full_path, 'wb')

    # Set file parameters
    file_out.setnchannels(1)  # 1 = Mono sound
    file_out.setsampwidth(2)  # 2 = 16-bit (2-byte) samples
    file_out.setframerate(max_fps)
    file_out.setnframes(len(mipmaps))

    file_out.writeframesraw(mipmaps)
    file_out.close()


def export_mipmap_bytes(waveforms, directory, filename, max_fps, cycles_per_level=1, oversampling=8, **kwargs):
    mipmaps, sizes, nwaves = format_mipmap(waveforms, max_fps, cycles_per_level, oversampling, **kwargs)

    header = np.array([mipmaps.shape[0], np.max(sizes), np.min(sizes), nwaves, cycles_per_level, oversampling],
                      dtype=np.uint32).tobytes("C")
    mipmap_bin = mipmaps.astype(np.float64).tobytes("C")
    bytestream = bytearray(header + mipmap_bin)

    with open(os.path.join(directory, filename), 'wb') as f:
        f.write(bytestream)


WT_FORMAT = dict({"Length": np.uint32,
                  "MaxLevelSize": np.uint32,
                  "MinLevelSize": np.uint32,
                  "NumWaveforms": np.uint32,
                  "CyclesPerLevel": np.uint32,
                  "Oversampling": np.uint32,
                  "SamplesData": np.float64})
WT_HEADER_ITEM_SIZE = 4  # 32-bit integer byte width
WT_HEADER_LENGTH = len(WT_FORMAT.keys()) - 1


def read_wt_file(path):
    header = np.fromfile(path, dtype=np.uint32, count=WT_HEADER_LENGTH, offset=0)
    data = np.fromfile(path, dtype=np.float64, count=-1, offset=WT_HEADER_LENGTH * WT_HEADER_ITEM_SIZE)
    return header, data
