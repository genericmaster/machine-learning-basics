import numpy as np
import matplotlib.pyplot as plt

# Simple example - two sine waves added together
sr = 16000
t = np.linspace(0, 1, sr)  # 1 second

freq1 = 440   # A note
freq2 = 880   # A note one octave up
wave = np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)

# Fourier transform
fft = np.abs(np.fft.rfft(wave))
freqs = np.fft.rfftfreq(len(wave), 1/sr)

plt.plot(freqs[:2000], fft[:2000])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT — you should see two spikes at 440 and 880')
plt.show()