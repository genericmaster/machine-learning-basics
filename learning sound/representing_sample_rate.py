import librosa
import matplotlib.pyplot as plt
y,sr = librosa.load(librosa.ex('trumpet'))
print(y)
print(sr)
print(len(y)/sr)

plt.plot(y[:1000])  # just first 1000 samples so it's not overcrowded
plt.show()