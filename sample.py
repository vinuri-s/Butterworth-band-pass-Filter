import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.signal import freqz

lowcut = 10.0
highcut = 150.0
FRAME_RATE = 600

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    #y = filtfilt(b, a, data)
    return y

def bandpass_filter(buffer):
    return butter_bandpass_filter(buffer, lowcut, highcut, FRAME_RATE, order=6)

samplerate, data = wavfile.read('sample.wav')
filtered = np.apply_along_axis(bandpass_filter, 0, data).astype('int16')
wavfile.write('filtered_sample.wav', samplerate, filtered)


# Plot the frequency response for a few different orders.
plt.figure(1)
plt.clf()
for order in [3, 6, 9,12]:
        b, a = butter_bandpass(lowcut, highcut, FRAME_RATE, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((FRAME_RATE* 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

plt.plot([0, 0.5 * FRAME_RATE], [np.sqrt(0.5), np.sqrt(0.5)],'--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')

# Filter a noisy signal.
t = np.arange(len(data)) / float(samplerate);  # Retrieving Time
plt.figure(2)
plt.clf()
plt.plot(t, data, label='Noisy signal')

y = butter_bandpass_filter(data, lowcut, highcut, FRAME_RATE, order=9)
plt.plot(t, y, label='Filtered signal (%g Hz)')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

plt.show()