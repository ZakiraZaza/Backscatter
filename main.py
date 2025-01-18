import numpy as np
from PIL import Image, ImageQt
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks, decimate, correlate, sosfilt


def read_iq_samples(filename, count=-1):
    return np.fromfile(filename, count=count, dtype=np.complex64)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def downconvert(data, carrier_freq, fs):
    t = np.arange(len(data)) / fs
    return data * np.exp(-2j * np.pi * carrier_freq * t)

def decimate_signal(y, decimation_factor):
    remaining_dec = decimation_factor

    MAX_Q = 10

    while remaining_dec > 1:
        if remaining_dec < MAX_Q:
            q = remaining_dec
        else:
            q = MAX_Q  # max decimation is 13 https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html
        y = decimate(y, q)
        assert remaining_dec % q == 0, f"Decimation factor should be divisible by {q}"
        remaining_dec = remaining_dec // q
    return y

def read_bin_to_seq(file_path, binary=False):
    # Open the file and read its contents
    with open(file_path, "r") as file:
        # Read the file contents and strip any whitespace or newline characters
        binary_string = file.read().strip()

    # Convert the string into a list of integers (0s and 1s)
    binary_list = [int(char) for char in binary_string if char in "01"]

    if not binary:
        res_list = [-1 if b == 0 else 1 for b in binary_list]
    else:
        res_list = binary_list
    return res_list

def full_cross_correlation_scipy(arr, seq):
    return correlate(seq,arr, 'full')[::-1]

def nrz_to_bin(arr):
    return [0 if b == -1 else 1 for b in arr]


def early_late_synchronizer(signal, sampling_rate, d=5, alpha=0.01, num_iterations=1000):
    """
    Early-Late Synchronization

    This function synchronizes a signal using the early-late synchronization technique. 

    Parameters:
    signal (numpy array): The input signal to be synchronized.
    sampling_rate (int): The sampling rate of the signal (samples per second).
    d (int, optional): The number of samples used for early and late synchronization. Default is 5.
    alpha (float, optional): The learning rate for phase adjustment. Default is 0.01.
    num_iterations (int, optional): The number of iterations to perform the synchronization. Default is 1000.

    Returns:
    numpy array: The synchronized signal.
    """
    phase_offset = 0  
    synced_signal = np.copy(signal)
    N = len(signal)
    for _ in range(num_iterations):
        mid_idx = int((phase_offset + sampling_rate // 2) % N)
        early_idx = int((mid_idx - d) % N)
        late_idx = int((mid_idx + d) % N)
        
        early_sample = signal[early_idx]
        late_sample = signal[late_idx]
        
        error_signal = early_sample - late_sample  
        phase_offset -= alpha * error_signal  
        phase_offset = np.clip(phase_offset, -sampling_rate//2, sampling_rate//2)
    
    synced_signal = np.roll(signal, -round(phase_offset))
    return synced_signal

# %%
# Parameters
filename = "271124_920MHz_1_tx_05_END_05_Rx_-10dbm.iq"  # IQ samples from GNURADIO, complex64 format
fs = 2e6
baudrate_tx = 1001# baudrate backscatter
bandwidth = 2 * baudrate_tx  # bandwidth low pass
preamble_bits = 80
FFT_SIZE = 2048 * 32
decimation_factor = 1
decimation = False

samples_per_symb = 2003
#samples_per_symb =  int(fs//baudrate_tx)

NUM_AVG = 1800
frequency_of_interest_low= 510000 #between which frequencies is the local oscillator located
frequency_of_interest_high= 530000

#read iq samples from file
iq_samples = read_iq_samples(filename, count=FFT_SIZE * NUM_AVG)  # read samples

#calculate power spectral density (PSD)
plt.subplot(4, 1, 1)
AVG_PSD = np.asarray([0] * FFT_SIZE, dtype=np.float64)
for i, iq in enumerate(np.split(iq_samples, NUM_AVG)):
    AVG_PSD += np.abs(np.fft.fft(iq)) ** 2 / (FFT_SIZE * fs)

AVG_PSD /= NUM_AVG
PSD_log = 10.0 * np.log10(AVG_PSD)
PSD_shifted = np.fft.fftshift(PSD_log)

f = np.arange(fs / -2.0, fs / 2.0, fs / FFT_SIZE)  # start, stop, step

plt.plot(f, PSD_shifted)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")

#Search for the local oscillator frequency with the OOK data in interval
resolution = fs / FFT_SIZE
interval_border_1= round(frequency_of_interest_low/resolution)
interval_border_2= round(frequency_of_interest_high/resolution)

peaks, _ = find_peaks(PSD_shifted[int(len(PSD_shifted)/2)+interval_border_1:int(len(PSD_shifted)/2)+interval_border_2], height=-140, distance=FFT_SIZE / 5)
sorted_idx = np.argsort(PSD_shifted[peaks])[::1]
sorted_peaks = peaks[sorted_idx]
plt.scatter(f[int(len(PSD_shifted)/2)+interval_border_1+sorted_peaks[0]], PSD_shifted[int(len(PSD_shifted)/2)+interval_border_1+sorted_peaks[0]])
print(f[int(len(PSD_shifted)/2)+interval_border_1+sorted_peaks[0]])
print(f"Resolution {resolution:.2f} Hz")
plt.grid(True)
fc = f[int(len(PSD_shifted)/2)+interval_border_1+sorted_peaks[0]]

#Put a band pass filter around this local oscillator frequency
filtered_signal = bandpass_filter(iq_samples, fc - bandwidth / 2, fc + bandwidth / 2, fs)
filtered_signal /= np.max(np.abs(filtered_signal))  # scale so max = 1

plt.subplot(4, 1, 2)
FFT_SIZE = 1024
AVG_PSD = np.asarray([0] * FFT_SIZE, dtype=np.float64)
NUM_TIMES = len(filtered_signal) // FFT_SIZE
for iq in np.split(filtered_signal[: int(NUM_TIMES * FFT_SIZE)], NUM_TIMES):
    AVG_PSD += np.abs(np.fft.fft(iq)) ** 2 / (FFT_SIZE * fs)

AVG_PSD /= NUM_AVG
PSD_log = 10.0 * np.log10(AVG_PSD)
PSD_shifted = np.fft.fftshift(PSD_log)

f = np.arange(fs / -2.0, fs / 2.0, fs / FFT_SIZE)  # start, stop, step

plt.plot(f, PSD_shifted)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")

plt.grid(True)


# Downconvert the filtered signal to baseband
baseband_signal = downconvert(filtered_signal, fc, fs)

plt.subplot(4, 1, 3)
FFT_SIZE = 1024
AVG_PSD = np.asarray([0] * FFT_SIZE, dtype=np.float64)
NUM_TIMES = len(baseband_signal) // FFT_SIZE
for iq in np.split(baseband_signal[: int(NUM_TIMES * FFT_SIZE)], NUM_TIMES):
    AVG_PSD += np.abs(np.fft.fft(iq)) ** 2 / (FFT_SIZE * fs)

AVG_PSD /= NUM_AVG
PSD_log = 10.0 * np.log10(AVG_PSD)
PSD_shifted = np.fft.fftshift(PSD_log)

f = np.arange(fs / -2.0, fs / 2.0, fs / FFT_SIZE)  # start, stop, step

plt.plot(f, PSD_shifted)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")

plt.grid(True)

#Check if we want to do a decimation
if decimation:
    decimated_signal = decimate_signal(baseband_signal, decimation_factor)  # decimate to reduce samples
    iq_power = np.abs(decimated_signal) ** 2
else:
    iq_power = np.abs(baseband_signal) ** 2

ax = plt.subplot(4, 1, 4)
plt.plot(iq_power)


# %%
# Read the pseudorandom sequence
file_path = 'pseudorandombinarysequence.txt'
print(len(read_bin_to_seq(file_path)))
total_sequence = read_bin_to_seq(file_path)
preamble_sequence = total_sequence[:preamble_bits]
pseudorandom_sequence = total_sequence[preamble_bits:len(total_sequence)]

oversampled_sequence = np.repeat(total_sequence, samples_per_symb)
oversampled_preamble_sequence = np.repeat(preamble_sequence, samples_per_symb)
oversampled_pseudorandom_sequence = np.repeat(pseudorandom_sequence, samples_per_symb)

#perform cross correlation between preamble and total signal
correlation_result_full = full_cross_correlation_scipy(iq_power, oversampled_preamble_sequence)

#find the start index of the preamble sequence
start_idx = np.argmax(correlation_result_full) - len(oversampled_preamble_sequence)
plt.figure()
ax1 = plt.subplot(2, 1, 1)
plt.plot(iq_power)
plt.subplot(2, 1, 2, sharex=ax1)
plt.scatter(start_idx, correlation_result_full[start_idx])
plt.plot(correlation_result_full)
plt.show()

# %%
# now we need to extract the positive power and negative power


signal = iq_power[start_idx:int(start_idx + len(oversampled_sequence))]

# Synchronization
signal = early_late_synchronizer(signal, fs)

plt.figure(figsize=(10, 5))
plt.plot(signal, label="Before Synchronization", color='r', alpha=0.7)
signal = early_late_synchronizer(signal, fs)
plt.plot(signal, label="After Synchronization", color='b', linestyle='dashed', alpha=0.9)
plt.title("Signal Before and After Synchronization")
plt.legend()
plt.grid()
plt.show()

print(len(signal))
print(len(oversampled_sequence))
signal_preamble = signal[:len(oversampled_preamble_sequence)]
signal_pseudorandom =  signal[len(oversampled_preamble_sequence)+1:len(oversampled_preamble_sequence)]

plt.figure()
plt.plot(signal_preamble)
plt.plot(nrz_to_bin(oversampled_preamble_sequence))
plt.show()


zero_power = np.mean(signal_preamble[oversampled_preamble_sequence == -1])
one_power = np.mean(signal_preamble[oversampled_preamble_sequence == 1])

threshold = (one_power - zero_power) / 2.0

plt.figure()
plt.hlines(y=zero_power, xmin=0, xmax=len(signal_preamble), ls="--")
plt.hlines(y=one_power, xmin=0, xmax=len(signal_preamble), ls="--")
plt.hlines(y=threshold, xmin=0, xmax=len(signal_preamble), ls="--")
plt.plot(signal_preamble)
plt.show()

R = len(signal) // samples_per_symb

signal = signal[: int(R * samples_per_symb)]
reshaped_data = signal.reshape(-1, samples_per_symb)




num_symb = len(signal)/samples_per_symb
interval_len = 100
total_symb = int((num_symb // interval_len) * interval_len)

num_intervals = total_symb // interval_len

demod = np.zeros(total_symb)
thresholds = [0] * num_intervals
ber = np.zeros(num_intervals)
# lets start from the beginning, including the preamble
for i in range(num_intervals):
    thresholds[i] = threshold
    start_i = i * interval_len
    end_i = start_i + interval_len

    start_i = start_i*samples_per_symb
    end_i= end_i*samples_per_symb

    samples = signal[start_i:end_i]
    demod_interval = np.zeros_like(samples)
    # avg_signal = samples.mean()


    demod_interval[samples > threshold] = 1.0
    sequence_part = oversampled_sequence[start_i:end_i]
    # now that we have demodulated the signal, let re-compute the zero and one power

    # update threshold with new data
    # zero_power = np.median(samples[demod_interval == 0])  # here it is zero as per above, instead of -1 above
    # one_power = np.median(samples[demod_interval == 1])
    # new_threshold = (one_power - zero_power) / 2.0
    #
    # threshold = 0.1 * threshold + 0.9 * new_threshold

    correct_symb_i = [
        1 if a == b else 0 for a, b in zip(demod_interval, nrz_to_bin(sequence_part))
    ]

    ber[i] = 1 - (np.sum(correct_symb_i) / len(correct_symb_i))

    # plt.figure()
    # plt.plot(samples)
    # plt.plot(nrz_to_bin(sequence_part))
    # # plt.hlines(y=zero_power, xmin=0, xmax=len(samples), ls="--")
    # # plt.hlines(y=one_power, xmin=0, xmax=len(samples), ls="--")
    # # plt.hlines(y=new_threshold, xmin=0, xmax=len(samples), ls="--")
    # plt.show(block=True)
# Compute the mean along axis 1 (row-wise average)
# avg_signal = reshaped_data.mean(axis=1)


# demod[avg_signal > threshold] = 1.0
# %%
plt.figure()
plt.title("Bit error rate: 2003 samples per symbol")
plt.xlabel("Interval (/100 symbols)")
plt.ylabel("Bit Error Rate (log)")
plt.plot(10*np.log10(1-ber))
plt.show()

