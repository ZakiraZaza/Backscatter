import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

def full_cross_correlation_scipy(arr, seq):
    return correlate(arr, seq, 'full')

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
        mid_idx = (int(phase_offset) + sampling_rate // 2) % N
        early_idx = (mid_idx - d) % N
        late_idx = (mid_idx + d) % N
        
        early_sample = signal[early_idx]
        late_sample = signal[late_idx]
        
        error_signal = early_sample - late_sample  
        phase_offset -= alpha * error_signal  
        phase_offset = np.clip(phase_offset, -sampling_rate//2, sampling_rate//2)
    
    synced_signal = np.roll(signal, -round(phase_offset))
    return synced_signal

# Generating NRZ signal
fs = 2003  # Sampling frequency
n_bits = 100  # Number of bits in the signal
bit_duration = 20  # Number of samples per bit
nrz_bits = np.random.choice([-1, 1], n_bits)  # Generated NRZ bits
nrz_signal = np.repeat(nrz_bits, bit_duration)  # Oversampling to the specified number of samples per bit

# Adding preamble and noise
preamble_sequence = np.random.choice([-1, 1], 80)  # 80-bit preamble
oversampled_preamble_sequence = np.repeat(preamble_sequence, bit_duration)

# Simulating received signal (preamble + NRZ signal + noise)
received_signal = np.concatenate([
    np.random.randn(500) * 0.5,  # Initial noise
    oversampled_preamble_sequence,
    nrz_signal + np.random.randn(len(nrz_signal)) * 0.1,  # NRZ signal with noise
    np.random.randn(500) * 0.5  # Final noise
])

# Cross-correlation with preamble to find the start of the signal
correlation_result = full_cross_correlation_scipy(received_signal, oversampled_preamble_sequence)
peaks, _ = find_peaks(correlation_result, height=np.max(correlation_result) * 0.8)
start_idx = peaks[0] if len(peaks) > 0 else np.argmax(correlation_result)

# Extracting the signal
signal_length = len(nrz_signal)
start_idx = max(0, min(start_idx, len(received_signal) - signal_length))  # Ensure valid index
signal = received_signal[start_idx:start_idx + signal_length]

# Synchronization
synced_signal = early_late_synchronizer(signal, fs)

plt.figure(figsize=(10, 5))
plt.plot(signal, label="Before synchronization", color='r', alpha=0.7)
plt.plot(synced_signal, label="After synchronization", color='b', linestyle='dashed', alpha=0.9)
plt.title("Signal before and after synchronization")
plt.legend()
plt.grid()
plt.show()