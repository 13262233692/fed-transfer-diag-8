import numpy as np
from typing import Dict, List, Tuple
import pickle
import os


class FaultDiagnosticsDataSimulator:
    def __init__(self, sample_rate: int = 1000, duration: float = 1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.time = np.linspace(0, duration, self.n_samples)
        
        self.fault_types = {
            'normal': {'freq': 50, 'amplitude': 1.0, 'noise': 0.1},
            'bearing': {'freq': 120, 'amplitude': 2.0, 'noise': 0.3},
            'gear': {'freq': 80, 'amplitude': 1.8, 'noise': 0.25},
            'unbalance': {'freq': 25, 'amplitude': 2.5, 'noise': 0.2},
            'misalignment': {'freq': 100, 'amplitude': 2.2, 'noise': 0.28}
        }
        
        self.working_conditions = {
            'factory_A': {'speed': 1.0, 'load': 1.0, 'noise_factor': 1.0},
            'factory_B': {'speed': 1.2, 'load': 0.8, 'noise_factor': 1.3},
            'factory_C': {'speed': 0.9, 'load': 1.2, 'noise_factor': 0.9},
            'factory_D': {'speed': 1.1, 'load': 1.1, 'noise_factor': 1.5}
        }
    
    def generate_vibration_signal(self, fault_type: str, condition: str) -> np.ndarray:
        fault_params = self.fault_types[fault_type]
        cond_params = self.working_conditions[condition]
        
        base_freq = fault_params['freq'] * cond_params['speed']
        amplitude = fault_params['amplitude'] * cond_params['load']
        noise_level = fault_params['noise'] * cond_params['noise_factor']
        
        signal = amplitude * np.sin(2 * np.pi * base_freq * self.time)
        
        harmonics = [2, 3, 4]
        for harmonic in harmonics:
            signal += (amplitude / harmonic) * np.sin(2 * np.pi * base_freq * harmonic * self.time)
        
        signal += np.random.normal(0, noise_level, self.n_samples)
        
        if fault_type != 'normal':
            impulse_freq = max(base_freq / 4, 1.0)
            impulse_amp = amplitude * 0.5
            num_impulses = min(int(self.duration * impulse_freq), 20)
            for i in range(num_impulses):
                idx = int(i * self.sample_rate / max(impulse_freq, 1.0))
                if idx < self.n_samples:
                    remaining = self.n_samples - idx
                    impulse_len = min(50, remaining)
                    signal[idx:idx+impulse_len] += impulse_amp * np.exp(-np.arange(impulse_len)/10)
        
        return signal
    
    def generate_dataset(self, condition: str, n_samples_per_class: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        
        for fault_idx, fault_type in enumerate(self.fault_types.keys()):
            for _ in range(n_samples_per_class):
                signal = self.generate_vibration_signal(fault_type, condition)
                X.append(signal)
                y.append(fault_idx)
        
        return np.array(X), np.array(y)
    
    def generate_federated_datasets(self, n_samples_per_class: int = 100) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        datasets = {}
        for condition in self.working_conditions.keys():
            X, y = self.generate_dataset(condition, n_samples_per_class)
            datasets[condition] = (X, y)
        return datasets
    
    def save_datasets(self, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for condition, (X, y) in datasets.items():
            filepath = os.path.join(output_dir, f'{condition}_dataset.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump({'X': X, 'y': y}, f)
            print(f'Saved dataset for {condition}: {X.shape}')
    
    def load_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y']


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True) + 1e-8
    return (signal - mean) / std


def to_spectrogram(signal: np.ndarray, nperseg: int = 64, noverlap: int = 32) -> np.ndarray:
    from scipy import signal as scipy_signal
    freqs, times, Sxx = scipy_signal.spectrogram(
        signal, fs=1000, nperseg=nperseg, noverlap=noverlap, mode='magnitude'
    )
    Sxx = np.log10(Sxx + 1e-10)
    return Sxx
