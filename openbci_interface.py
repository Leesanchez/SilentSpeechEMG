import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from data_utils import preprocess_openbci_emg, extract_emg_features

class OpenBCICollector:
    def __init__(self, serial_port=None):
        """Initialize OpenBCI data collection interface."""
        self.params = BrainFlowInputParams()
        if serial_port:
            self.params.serial_port = serial_port
        
        # Initialize Cyton board
        self.board_id = BoardIds.CYTON_BOARD.value
        self.board = BoardShim(self.board_id, self.params)
        self.sfreq = self.board.get_sampling_rate(self.board_id)
        self.channels = [1, 2, 3, 4, 5, 6, 7, 8]  # EMG channels
        
    def start_stream(self):
        """Start the EMG data stream."""
        self.board.prepare_session()
        self.board.start_stream()
        
    def stop_stream(self):
        """Stop the EMG data stream."""
        self.board.stop_stream()
        self.board.release_session()
        
    def get_current_data(self, duration=1.0):
        """Get the most recent duration seconds of EMG data."""
        num_samples = int(duration * self.sfreq)
        data = self.board.get_current_board_data(num_samples)
        emg_data = data[self.channels, :]
        
        # Preprocess each channel
        processed_data = np.zeros_like(emg_data)
        for i in range(len(self.channels)):
            processed_data[i] = preprocess_openbci_emg(emg_data[i])
            
        return processed_data
        
    def collect_training_sample(self, duration=2.0, label=None):
        """Collect a training sample with optional label."""
        data = self.get_current_data(duration)
        
        # Extract features from each channel
        features = []
        for channel_data in data:
            channel_features = extract_emg_features(channel_data)
            features.append(channel_features)
            
        return np.array(features), label

class DataCollectionSession:
    def __init__(self, collector, output_dir='collected_data'):
        """Initialize a data collection session."""
        self.collector = collector
        self.output_dir = output_dir
        self.samples = []
        self.labels = []
        
    def collect_sample(self, prompt, duration=2.0):
        """Collect a single labeled sample."""
        print(f"Please silently articulate: {prompt}")
        input("Press Enter when ready...")
        print("Recording...")
        
        data, _ = self.collector.collect_training_sample(duration)
        self.samples.append(data)
        self.labels.append(prompt)
        
        print("Recording complete.")
        
    def save_session(self):
        """Save the collected data."""
        np.save(f"{self.output_dir}/emg_samples.npy", np.array(self.samples))
        np.save(f"{self.output_dir}/labels.npy", np.array(self.labels)) 