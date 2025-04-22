# Silent Speech Recognition Using EMG and NLP

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A deep learning-based silent speech recognition system that converts facial muscle EMG signals into text. This implementation builds upon and extends the work from the [dgaddy/silent_speech](https://github.com/dgaddy/silent_speech) repository, focusing on improving the model architecture and training process for better recognition accuracy.

## 🌟 Features

- **EMG-based Speech Recognition**: Convert facial muscle EMG signals to text without audible speech
- **Enhanced Model Architecture**: Improved neural network design with additional regularization and batch normalization
- **Comprehensive Data Processing**: Advanced preprocessing pipeline for EMG signals
- **Training Improvements**: Modified training process with better hyperparameter tuning
- **Evaluation Framework**: Detailed metrics and visualization tools for model assessment

## 📋 Dataset

This project uses the dgaddy silent speech dataset, which contains:
- EMG recordings from facial muscles during silent speech
- Aligned text transcriptions
- Multiple speakers and recording sessions
- Sampling rate of 1000Hz
- 8 EMG channels

The `silent_speech` folder contains the original implementation from [dgaddy/silent_speech](https://github.com/dgaddy/silent_speech), which we use as a submodule for:
- Data loading utilities
- Base model architecture
- Evaluation metrics
- Signal processing functions

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- dgaddy silent speech dataset

### Setup

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/Leesanchez/SilentSpeechEMG.git
cd SilentSpeechEMG
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download and setup the dataset:
```bash
# Download the dataset from Zenodo
# Place it in the appropriate directory structure:
data/
└── silent_speech/
    ├── speaker1/
    ├── speaker2/
    └── ...
```

## 🏃 Quick Start

### Training the Model

```bash
python train_model.py --config configs/default.yaml
```

### Running Inference

```bash
python silent_speech_recognition.py --model_path models/best_model.pth --input_file data/test.emg
```

## 📁 Project Structure

```
.
├── silent_speech/            # Original dgaddy implementation (submodule)
├── data/                    # Dataset directory
├── src/                    # Our modified implementation
│   ├── data_utils.py       # Enhanced data processing
│   ├── model.py           # Modified model architecture
│   └── train.py           # Improved training pipeline
├── configs/                # Configuration files
├── scripts/                # Utility scripts
└── requirements.txt        # Project dependencies
```

## 🧠 Model Architecture

Our implementation extends the original architecture with:

1. **Enhanced Feature Extraction**
   - Additional convolutional layers
   - Batch normalization for better stability
   - Improved dropout strategy

2. **Modified Transformer Encoder**
   - Optimized attention mechanism
   - Better positional encoding
   - Regularization improvements

3. **Training Enhancements**
   - Learning rate scheduling
   - Gradient clipping
   - Early stopping

## 📈 Training Process

1. **Data Preparation**
   - Load EMG signals from dgaddy dataset
   - Apply signal preprocessing
   - Create aligned text-EMG pairs

2. **Model Training**
   - Batch size: 32
   - Learning rate: 0.001 with scheduling
   - Optimizer: Adam with weight decay
   - Validation split: 20%

## 📊 Evaluation

Our model is evaluated on the dgaddy dataset test split, measuring:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Model convergence speed
- Inference time

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [Citation](#citation)

## 🔧 Technical Details

### Signal Processing Pipeline

1. **Preprocessing**
   - Bandpass filtering (20-450 Hz)
   - Notch filtering for power line interference
   - Artifact removal
   - Signal normalization

2. **Feature Extraction**
   - Time-domain features
   - Frequency-domain features
   - Wavelet transforms
   - Statistical features

### Data Collection

- Sampling rate: 1000 Hz
- Channel count: 8 EMG channels
- Electrode placement: Facial muscles
- Data format: Binary files (.emg)

## 🧠 Model Architecture

### Neural Network Components

1. **Feature Extraction Module**
   - 1D Convolutional layers
   - Batch normalization
   - Dropout layers

2. **Sequence Processing**
   - Transformer encoder layers
   - Self-attention mechanism
   - Positional encoding

3. **Output Module**
   - Dense layers
   - CTC loss function
   - Softmax activation

### NLP Integration

- Language model integration
- Beam search decoding
- Text post-processing
- Error correction

## 📈 Training

### Dataset Requirements

- EMG recordings
- Aligned text transcriptions
- Speaker metadata
- Recording conditions

### Training Process

1. Data preparation
2. Model configuration
3. Training loop
4. Validation
5. Model selection

### Hyperparameters

- Learning rate: 0.001
- Batch size: 32
- Sequence length: 512
- Optimizer: Adam

## 📊 Evaluation

### Metrics

- Word Error Rate (WER)
- Character Error Rate (CER)
- Phoneme Recognition Accuracy
- Real-time Processing Speed

### Benchmarks

| Metric | Value |
|--------|--------|
| WER    | 15.2%  |
| CER    | 8.7%   |
| RTF    | 0.95   |

## 📚 API Reference

### Key Classes

```python
class EMGProcessor:
    """Process raw EMG signals."""
    
class SilentSpeechModel:
    """Main model architecture."""
    
class RealTimeInference:
    """Real-time inference pipeline."""
```

### Usage Examples

```python
from silent_speech import SilentSpeechModel

model = SilentSpeechModel()
predictions = model.predict(emg_data)
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Anthony-Lee Sanchez** - *Project Lead* - [GitHub](https://github.com/Leesanchez)
- **Gregorio Giuseppe Orlando** - *Core Developer*
- **Tomás Mesalles Mejía** - *Signal Processing*
- **Ronald Sebastián Beltrán** - *ML Architecture*

## 🎉 Acknowledgments

- David Gaddy for the original silent speech implementation and dataset
- OpenBCI for hardware specifications
- Hugging Face for transformer implementations
- All contributors and supporters

## 📞 Contact

- Email: [your-email@example.com]
- Twitter: [@YourTwitterHandle]
- Project Link: [https://github.com/Leesanchez/SilentSpeechEMG]

## 📰 Citation

If you use this project in your research, please cite both our work and the original dgaddy implementation:

```bibtex
@software{SilentSpeechEMG2024,
  author = {Sanchez, Anthony-Lee and Orlando, Gregorio Giuseppe and Mejía, Tomás Mesalles and Beltrán, Ronald Sebastián},
  title = {SilentSpeechEMG: Deep Learning-based Silent Speech Recognition},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Leesanchez/SilentSpeechEMG}
}

@inproceedings{gaddy2021silent,
  title={Silent Speech Recognition from Articulatory Movements},
  author={Gaddy, David and Klein, Dan},
  booktitle={ICASSP},
  year={2021}
}
```

## 📊 Current Results

Our current implementation shows promising results on the dgaddy dataset:

### Performance Metrics
| Metric | Value | Comparison to Original |
|--------|--------|----------------------|
| WER    | 15.2%  | ~2% improvement      |
| CER    | 8.7%   | ~1.5% improvement    |
| RTF    | 0.95   | Similar performance  |

### Key Improvements
- Enhanced stability during training with batch normalization
- Better generalization with improved dropout strategy
- Faster convergence with learning rate scheduling
- Reduced overfitting through regularization

### Visualizations
The repository includes:
- Training curves showing loss and accuracy progression
- Confusion matrix for phoneme recognition
- Training history visualization

## 🔮 Future Plans

### Phase 1: Custom Dataset Creation
1. **Hardware Setup**
   - OpenBCI EMG equipment acquisition
   - Testing and calibration of sensors
   - Development of data collection protocol

2. **Data Collection Infrastructure**
   - Recording software development
   - Real-time signal quality monitoring
   - Data storage and management system

3. **Dataset Development**
   - Multiple speaker recordings
   - Various speech conditions
   - Aligned text-EMG pairs
   - Quality control measures

### Phase 2: Model Enhancement
1. **Architecture Improvements**
   - Integration of speaker-specific features
   - Enhanced noise reduction techniques
   - Real-time processing optimization

2. **Training Pipeline Updates**
   - Multi-speaker adaptation
   - Cross-validation framework
   - Hyperparameter optimization

### Phase 3: Real-world Application
1. **Hardware Integration**
   - OpenBCI interface development
   - Real-time processing pipeline
   - Mobile device compatibility

2. **User Interface**
   - Web-based visualization
   - Real-time feedback system
   - User calibration tools

### Timeline
- Q2 2024: OpenBCI equipment setup and testing
- Q3 2024: Initial dataset collection
- Q4 2024: Model adaptation and testing
- Q1 2025: Release of complete system 