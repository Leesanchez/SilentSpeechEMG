# Silent Speech EMG Interface

A system for silent speech recognition using EMG signals.

## Overview
This project implements a silent speech interface that converts EMG signals from facial muscles into text, enabling silent communication.

## Project Structure
```
silent_speech/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model implementations
│   ├── utils/             # Utility functions
│   └── visualization/     # Visualization tools
├── tests/                 # Test suite
├── docs/                  # Documentation
├── configs/               # Configuration files
└── scripts/               # Training and analysis scripts
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Leesanchez/SilentSpeechEMG.git
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
pip install -r requirements-dev.txt  # For development
```

## Model Weights and Data Files
Due to size limitations, the following files are not included in the repository:
- Pre-trained model weights (`*.pth`)
- Training logs and visualizations (`wandb/`)
- Generated EMG data files (`*.npy`, `*.pkl`)

To obtain these files:
1. **Model Weights**: Train the model using the provided scripts or download pre-trained weights from [releases](https://github.com/Leesanchez/SilentSpeechEMG/releases).
2. **Training Logs**: These are automatically generated during training using Weights & Biases.
3. **EMG Data**: Generate synthetic data using the provided scripts or collect your own EMG data.

## Usage
1. Train the model:
```bash
python train.py --config configs/default.yaml
```

2. Run electrode reduction study:
```bash
python scripts/electrode_study.py --data_dir data/ --output_dir results/
```

3. Interactive demo:
```bash
python interactive_demo.py
```

## Development
- Run tests: `pytest tests/`
- Format code: `black .`
- Check types: `mypy src/`

## Contributing
Please read [CONTRIBUTING.md](docs/guides/CONTRIBUTING.md) for details on our code of conduct and development process.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Features

- Real-time EMG signal processing and text prediction
- Advanced noise reduction and signal filtering
- Multi-model ensemble approach for improved accuracy
- Comprehensive visualization tools
- Cross-validation training pipeline
- Interactive demo interface

## Acknowledgments

- Original research and methodology references
- Contributors and collaborators
- Supporting institutions

## Contact

Anthony Lee Sanchez - [@YourTwitter](https://twitter.com/yourusername)

Project Link: [https://github.com/Leesanchez/SilentSpeechEMG](https://github.com/Leesanchez/SilentSpeechEMG)
