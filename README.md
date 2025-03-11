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
The model weights and large data files are stored separately from this repository. Here's how to obtain them:

### Pre-trained Models
You can access the pre-trained models in two ways:
1. **Download from Hugging Face**: Our models are hosted on the Hugging Face Model Hub
   ```bash
   # Install huggingface_hub
   pip install huggingface_hub
   
   # Download the model
   python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Leesanchez/SilentSpeechEMG', filename='best_model.pth')"
   ```

2. **Direct Download**: Access the models through our Google Drive repository
   - [best_model.pth](https://drive.google.com/drive/folders/your-folder-id) (319 MB)
   - [electrode_study_model.pth](https://drive.google.com/drive/folders/your-folder-id) (280 MB)

### Training Your Own Models
If you prefer to train your own models:
1. Prepare your dataset using the scripts in `src/data/`
2. Configure your training parameters in `configs/training_config.yaml`
3. Run the training script:
   ```bash
   python train.py --config configs/training_config.yaml
   ```

### Model Checkpoints
During training, model checkpoints are saved in the `checkpoints/` directory (not tracked by Git). You can:
- Use Weights & Biases (wandb) to track your experiments
- Configure checkpoint saving frequency in the training config
- Access historical training runs through wandb.ai

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
