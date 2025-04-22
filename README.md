# Silent Speech Recognition Using EMG and NLP

This project implements a deep learning-based silent speech recognition system using EMG signals and NLP techniques. The system can recognize speech from muscle activity without requiring audible vocalization.

## Project Structure

```
.
├── silent_speech_model.ipynb     # Jupyter notebook with model implementation
├── silent_speech_recognition.py  # Main Python script
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Jupyter Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `silent_speech_model.ipynb`

### Running the Python Script

```bash
python silent_speech_recognition.py
```

## Model Architecture

The model combines several components:

1. **Signal Preprocessing**
   - Bandpass filtering (20-450 Hz)
   - Normalization
   - Noise removal

2. **Feature Extraction**
   - MFCC coefficients
   - Delta and delta-delta features
   - PCA for dimensionality reduction

3. **Deep Learning Model**
   - Feature extraction layers
   - Transformer encoder
   - Classification head

4. **NLP Post-processing**
   - T5 model for text generation
   - Beam search decoding
   - Language model integration

## Evaluation Metrics

The model is evaluated using:
- Phoneme recognition accuracy
- Word-level accuracy
- Character error rate (CER)
- Confusion matrix analysis

## Deployment

The model can be deployed in several ways:

1. **Real-time Processing**
   - Use the `RealTimeProcessor` class for live EMG signal processing
   - Integrate with OpenBCI hardware for real-time data acquisition

2. **Web Interface**
   - A simple web interface can be created using Flask or Streamlit
   - Allows for easy interaction with the model

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Gregorio Giuseppe Orlando
- Anthony-Lee Sanchez
- Tomás Mesalles Mejía
- Ronald Sebastián Beltrán

## Acknowledgments

- OpenBCI for hardware support
- Hugging Face for transformer models
- The silent speech research community 