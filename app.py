from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
from silent_speech_recognition import SilentSpeechModel, Preprocessor, FeatureExtractor, NLPPostprocessor
import json

app = Flask(__name__)

# Initialize components
model = SilentSpeechModel(input_size=1000)  # Adjust input size as needed
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

preprocessor = Preprocessor()
feature_extractor = FeatureExtractor()
nlp_processor = NLPPostprocessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get EMG data from request
        data = request.json
        emg_data = np.array(data['emg_data'])
        
        # Preprocess
        processed_data = preprocessor.preprocess_signal(emg_data)
        
        # Extract features
        features = feature_extractor.extract_features(processed_data)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(features_tensor)
            prediction = output.argmax(dim=1).item()
        
        # Post-process with NLP
        text = nlp_processor.beam_search_decode([str(prediction)])
        
        return jsonify({
            'success': True,
            'prediction': text,
            'confidence': float(torch.softmax(output, dim=1)[0][prediction])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 