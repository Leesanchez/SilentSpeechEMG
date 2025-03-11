import gradio as gr
import torch
import numpy as np
from spatiotemporal_model import SilentSpeechTransformer
from data_utils import TextTransform, preprocess_openbci_emg, extract_emg_features
from visualization_utils import plot_emg_channels, plot_spectrogram
import matplotlib.pyplot as plt
from model_ensemble import ModelEnsemble
from generate_synthetic_data import generate_synthetic_emg
import time
from scipy import signal
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F
import os

class InteractiveSilentSpeechDemo:
    def __init__(self, model_path='models/fold_1'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_transform = TextTransform()
        self.has_models = False
        
        try:
            # Load model ensemble if available
            if os.path.exists(model_path):
                self.ensemble = ModelEnsemble(
                    num_models=3,
                    model_params={
                        'num_channels': 8,
                        'input_size': 128,
                        'num_classes': len(self.text_transform.chars)
                    },
                    device=self.device
                )
                self.ensemble.load_models(model_path)
                self.has_models = True
                print("Successfully loaded trained models!")
            else:
                print("No trained models found. Running in visualization-only mode.")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Running in visualization-only mode.")
        
        # Initialize T5 for post-processing
        try:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(self.device)
            self.has_nlp = True
        except Exception as e:
            print(f"Error loading NLP models: {e}")
            self.has_nlp = False
    
    def extract_advanced_features(self, emg_data):
        """Extract comprehensive features following the methodology."""
        features = []
        
        for channel_data in emg_data:
            channel_features = {}
            
            # Time-domain features
            channel_features['rms'] = np.sqrt(np.mean(channel_data**2))
            channel_features['zero_crossings'] = np.sum(np.diff(np.signbit(channel_data)))
            channel_features['mav'] = np.mean(np.abs(channel_data))
            
            # Frequency-domain features
            freqs, psd = signal.welch(channel_data, fs=1000, nperseg=256)
            channel_features['peak_freq'] = freqs[np.argmax(psd)]
            channel_features['median_freq'] = freqs[np.argwhere(np.cumsum(psd) >= np.sum(psd)/2)[0][0]]
            channel_features['mean_power'] = np.mean(psd)
            
            # Add features to list
            features.append(list(channel_features.values()))
        
        return np.array(features)
    
    def post_process_prediction(self, predicted_text, confidence):
        """Apply NLP-based post-processing using T5."""
        # Only post-process if confidence is below threshold
        if confidence < 0.8:
            input_text = f"correct text: {predicted_text}"
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            
            outputs = self.t5_model.generate(
                input_ids,
                max_length=50,
                num_beams=5,
                no_repeat_ngram_size=2,
                num_return_sequences=1
            )
            
            corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected_text
        return predicted_text
    
    def text_to_synthetic_emg(self, text: str, duration_per_char: float = 200, 
                             noise_level: float = 0.1, frequency: float = 50):
        """Generate synthetic EMG data from text."""
        # Generate synthetic EMG data
        duration_ms = len(text) * duration_per_char
        emg_data = generate_synthetic_emg(
            duration_ms=duration_ms,
            num_channels=8,
            sampling_rate=1000,
            text=text,
            noise_std=noise_level,
            base_frequency=frequency
        )
        
        # Preprocess EMG data following methodology
        processed_emg = preprocess_openbci_emg(emg_data)
        
        return processed_emg
    
    def compute_signal_metrics(self, emg_data):
        """Compute basic signal metrics when no model is available."""
        metrics = {}
        
        # Signal quality metrics
        metrics['rms'] = np.sqrt(np.mean(emg_data**2))
        metrics['snr'] = 10 * np.log10(np.mean(emg_data**2) / np.var(np.diff(emg_data)))
        
        # Frequency analysis
        for i, channel in enumerate(emg_data):
            freqs, psd = signal.welch(channel, fs=1000, nperseg=256)
            metrics[f'channel_{i}_peak_freq'] = freqs[np.argmax(psd)]
            metrics[f'channel_{i}_mean_power'] = np.mean(psd)
        
        return metrics
    
    def compute_detailed_metrics(self, emg_data, predicted_text, true_text):
        """Compute comprehensive evaluation metrics."""
        metrics = {}
        
        # Signal quality metrics
        metrics['rms'] = np.sqrt(np.mean(emg_data**2))
        metrics['snr'] = 10 * np.log10(np.mean(emg_data**2) / np.var(np.diff(emg_data)))
        
        # Recognition metrics
        char_errors = sum(1 for a, b in zip(true_text, predicted_text) if a != b)
        metrics['char_error_rate'] = char_errors / len(true_text) if true_text else 0
        metrics['word_accuracy'] = (predicted_text.split() == true_text.split()).all()
        
        # Frequency analysis
        for i, channel in enumerate(emg_data):
            freqs, psd = signal.welch(channel, fs=1000, nperseg=256)
            metrics[f'channel_{i}_peak_freq'] = freqs[np.argmax(psd)]
        
        return metrics
    
    def process_and_predict(self, text: str, duration_per_char: float = 200, 
                          noise_level: float = 0.1, frequency: float = 50,
                          num_ensemble: int = 3, confidence_threshold: float = 0.5,
                          progress=gr.Progress()):
        """Process input text and return predictions with visualizations."""
        results = {}
        
        # Step 1: Generate EMG data
        progress(0.2, desc="Generating synthetic EMG data...")
        emg_data = self.text_to_synthetic_emg(text, duration_per_char, noise_level, frequency)
        results["raw_emg"] = plot_emg_channels(emg_data, title="Raw EMG Signals", return_fig=True)
        
        # Step 2: Preprocess and extract features
        progress(0.4, desc="Extracting features...")
        features = self.extract_advanced_features(emg_data)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        if self.has_models:
            # Step 3: Model prediction
            progress(0.6, desc="Running model prediction...")
            outputs, confidence = self.ensemble.predict(features_tensor)
            _, predicted = outputs.max(-1)
            predicted_text = self.text_transform.int_to_text(predicted[0].cpu().numpy())
            
            # Step 4: NLP post-processing
            if self.has_nlp:
                progress(0.8, desc="Applying NLP post-processing...")
                final_text = self.post_process_prediction(predicted_text, confidence)
            else:
                final_text = predicted_text
                
            # Compute metrics
            metrics = self.compute_detailed_metrics(emg_data, final_text, text)
            
            # Feature importance visualization
            feature_imp_fig = plt.figure(figsize=(10, 6))
            feature_weights = torch.mean(torch.abs(outputs), dim=0).cpu().numpy()
            plt.bar(range(len(feature_weights)), feature_weights)
            plt.title('Feature Importance')
            plt.xlabel('Feature Index')
            plt.ylabel('Average Absolute Weight')
            results["feature_importance"] = feature_imp_fig
        else:
            final_text = "(Model not loaded - showing EMG visualization only)"
            confidence = 0.0
            metrics = self.compute_signal_metrics(emg_data)
            
            # Placeholder for feature importance
            feature_imp_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Model not loaded', ha='center', va='center')
            results["feature_importance"] = feature_imp_fig
        
        # Create visualizations
        results["processed_emg"] = plot_emg_channels(emg_data, title="Processed EMG Signals", return_fig=True)
        results["spectrogram"] = plot_spectrogram(emg_data, title="EMG Spectrograms", return_fig=True)
        
        # Time-frequency analysis
        time_freq_fig = plt.figure(figsize=(12, 8))
        for i, channel in enumerate(emg_data):
            plt.subplot(8, 1, i+1)
            f, t, Sxx = signal.spectrogram(channel, fs=1000, nperseg=256, noverlap=128)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.ylabel(f'Ch{i+1} Freq [Hz]')
        plt.tight_layout()
        results["time_frequency"] = time_freq_fig
        
        # Prepare results
        results.update({
            "predicted_text": final_text,
            "confidence": confidence,
            "input_text": text,
            "metrics": metrics
        })
        
        progress(1.0, desc="Complete!")
        return results
    
    def create_interface(self):
        """Create enhanced Gradio interface."""
        def predict_and_visualize(text, duration_per_char, noise_level, frequency,
                                num_ensemble, confidence_threshold, progress=gr.Progress()):
            results = self.process_and_predict(
                text, duration_per_char, noise_level, frequency,
                num_ensemble, confidence_threshold, progress
            )
            
            metrics_text = "\n".join([
                f"{k}: {v:.3f}" for k, v in results["metrics"].items()
            ])
            
            status = "VISUALIZATION MODE - Model not loaded" if not self.has_models else "Model loaded and ready"
            
            prediction_text = (
                f"Status: {status}\n\n"
                f"Input Text: {results['input_text']}\n"
                f"Predicted Text: {results['predicted_text']}\n"
                f"Confidence: {results['confidence']:.2f}\n\n"
                f"Detailed Metrics:\n{metrics_text}"
            )
            
            return [
                results["raw_emg"],
                results["processed_emg"],
                results["spectrogram"],
                results["time_frequency"],
                results["feature_importance"],
                prediction_text
            ]
        
        with gr.Blocks(title="Enhanced Silent Speech Interface Demo") as iface:
            gr.Markdown("# Silent Speech Interface Demo")
            gr.Markdown("""
            This demo simulates the EMG-based Silent Speech Recognition system.
            It follows the methodology of:
            1. EMG signal generation (simulating OpenBCI data)
            2. Signal preprocessing (20-450Hz bandpass filtering)
            3. Feature extraction (time and frequency domain)
            4. Deep learning classification
            5. NLP-based post-processing
            """)
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter text to convert to EMG...",
                        lines=2
                    )
                    with gr.Row():
                        duration = gr.Slider(
                            minimum=100,
                            maximum=500,
                            value=200,
                            label="Duration per Character (ms)"
                        )
                        noise = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=0.1,
                            label="Noise Level"
                        )
                    with gr.Row():
                        frequency = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            label="Base Frequency (Hz)"
                        )
                        ensemble_size = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Number of Ensemble Models"
                        )
                    confidence_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        label="Confidence Threshold"
                    )
                    
                    predict_btn = gr.Button("Generate and Predict")
            
            with gr.Row():
                raw_emg_plot = gr.Plot(label="Raw EMG Signals")
                processed_emg_plot = gr.Plot(label="Processed EMG Signals")
            
            with gr.Row():
                spec_plot = gr.Plot(label="Spectrograms")
                time_freq_plot = gr.Plot(label="Time-Frequency Analysis")
            
            with gr.Row():
                feature_imp_plot = gr.Plot(label="Feature Importance")
                results_text = gr.Textbox(
                    label="Prediction Results and Metrics",
                    lines=15
                )
            
            predict_btn.click(
                predict_and_visualize,
                inputs=[
                    text_input, duration, noise, frequency,
                    ensemble_size, confidence_threshold
                ],
                outputs=[
                    raw_emg_plot, processed_emg_plot,
                    spec_plot, time_freq_plot,
                    feature_imp_plot, results_text
                ]
            )
            
            gr.Examples(
                examples=[
                    ["hello world", 200, 0.1, 50, 3, 0.5],
                    ["testing 123", 250, 0.2, 40, 4, 0.6],
                    ["silent speech interface", 180, 0.15, 60, 3, 0.4],
                ],
                inputs=[
                    text_input, duration, noise, frequency,
                    ensemble_size, confidence_threshold
                ]
            )
        
        return iface

if __name__ == "__main__":
    demo = InteractiveSilentSpeechDemo()
    interface = demo.create_interface()
    interface.launch(share=True) 