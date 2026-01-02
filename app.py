import streamlit as st
import numpy as np
import librosa
import plotly.graph_objects as go

st.set_page_config(page_title="AI Voice Noise Detector", layout="wide")

st.title("🎙️ Smart AI Voice Recommendation System")
st.write("Record your voice live, and the AI will analyze the background noise.")

# 1. Live Audio Input Widget
audio_file = st.audio_input("Click to record your voice")

if audio_file:
    # 2. Convert Audio for AI Analysis
    # We load the recorded bytes into librosa
    y, sr = librosa.load(audio_file, sr=16000)
    
    st.audio(audio_file) # Playback for user
    
    # 3. AI Logic: Smart Noise Recommendation
    # We estimate noise by looking at the quietest parts of the recording
    st.subheader("📊 AI Analysis Results")
    
    # Calculate Root Mean Square (Energy)
    rms = librosa.feature.rms(y=y)
    avg_energy = np.mean(rms)
    noise_threshold = np.percentile(rms, 20) # Assume bottom 20% is background noise
    
    # Simple Recommendation Logic
    snr = 20 * np.log10(avg_energy / (noise_threshold + 1e-6))
    
    if snr > 15:
        status = "✅ High Quality / Clean"
        rec = "Perfect for recording. Minimal background noise detected."
        color = "green"
    elif snr > 8:
        status = "⚠️ Moderate Noise"
        rec = "Readable, but consider using a noise filter or moving to a quieter room."
        color = "orange"
    else:
        status = "❌ High Noise Detected"
        rec = "Too much background noise! The AI recommends re-recording."
        color = "red"

    # 4. Display Recommendation UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Signal-to-Noise Ratio (SNR)", f"{snr:.2f} dB")
        st.markdown(f"### Status: :{color}[{status}]")
        st.info(f"**Recommendation:** {rec}")

    with col2:
        # Visualizing the Waveform
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y[:5000], line=dict(color='royalblue')))
        fig.update_layout(title="Voice Waveform (First 5000 Samples)", xaxis_title="Time", yaxis_title="Amplitude")
        st.plotly_chart(fig)

    # Export Report
    report_data = f"Audio Report\nStatus: {status}\nSNR: {snr:.2f} dB\nRecommendation: {rec}"
    st.download_button("Download AI Report", report_data, "voice_report.txt")
