# app.py
import streamlit as st
import sounddevice as sd
import numpy as np

st.set_page_config(page_title="AI Smart Classroom Noise Detection", page_icon="🎓")
st.title("🎓 AI Smart Classroom Noise Detection")
st.write("Detect classroom noise levels in real-time using your microphone.")

# -----------------------------
# Function: Calculate dB
# -----------------------------
def calculate_db(audio):
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return 0
    db = 20 * np.log10(rms)
    return abs(db)

# -----------------------------
# Function: Classify Noise
# -----------------------------
def classify_noise(db):
    if db < 40:
        return "Low Noise"
    elif db < 70:
        return "Medium Noise"
    else:
        return "High Noise"

# -----------------------------
# Microphone Input
# -----------------------------
duration = st.slider("Recording Duration (seconds)", min_value=1, max_value=10, value=3)
fs = 44100  # Sampling rate

if st.button("🎤 Record Classroom Sound"):
    st.info("Recording...")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        st.success("Recording finished!")

        db = calculate_db(audio)
        status = classify_noise(db)

        st.write(f"**Noise Level:** {db:.2f} dB")
        st.write(f"**Status:** {status}")

        if status == "High Noise":
            st.error("⚠ High noise detected! Please maintain silence.")
        elif status == "Medium Noise":
            st.warning("⚠ Moderate noise. Try to be quieter.")
        else:
            st.success("✅ Low noise. Good environment for study.")

    except Exception as e:
        st.error(f"Error recording audio: {e}")

# -----------------------------
# Optional: Show Graph of Audio
# -----------------------------
if st.checkbox("Show Audio Waveform"):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 3))
        plt.plot(audio)
        plt.title("Audio Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error plotting waveform: {e}")

st.markdown("---")
st.markdown("**Developed by:** Amna Mudassar Ali")

