import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

st.title("🔍 Class Noise Detector")
st.write("Upload your dataset to find potentially mislabeled rows.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    
    target_col = st.selectbox("Select the Label (Target) column", df.columns)
    
    if st.button("Detect Noise"):
        # Simple Logic: Cross-validation prediction
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # This is a basic example using simple prediction mismatch
        model = RandomForestClassifier()
        preds = cross_val_predict(model, X, y, cv=5)
        
        # Identify mismatches
        df['is_noise'] = preds != y
        noise_df = df[df['is_noise'] == True]
        
        st.subheader(f"Found {len(noise_df)} suspicious labels")
        st.dataframe(noise_df)
        
        # Download button for cleaned data
        cleaned_df = df[df['is_noise'] == False].drop(columns=['is_noise'])
        st.download_button("Download Cleaned Data", cleaned_df.to_csv(index=False), "cleaned_data.csv")
        
