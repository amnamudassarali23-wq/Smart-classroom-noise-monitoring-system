import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from cleanlab.filter import find_label_issues
import plotly.express as px

st.set_page_config(page_title="Smart AI Noise Fixer", layout="wide")

st.title("🤖 Smart AI Noise Recommendation System")
st.write("Upload your dataset, and the AI will recommend which labels to change.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # 1. Setup Data
    st.subheader("1. Configure Data")
    cols = df.columns.tolist()
    target = st.selectbox("Which column is the Label?", cols)
    features = [c for c in cols if c != target]
    
    if st.button("Run AI Analysis"):
        # Preprocessing (Simple version for Lab)
        X = pd.get_dummies(df[features])
        y = df[target].astype('category').cat.codes
        mapping = dict(enumerate(df[target].astype('category').cat.categories))
        
        # 2. AI Logic - Get Prediction Probabilities
        model = RandomForestClassifier()
        # We use simple fit/predict_proba for the lab demo
        model.fit(X, y)
        probs = model.predict_proba(X)
        
        # 3. Noise Recommendation using Cleanlab
        # This identifies indices where the label is likely wrong
        issue_indices = find_label_issues(labels=y.values, pred_probs=probs)
        
        # 4. Create Recommendations
        df['AI_Recommendation'] = "Keep"
        df.loc[issue_indices, 'AI_Recommendation'] = "⚠️ Relabel/Review"
        
        # Show Results
        st.subheader("2. Smart Recommendations")
        
        # Visualization
        fig = px.pie(df, names='AI_Recommendation', title="Dataset Health", color_discrete_sequence=['#00CC96', '#EF553B'])
        st.plotly_chart(fig)
        
        # Filtered Table
        noise_only = df[df['AI_Recommendation'] == "⚠️ Relabel/Review"]
        st.write(f"The AI found **{len(noise_only)}** suspicious labels:")
        st.dataframe(noise_only)
        
        st.download_button("Download Recommendations", df.to_csv(index=False), "ai_recommendations.csv")
