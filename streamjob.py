import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import os
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    # Save the uploaded PDF to a temporary file
    with open("temp_resume.pdf", "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    
    text = ""
    with open("temp_resume.pdf", "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
    
    # Remove the temporary file
    os.remove("temp_resume.pdf")
    return text

# Load the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Process job descriptions
job_desc_df = pd.read_csv('job_descriptions.csv')
job_descriptions = job_desc_df['job_description'].tolist()
job_desc_inputs = tokenizer(job_descriptions, padding=True, truncation=True, return_tensors='tf', max_length=128)
job_desc_inputs = {key: tf.convert_to_tensor(val) for key, val in job_desc_inputs.items()}  # Convert to TensorFlow tensors
job_desc_outputs = model(job_desc_inputs)
job_desc_embeddings = tf.reduce_mean(job_desc_outputs.last_hidden_state, axis=1)

# Load and process resumes
uploaded_file = st.file_uploader("Upload a Resume", type=["pdf"])
if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    resume_texts = resume_text.split('\n\n')  # Split resumes based on some delimiter, adjust as needed
    
    resume_embeddings = []
    for text in resume_texts:
        resume_inputs = tokenizer(text, padding=True, truncation=True, return_tensors='tf', max_length=128)
        resume_inputs = {key: tf.convert_to_tensor(val) for key, val in resume_inputs.items()}  # Convert to TensorFlow tensors
        resume_outputs = model(resume_inputs)
        resume_embedding = tf.reduce_mean(resume_outputs.last_hidden_state, axis=1)
        resume_embeddings.append(resume_embedding)
    
    resume_embeddings = tf.concat(resume_embeddings, axis=0)
    
    # Calculate cosine similarity between job descriptions and resumes
    similarity_matrix = cosine_similarity(job_desc_embeddings.numpy(), resume_embeddings.numpy())

    # Rank and select top 5 candidates for each job description
    top_candidates = {}
    for i, job_desc in enumerate(job_desc_df['position_title']):
        similarity_scores = similarity_matrix[i]
        ranked_indices = similarity_scores.argsort()[::-1][:5]
        top_candidates[job_desc] = [resume_texts[idx] for idx in ranked_indices]

    # Display the top 5 candidates for each job description
    for job_desc, candidates in top_candidates.items():
        st.write(f"Job Description: {job_desc}")
       # for i, candidate in enumerate(candidates, 1):
            #st.write(f"Top Candidate {i}:")
            #st.write(candidate)
            