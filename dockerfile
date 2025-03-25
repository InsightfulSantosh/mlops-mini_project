FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker's caching
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader stopwords wordnet

# Copy the rest of the application
COPY streamlit_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Expose the correct port
EXPOSE 8501  

# Use the correct CMD for Streamlit or Flask
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# If using Flask, use:
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
