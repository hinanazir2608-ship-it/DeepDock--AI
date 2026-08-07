FROM python:3.10-slim

# System level C++ dependencies install karne ke liye
RUN apt-get update && apt-get install -y \
    build-essential \
    libboost-all-dev \
    swig \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements file copy karke dependencies install karna
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baqi ka code copy karna
COPY . .

# Streamlit port expose karna
EXPOSE 8501

# Application run karne ki command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
