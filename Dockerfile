FROM python:3.10-slim

# System dependencies for Boost C++, Vina, and RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    libboost-all-dev \
    swig \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement file
COPY requirements.txt .

# Install PyTorch CPU first, then rest of requirements
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
