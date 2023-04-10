FROM gcr.io/deeplearning-platform-release/tf-gpu.2-11
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt