# Stage 1: Build the training module
FROM pytorch/pytorch:latest as training_stage

WORKDIR /app

COPY training ./training

# Stage 2: Build the inference module
FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the training directory from the training_stage
COPY --from=training_stage /app/training ./training

COPY inference .

COPY data /app/data

COPY model.pth ./model.pth

CMD ["python", "inference.py"]