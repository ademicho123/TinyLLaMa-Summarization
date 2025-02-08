# Use an official Azure ML base image
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

# Set up Python environment
RUN apt-get update && apt-get install -y python3-pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install torch transformers scikit-learn rouge-score nltk fastapi uvicorn matplotlib pandas numpy seaborn emoji azureml-mlflow azureml-core

# Set default command
CMD ["bash"]
