# Use an official Python runtime as a parent image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install make utility (for running the Makefile)
RUN apt-get update && apt-get install -y make

# Set environment variables for Miniconda installation
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install dependencies required for Miniconda
RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -f -p $CONDA_DIR && \
    rm miniconda.sh

# Update Conda
RUN conda update -y conda

# Set the working directory in the container
WORKDIR /code

# Copy the current directory contents into the container at /app
COPY forestid /code/forestid
COPY model /code/model
COPY environment.yml /code
COPY Makefile /code
COPY app /code/app
COPY pyproject.toml /code

# Install any needed packages specified in requirements.txt
RUN make venv

# Expose port 8050 for Dash
EXPOSE 8050