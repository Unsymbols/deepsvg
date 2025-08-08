# FROM python:3.9-slim
FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    sudo \
    libgl1-mesa-glx \
    libcairo2-dev \
    libffi-dev \
    libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ADD . /app

WORKDIR /app

RUN uv sync --locked

# Install CUDA stuff to get it running
RUN uv pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

CMD bash preprocess_and_or_train.sh

# CMD uv run dataset/preprocess.py --data_folder $DSVG_PRP_DATA_FOLDER --output_folder $DSVG_PRP_OUTPUT_FOLDER  --output_meta_file  $DSVG_PRP_OUTPUT_FOLDER/meta.csv

# CMD uv run dataset/preprocess.py --data_folder ~/o/unsymbols/preprocessing/data/0.5-svg/ --output_folder ~/o/unsymbols/preprocessing/data/22-0.7-svg-preprocessed/  --output_meta_file  ~/o/unsymbols/preprocessing/data/22-0.7-svg-preprocessed/meta.csv


# # Install Miniconda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
#     bash /miniconda.sh -b -p /opt/conda && \
#     rm /miniconda.sh && \
#     /opt/conda/bin/conda clean --all -y
#
# # Set Conda environment as default
# RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
# SHELL ["/bin/bash", "-c"]
#
# # Clone the DeepSVG repository
# WORKDIR /app
# RUN git clone https://github.com/alexandre01/deepsvg.git
#
# # Update requirements.txt to replace sklearn with scikit-learn
# WORKDIR /app/deepsvg
# RUN sed -i 's/^sklearn$/scikit-learn/' requirements.txt
#
# RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
#
# # Set up the Conda environment and install dependencies
# RUN conda create -n deepsvg python=3.7 -y && \
#     echo "conda activate deepsvg" >> ~/.bashrc && \
#     source ~/.bashrc && \
#     pip install --no-cache-dir -r requirements.txt
#
# # Optional: Set up cairosvg dependencies for specific platforms
# RUN if [ "$(uname -s)" = "Linux" ]; then \
#         apt-get update && apt-get install -y --no-install-recommends libcairo2-dev; \
#     fi
#
# # Default entrypoint to activate the environment and start the container in a shell
# CMD ["/bin/bash", "-c", "source ~/.bashrc && conda activate deepsvg && bash"]
