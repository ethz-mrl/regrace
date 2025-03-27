FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Install Python
RUN apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken && apt-get -y update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*