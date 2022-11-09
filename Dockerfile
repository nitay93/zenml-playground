FROM zenmldocker/zenml:0.20.5

COPY . /workspace
WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install "cython>=0.29.21,<1.0.0" \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove gcc libc6-dev

RUN zenml stack set default