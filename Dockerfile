FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /app

RUN pip install einops vector-quantize-pytorch==0.10.15 librosa==0.10.0 torchlibrosa==0.1.0 ftfy tqdm transformers encodec==0.1.1 gdown accelerate>=0.17.0 beartype joblib h5py sklearn wget
RUN pip install -U torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install meteorite
RUN pip install boto3

COPY . .

EXPOSE 4000

CMD ["python", "scripts/infer_top_match.py"]
