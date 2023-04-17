import os
import sys

import torch
import torchaudio
from einops import rearrange
from pathlib import Path
import argparse

import json
import threading
import meteorite
import boto3
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.config import load_model_config, create_musiclm_from_config


checkpoint_directory = "/app/checkpoints/"
# checkpoint_directory = "/home/ubuntu/data/model/checkpoints/"

model_config = load_model_config(checkpoint_directory + "musiclm_large_small_context.json")
semantic_path = checkpoint_directory + "semantic.transformer.14000.pt"
coarse_path = checkpoint_directory + "coarse.transformer.18000.pt"
fine_path = checkpoint_directory + "fine.transformer.24000.pt"
return_coarse_wave = False
rvq_path = checkpoint_directory + "clap.rvq.950_no_fusion.pt"
kmeans_path = checkpoint_directory + "kmeans_10s_no_fusion.joblib"
seed = 0
results_folder = "/app/results"
# results_folder = "/home/ubuntu/data/results"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


musiclm = create_musiclm_from_config(
    model_config=model_config,
    semantic_path=semantic_path,
    coarse_path=coarse_path,
    fine_path=fine_path,
    rvq_path=rvq_path,
    kmeans_path=kmeans_path,
    device=device)

torch.manual_seed(seed)

print("created musiclm")

s3_client = boto3.client(
            "s3",
            region_name=os.environ["AWS_REGION"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
s3_bucket = os.environ["AWS_S3_BUCKET"]

print("created s3 client")


def infer(body, object_key):
    generated_wave, similarities = musiclm.generate_top_match(
        text=[body['prompt']],
        num_samples=body['num_samples'],
        num_top_matches=body['num_top_matches'],
        output_seconds=body['duration'],
        semantic_window_seconds=model_config.global_cfg.semantic_audio_length_seconds, 
        coarse_window_seconds=model_config.global_cfg.coarse_audio_length_seconds, 
        fine_window_seconds=model_config.global_cfg.fine_audio_length_seconds, 
        semantic_steps_per_second=model_config.hubert_kmeans_cfg.output_hz,
        acoustic_steps_per_second=model_config.encodec_cfg.output_hz,
        return_coarse_generated_wave=return_coarse_wave,
    )

    for i, (wave, sim) in enumerate(zip(generated_wave, similarities)):
        wave = rearrange(wave, 'b n -> b 1 n').detach().cpu()
        print(f'topk similarities: {sim}')
        for j, w in enumerate(wave):
            print(f"j: {j}")
            torchaudio.save(Path(results_folder) / Path(f'{body["prompt"][:35]}_top_match_{j}.wav'), w, musiclm.neural_codec.sample_rate)
            filepath = str(Path(results_folder) / Path(object_key))
            torchaudio.save(Path(results_folder) / Path(object_key), w, musiclm.neural_codec.sample_rate)
            s3_client.upload_file(filepath, s3_bucket, object_key)
            print(f"uploaded {filepath} to {s3_bucket} with key: {object_key}")




app = meteorite.Meteorite()
@app.predict
def predict(data):
# if __name__ == '__main__':

    body = json.loads(data)
    print(body)
    # parser = argparse.ArgumentParser(description='run inference on trained musiclm model')

    # parser.add_argument('prompt', help='prompts to generate audio for', type=str, nargs='+')
    # parser.add_argument('--num_samples', default=4, type=int)
    # parser.add_argument('--num_top_matches', default=1, type=int)
    # parser.add_argument('--model_config', default='./configs/model/musiclm_small.json', help='path to model config')
    # parser.add_argument('--semantic_path', required=True, help='path to semantic stage checkpoint')
    # parser.add_argument('--coarse_path', required=True, help='path to coarse stage checkpoint')
    # parser.add_argument('--fine_path', required=True, help='path to fine stage checkpoint')
    # parser.add_argument('--rvq_path', default='./checkpoints/clap.rvq.350.pt')
    # parser.add_argument('--kmeans_path', default='./results/hubert_kmeans/kmeans.joblib')
    # parser.add_argument('--results_folder', default='./results', type=str)
    # parser.add_argument('--return_coarse_wave', default=False, action=argparse.BooleanOptionalAction)
    # parser.add_argument('--duration', default=4, type=float, help='duration of audio to generate in seconds')
    # parser.add_argument('--seed', default=0)

    # args = parser.parse_args()    

    

    print(f'prompt: {body["prompt"]}')

    object_key = str(Path(f'{body["prompt"][:35].replace(" ", "_")}_top_match_0_{round(datetime.now().timestamp())}.wav'))

    thread = threading.Thread(target=infer, args=(body, object_key))
    thread.start()

    return {
            "key": object_key,
            "url": s3_client.generate_presigned_url('get_object', Params={'Bucket': s3_bucket, 'Key': object_key}, ExpiresIn=3600*24*7)
        }

app.start(port=4000)