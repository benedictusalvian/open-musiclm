import os
import sys

import torch
import torchaudio
from einops import rearrange
from pathlib import Path
import argparse

import json
import meteorite

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.config import load_model_config, create_musiclm_from_config

app = meteorite.Meteorite()
@app.predict
def predict(data):
# if __name__ == '__main__':

    body = data.decode("utf-8")
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

    model_config = load_model_config(body.model_config)

    semantic_path = body.semantic_path
    coarse_path = body.coarse_path
    fine_path = body.fine_path
    return_coarse_wave = body.return_coarse_wave
    duration = body.duration
    kmeans_path = body.kmeans_path
    rvq_path = body.rvq_path
    seed = body.seed
    results_folder = body.results_folder

    Path(results_folder).mkdir(parents=True, exist_ok=True)

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

    print(f'prompt: {body.prompt}')

    generated_wave, similarities = musiclm.generate_top_match(
        text=body.prompt,
        num_samples=body.num_samples,
        num_top_matches=body.num_top_matches,
        output_seconds=duration,
        semantic_window_seconds=model_config.global_cfg.semantic_audio_length_seconds, 
        coarse_window_seconds=model_config.global_cfg.coarse_audio_length_seconds, 
        fine_window_seconds=model_config.global_cfg.fine_audio_length_seconds, 
        semantic_steps_per_second=model_config.hubert_kmeans_cfg.output_hz,
        acoustic_steps_per_second=model_config.encodec_cfg.output_hz,
        return_coarse_generated_wave=return_coarse_wave,
    )

    for i, (wave, sim) in enumerate(zip(generated_wave, similarities)):
        wave = rearrange(wave, 'b n -> b 1 n').detach().cpu()
        print(f'prompt: {body.prompt[i]}')
        print(f'topk similarities: {sim}')
        for j, w in enumerate(wave):
            torchaudio.save(Path(results_folder) / Path(f'{body.prompt[i][:35]}_top_match_{j}.wav'), w, musiclm.neural_codec.sample_rate)

    return body

app.start(port=4000)