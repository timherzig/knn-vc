import os
import sys
import argparse
import torch, torchaudio

from hifigan.utils import load_checkpoint
from hubconf import knn_vc


def infere(src_wav_path, ref_wav_paths, model, device):
    # Load the model
    knn_vc_model = knn_vc(remove_weight_norm=False)
    knn_vc_model = knn_vc_model.to(device)

    # # Load the checkpoint
    checkpoint = load_checkpoint(model, device)["generator"]
    knn_vc_model.hifigan.load_state_dict(checkpoint)

    # # Set the model to evaluation mode and remove weight normalization
    knn_vc_model.hifigan.eval()
    knn_vc_model.hifigan.remove_weight_norm()

    query_seq = knn_vc_model.get_features(src_wav_path)
    matching_set = knn_vc_model.get_matching_set(ref_wav_paths)

    out_wav = knn_vc_model.match(query_seq, matching_set, topk=4).unsqueeze(0)

    os.makedirs("inference", exist_ok=True)
    torchaudio.save("inference/out.wav", out_wav, 16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_wav_path",
        type=str,
        default="/ds/audio/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac",
    )
    parser.add_argument(
        "--ref_wav_paths",
        type=str,
        nargs="*",
        default=[
            "/ds/audio/LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac",
            "/ds/audio/LibriSpeech/test-clean/1089/134686/1089-134686-0002.flac",
            "/ds/audio/LibriSpeech/test-clean/1089/134686/1089-134686-0003.flac",
        ],
    )
    parser.add_argument("--model", type=str, default="bshall/knn-vc")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    infere(args.src_wav_path, args.ref_wav_paths, args.model, device)
