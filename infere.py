import os
import argparse
import torch, torchaudio

from hifigan.models import Generator
from hifigan.utils import load_checkpoint


def infere(src_wav_path, ref_wav_paths, model, device):
    # Load the model
    knn_vc = torch.hub.load(
        "bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True
    )
    knn_vc = knn_vc.to(device)

    # Load the checkpoint
    checkpoint = load_checkpoint(model, device)
    checkpoint["generator"] = {
        k.replace("_v", ""): v for k, v in checkpoint["generator"].items()
    }
    g_keys = [i for i in list(checkpoint["generator"].keys()) if "_g" in i]
    for k in g_keys:
        checkpoint["generator"].pop(k)

    knn_vc.hifigan.load_state_dict(checkpoint["generator"])

    query_seq = knn_vc.get_features(src_wav_path)
    matching_set = knn_vc.get_matching_set(ref_wav_paths)

    out_wav = knn_vc.match(query_seq, matching_set, topk=4).unsqueeze(0)
    print(out_wav.shape)

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
