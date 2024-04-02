import os
import time
import shutil
import tarfile
import pandas as pd

from tqdm import tqdm


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


split = "train"
root = "/ds/audio/IEMOCAP_LibriSpeech_knnvc"

df = pd.read_csv(f"{root}/{split}.csv")

os.makedirs(f"{root}/{split}", exist_ok=True)

t = time.time()
for i, row in tqdm(df.iterrows(), total=len(df)):
    audio_path = row["audio_path"]
    feat_path = row["feat_path"]

    audio_ext = audio_path.split(".")[-1]
    feat_ext = feat_path.split(".")[-1]

    shutil.copyfile(f"{audio_path}", f"{root}/{split}/{i}.{audio_ext}")
    shutil.copyfile(f"{feat_path}", f"{root}/{split}/{i}.{feat_ext}")

print(f"Done copying {split} files, took {time.time() - t} seconds.")

t = time.time()
make_tarfile(f"{root}/{split}.tar.gz", f"{root}/{split}")

shutil.rmtree(f"{root}/{split}")

print(f"Done making tar for split {split}, took {time.time() - t} seconds.")
