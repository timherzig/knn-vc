import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataframe
df = pd.read_csv("/ds/audio/IEMOCAP_LibriSpeech_knnvc/metadata.csv")

iemoc_df = df[df["audio_path"].str.contains("IEMOCAP_full_release")]
libri_df = df[~df["audio_path"].str.contains("IEMOCAP_full_release")]

print(iemoc_df.head())
print(libri_df.head())

iemoc_train_df, iemoc_val_df = train_test_split(
    iemoc_df, test_size=0.2, random_state=42
)

libri_train_df = libri_df[libri_df["audio_path"].str.contains("train")]
libri_val_df = libri_df[~libri_df["audio_path"].str.contains("train")]

# print lengths of each dataframe
print(
    f"iemoc_train_df: {len(iemoc_train_df)}\niemoc_val_df: {len(iemoc_val_df)}\nlibri_train_df: {len(libri_train_df)}\nlibri_val_df: {len(libri_val_df)}"
)

train_df = pd.concat([iemoc_train_df, libri_train_df], ignore_index=True)
val_df = pd.concat([iemoc_val_df, libri_val_df], ignore_index=True)

# print lengths of each dataframe
print(f"train_df: {len(train_df)}\nval_df: {len(val_df)}")

# Save the dataframes
train_df.to_csv("/ds/audio/IEMOCAP_LibriSpeech_knnvc/train.csv", index=False)
val_df.to_csv("/ds/audio/IEMOCAP_LibriSpeech_knnvc/val.csv", index=False)
