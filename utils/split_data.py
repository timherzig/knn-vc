import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataframe
df = pd.read_csv("/ds/audio/IEMOCAP_knnvc/metadata.csv")

df["audio_path"] = df["audio_path"].apply(lambda x: x.replace("/ds/audio/IEMOCAP/", ""))
df["feat_path"] = df["feat_path"].apply(
    lambda x: x.replace("/ds/audio/IEMOCAP_knnvc/", "")
)

# Split the dataframe into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the train and validation sets into separate dataframes
train_df.to_csv("/ds/audio/IEMOCAP_knnvc/train_set.csv", index=False)
val_df.to_csv("/ds/audio/IEMOCAP_knnvc/validation_set.csv", index=False)
