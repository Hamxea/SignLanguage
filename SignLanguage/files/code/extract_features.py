"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""











import numpy as np
import os.path
from data import DataSet
from extractor import Extractor
from tqdm import tqdm
from PIL import Image

# Set defaults.
seq_length = 40
class_limit = 8  # integer, number of classes to extract

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit)

# get the model.
model = Extractor()

# Loop through data.
pbar = tqdm(total=len(data.data))
for video in data.data:
    print(video[2])
    # Get the path to the sequence for this video.
    path = './data/sequences-ucf/' + video[2] + '-' + str(seq_length) + \
        '-features.txt'
    print("path = ", path)
    # Check if we already have it.
    if os.path.isfile(path):
        pbar.update(1)
        continue
    print("1")
    # Get the frames for this video.
    frames = data.get_frames_for_sample(video)
    print("2")
    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)
    print("3")
    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in frames:
        print("4")
        print(image)
        print("5")
        features = model.extract(image)
        print("6")
        sequence.append(features)

    # Save the sequence.
    np.savetxt(path, sequence)

    pbar.update(1)

pbar.close()
