import os
import scipy.io
import pandas as pd
from datetime import datetime

# Path to the .mat metadata file
mat_path = "imdb_crop/imdb.mat"
image_root = "imdb_crop"

# Load the .mat file
meta = scipy.io.loadmat(mat_path)
meta = meta["imdb"][0, 0]

photo_taken = meta["photo_taken"][0]
full_path = meta["full_path"][0]
dob = meta["dob"][0]

data = []

for i in range(len(full_path)):
    try:
        path = full_path[i][0]
        birthdate = datetime.fromordinal(int(dob[i])) if dob[i] > 0 else None
        taken = photo_taken[i]

        if birthdate:
            age = taken - birthdate.year
            image_path = os.path.join(image_root, path)

            if os.path.isfile(image_path) and 0 < age < 100:
                data.append([image_path, age])
    except Exception:
        continue

# Create DataFrame and save to CSV
df = pd.DataFrame(data, columns=["path", "age"])
df.to_csv("processed.csv", index=False)
print("processed.csv generated successfully.")
