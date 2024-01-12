import numpy as np
import cv2
import pandas as pd
from collections import Counter

training_data = np.load("./final_training_data.npy",allow_pickle=True)

df = pd.DataFrame(training_data)

print(df.head())
print(Counter(df[1].apply(str)))