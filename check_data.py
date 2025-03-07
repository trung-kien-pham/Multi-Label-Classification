import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
import os


path = "Variant-b(MultiLabel Classification)/Multi-Label dataset - with augmented.csv"
df = pd.read_csv(path)
df.head()