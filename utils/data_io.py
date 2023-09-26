import pandas as pd

def load_csv_data(file_path, N):
    data = pd.read_csv(file_path).to_numpy(dtype=int)
    label = data[:N, 0]
    data = data[:N, 1:] 
    return label, data