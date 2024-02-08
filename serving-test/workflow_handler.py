import numpy as np

def pre_processing(data, context):
    if data is None:
        return data
    data_input = np.load(data.get("data") or data.get("body"))
    return data_input
