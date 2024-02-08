import torch
import os
import numpy as np

# Create model object
model = None

def entry_point_function_name(data, context):
    global model

    if not data:
        manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        device = torch.device("cpu")

        # Read model serialize/pt file
        serialized_file = manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        model = torch.load(model_pt_path)
    else:
        # infererence and return result
        pred_data = data.get("forecaster")
        test_anomaly_indexes = model.anomaly_indexes(y_pred=pred_data)
        anomalies = {}
        for key, value in test_anomaly_indexes.items():
            anomalies.append(f'{key}: {value}')
        return anomalies
