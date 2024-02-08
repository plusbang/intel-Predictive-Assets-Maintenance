import torch
import os
import numpy as np

# Create model object
model = None

def entry_point_function_name(data, context):
    """
    Works on data and context to create model object or process inference request.
    Following sample demonstrates how model object can be initialized for jit mode.
    Similarly you can do it for eager mode models.
    :param data: Input data for prediction
    :param context: context contains model server system properties
    :return: prediction output
    """
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

        model = torch.jit.load(model_pt_path)
    else:
        # infererence and return result
        data_input = np.load(data.get("data") or data.get("body"))
        return model.predict(data_input)
