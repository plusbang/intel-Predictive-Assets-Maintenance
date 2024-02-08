# Serving Experiment

### Setup Env
```bash
conda create -n binbin_serving python=3.9 setuptools=58.0.4
conda activate binbin_serving
pip install bigdl-chronos[pytorch]
pip install torch-model-archiver
pip install torch-workflow-archiver
pip install torchserve
```

1) Create and train `TCNForecaster` based on `predictive-maintenance-dataset.csv`, then save the pth file with path "./checkpoints/forecaster_ckpt.pth". Meanwhile, initiate and fit `ThresholdDetector`, then save the pth file with path "./checkpoints/detector_ckpt.pt".
```bash
python generate_ckpt.py
```
Meanwhile, the input data for predicting and detecting anomalies is saved as `data.npy`.

2) Archine model and workflow
```bash
torch-model-archiver -f --model-name forecaster --version 1.1 --serialized-file ./checkpoints/forecaster_ckpt.pth --handler ./forecaster_handler:entry_point_function_name --export-path model_store

torch-model-archiver -f --model-name detector --version 1.1 --serialized-file ./checkpoints/detector_ckpt.pt --handler ./detector_handler:entry_point_function_name --export-path model_store

torch-workflow-archiver -f --workflow-name predictive_assets_maintenance --spec-file workflow.yaml --handler workflow_handler.py --export-path wf_store/
```

3) Serve the workflow
```bash
torchserve --start --model-store model_store/ --workflow-store wf_store/ --ncs
```

The input data is saved in `data.npy`.
```bash
curl -X POST "http://localhost:8080/workflows?url=predictive_assets_maintenance.war"
curl http://localhost:8080/test/predictive_assets_maintenance -T data.npy
```

```bash
torchserve --stop
```

### Issues
1. java version (https://github.com/pytorch/serve/issues/473)
```bash
(binbin_serving) icx@icx-5:/disk0/binbin/intel-Predictive-Assets-Maintenance/serving-test$ torchserve --start --model-store model_store/ --workflow-store wf_store/ --ncs
Removing orphan pid file.
(binbin_serving) icx@icx-5:/disk0/binbin/intel-Predictive-Assets-Maintenance/serving-test$ java.lang.NoSuchMethodError: java.nio.file.Files.readString(Ljava/nio/file/Path;)Ljava/lang/String;
        at org.pytorch.serve.util.ConfigManager.readFile(ConfigManager.java:253)
        at org.pytorch.serve.util.ConfigManager.<init>(ConfigManager.java:150)
        at org.pytorch.serve.util.ConfigManager.init(ConfigManager.java:311)
        at org.pytorch.serve.ModelServer.main(ModelServer.java:83)
```
**Solution:**
sudo apt-get install openjdk-11-jdk

