import os 
import numpy as np
import pandas as pd
from bigdl.chronos.data import TSDataset
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.pytorch.loss import AsymWeightLoss
from bigdl.chronos.detector.anomaly import ThresholdDetector
import torch

df = pd.read_csv("predictive-maintenance-dataset.csv")
lookback = 120
horizon = 1

df["time_step"] = pd.date_range(start='2023-01-01 16:30:00', end='2023-01-01 23:30:00', periods=len(df))
tsdata_train, tsdata_val, tsdata_test = TSDataset.from_pandas(df, dt_col="time_step", target_col="vibration",
                                                              extra_feature_col=["revolutions","humidity","x1","x2","x3","x4","x5"],
                                                              with_split=True, test_ratio=0.1)
standard_scaler = StandardScaler()

for tsdata in [tsdata_train, tsdata_test]:
    tsdata.scale(standard_scaler, fit=(tsdata is tsdata_train))\
          .roll(lookback=lookback, horizon=horizon)

x_train, y_train = tsdata_train.to_numpy()
x_test, y_test = tsdata_test.to_numpy()

forecaster = TCNForecaster(past_seq_len=lookback,
                           future_seq_len=horizon,
                           input_feature_num=8,
                           output_feature_num=1,
                           normalization=False,
                           kernel_size=5,
                           num_channels=[16]*8,
                           loss=AsymWeightLoss(underestimation_penalty=10))

print('Start training ...')
forecaster.num_processes = 1
# forecaster.fit(data=tsdata_train, epochs=5)
forecaster.fit(data=tsdata_train, epochs=0) # To save time for debugging
print('Training completed')
forecaster.save('./checkpoints/forecaster_ckpt.pth')

y_pred_train = forecaster.predict(x_train)
y_pred_train_unscale = tsdata_train.unscale_numpy(y_pred_train)
y_train_unscale = tsdata_train.unscale_numpy(y_train)

thd = ThresholdDetector()
vibration_th = 85
thd.set_params(trend_threshold=(0, vibration_th)) # if vibration>85, we think there may exist potential elevator failure
thd.fit(y_train_unscale, y_pred_train_unscale)

torch.save(thd, './checkpoints/detector_ckpt.pt')

y_test_unscale = tsdata_test.unscale_numpy(y_test)
np.save('data.npy', y_test_unscale)
