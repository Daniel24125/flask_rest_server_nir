import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, r2_score

    
def get_samples(data, split):
    Y_calib =data.sample(n=split)
    ix_cal= np.array(Y_calib.index)
    ix_val = np.delete(np.array(data.index), np.array(Y_calib.index))
    Y_val = np.delete(np.array(data), np.array(Y_calib.index))
    return (np.array(Y_calib),ix_cal,Y_val, ix_val)
  

data = pd.read_excel('calibration.xlsx')
wn = np.array(list(data)[2:])
processed_data =  savgol_filter(pd.DataFrame(data.iloc[:, 2:]).values, 5, polyorder = 2)
split_data = int(len(processed_data)*0.75)


# Get glucose concentrations
reference_data = pd.DataFrame(data["C"])
(Y_calib, ix_calib, Y_valid, ix_valid) = get_samples(reference_data,split_data)

X_calib = pd.DataFrame(processed_data).loc[ix_calib,:]
X_val = pd.DataFrame(processed_data).loc[ix_valid,:]

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

Y_pred = svr_rbf.fit(X_calib, Y_calib).predict(X_val)

score_p = r2_score(Y_valid, Y_pred)
mse_p = mean_squared_error(Y_valid, Y_pred)
sep = np.std(Y_pred-Y_valid)
rpd = np.std(Y_valid)/sep
bias = np.mean(Y_pred-Y_valid) 
# Plot regression and figures of merit
rangey = max(Y_valid) - min(Y_valid)
rangex = max(Y_pred) - min(Y_pred)
z = np.polyfit(Y_valid, Y_pred, 1)

with plt.style.context(('ggplot')):
  fig, ax = plt.subplots(figsize=(9, 5))
  ax.scatter(Y_pred, Y_valid, c='red', edgecolors='k')
  ax.plot(z[1]+z[0]*Y_valid, Y_valid, c='blue', linewidth=1)
  ax.plot(Y_valid, Y_valid, color='green', linewidth=1)
  plt.xlabel('Predicted')
  plt.ylabel('Measured')
  plt.title('Prediction')

  # Print the scores on the plot
  plt.text(min(Y_pred)+0.05*rangex, max(Y_valid)-0.1*rangey, 'R$^{2}=$ %5.3f'  % score_p)
  plt.text(min(Y_pred)+0.05*rangex, max(Y_valid)-0.15*rangey, 'MSE: %5.3f' % mse_p)
  plt.text(min(Y_pred)+0.05*rangex, max(Y_valid)-0.2*rangey, 'SEP: %5.3f' % sep)
  plt.text(min(Y_pred)+0.05*rangex, max(Y_valid)-0.25*rangey, 'RPD: %5.3f' % rpd)
  plt.text(min(Y_pred)+0.05*rangex, max(Y_valid)-0.3*rangey, 'Bias: %5.3f' %  bias)
  plt.show()   
