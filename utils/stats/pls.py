from sys import stdout
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error


def get_vip_scores(X, Y, model):
    
    # Calculate VIP scores
    p = X.shape[1]
    w = model.x_weights_
    tss = np.sum((Y - np.mean(Y, axis=0))**2)
    vip_scores = np.sqrt(p * np.dot(w**2, np.sum((w**2) * tss, axis=0)) / tss)
    
    print("VIP Scores:", vip_scores)    
    return vip_scores

def get_vip_scores2(X, Y, model, r2): 
    w = model.x_weights_
    print(w)

def vip(model):
  t = model.x_scores_
  w = model.x_weights_
  q = model.y_loadings_
  p, h = w.shape
  vips = np.zeros((p,))
  s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
  total_s = np.sum(s)
  for i in range(p):
      weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
      vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
  return vips

def get_num_components(X, Y):
    mse = []
    r2 = []
    component = np.arange(1, 4)

    for i in component:
        pls = PLSRegression(n_components=i)
        pls.fit(X, Y)
        Y_pred = pls.predict(X)
        mse_p = mean_squared_error(Y, Y_pred)
        mse.append(mse_p)
        r_score = pls.score(X,Y)
        factor = abs(r2[0] if len(r2) == 1 else np.squeeze(np.diff(r2)))
        r_score = r_score if i == 1 else r_score - factor
        # r2.append(r_score if i == 1 else r_score - r2[i-2] )
        r2.append(r_score)
        comp = 100*(i+1)/4

        # Trick to update status on the same line
        stdout.write("\r Calculating the number of components: %d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin+1)
    stdout.write("\n")
    return (msemin+1, r2, mse)    


def extract_model_info(X, Y): 
    num_comp, r2, mse = get_num_components(X,Y)
    print("Calculating PLS Coefficients")
    pls = PLSRegression(n_components=num_comp+1)
    pls.fit(X, Y)

    return (pls , r2, mse)
    


labels = ["V1", "V2", "V3"]

X = np.array([
    [1,2,10],
    [2,	5,	9],
    [3,	6,	8],
    [4,	23,	7],
    [5,	6,	6],
    [6,	8,	5],
    [7,	58,	4],
    [8,	2,	3],
    [9,	34,	2],
    [10, 8,	1]
])
Y = np.array([10,20,30,40,50,60,70,80,90,100])

model,  r2, mse = extract_model_info(X, Y)
vips = vip(model)

print(vips)
