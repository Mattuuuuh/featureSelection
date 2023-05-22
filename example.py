import numpy as np
import pandas as pd
from feature import featureSelection

# reading data
def read(filename):
    df=pd.read_csv(filename)
    v=np.array(df["target"])
    A=np.array(df.take(np.arange(1,99), axis=1))
    return A, v

A, v= read("Regression.csv")

# initializing featureSelection

# increase huber_slope to add weight to the loss.
# huber_slope gives the tradeoff between good approximation (low slope) and small number of nonzero coefficients (high slope)
# values to try: 1e-5, 1, 5, 10.

# increase num_features to get better approximation, but slower runtime.
# a higher number of features sometimes also converges quicker, however.
# values to try: 3, 5, 10.
model=featureSelection(huber_slope=1e-5, huber_cutoff=1e-5, opt_step=4e4, tol=1e-10, R_start=None, num_features=3)

# fit with feature matrix A, labels v
model.fit(A, v)

# plot values of loss that was minimized
model.plot()

print("############################################")
# High pass filters: removes coefficients of R iteratively,
# computing the loss function without the regularization (huber loss).
# redefines R to be that minimizing the distance
model.rolling_threshold()

# Plot distances and number of nonzero elements of R w.r.t. thresholds;
#model.biplot()

print("############################################")
# print info on matrix R, including number of non zero entries per column
model.print_info()

print("############################################")
# makes prediction based off of new data.
# here we just predict from the initial data
y = model.predict(A)

print(f"Predicted {y} \n Compared to input labels {v} \n Distance between the two: {np.linalg.norm(y- v)}")

# extract and save parameters if needed
R=model.R
x=model.x
np.save("R", R)
np.save("x", x)

