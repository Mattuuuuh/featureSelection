import numpy as np
import scipy as sp
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

########################################################################
######################### Unused functions #############################
########################################################################

# unused
def best_line_appx(A,v):
    P=A.T@v
    return np.argpartition(P,-3)[-3:]

# starting from best line approximate R
"""
R0=np.zeros((d,3))
chart=best_line_appx(A,v)
R0[chart]=np.eye(3)
"""

# unused
# brut force returns
#chart, val=np.array([17,72,73]), 0.6290979
def bruteforce(A,v):
    n, d=A.shape
    assert len(v)==n
    charts=[[i,j,k] for i in range(d) for j in range(i+1,d) for k in range(j+1,d)]
    vals=[]
    k=0
    for chart in charts:
        print(f"{k} / {len(charts)}")
        k+=1
        diag=np.zeros(d)
        diag[chart]=1
        R=np.diag(diag)
        val=f(A,v,R,optx(A,v,R))
        vals.append(val)
        print(val)
    i=np.argmin(vals)
    
    return np.array(charts[i]), vals[i]

# brute force
"""
t=time.time()

chart, val=bruteforce(A,v)

t=(time.time()-t)

print(f"Took {t} seconds.")
print(chart, val)
print(df.keys()[chart+1])
"""

# unused because x can be efficiently computed using optx()
def eu_x_gradient(A,v,R,x):
    M=A@R
    print(2*M.T@(M@x-v))
    return 2*M.T@(M@x-v)

# unused
# ideally we would use an optimal step,
# but the Huber loss makes it hard to compute exactly.
# 
def oldstep(A,v,R,x,G,s):
    global l
    global delta
    alpha=np.linalg.norm(A@G@x)**2 
    beta=2*(A@G@x).T@(A@R@x-v) + l*np.sum(G)
    if alpha!=0:
        return -beta/alpha/2
    else:
        return -1e-8

# unused
# replaces f(A,v,R,x)-f(A,v,R+s*G,x) and is more efficient
# the formula is wrong though lol
def fdiff(A,v,R,x,G,s):
    global l
    r=s**2 * np.linalg.norm(A@(G@x))**2 + 2*s*(A@(G@x)).T@(A@(R@x)-v) + l*s*np.sum(G)
    print(-r, f(A,v,R,x)-f(A,v,R+s*G,x))
    print(np.all(R+s*G>=0))
    return -r

# unused
# need to differentiate w.r.t. A instead of columns of A
def gradA(A,v,R):
    x=optx(A,v,R)
    dA=R@x
    dA=dA/np.linalg.norm(dA, ord=np.inf)
    print(f"Derivative w.r.t. feature vector: \n {dA}")
    return dA

# unused
def oldhuber(x, l, d):
    if np.abs(x) > d:
        return l*np.abs(x) + l*d**2/2 -l*d
    else:
        return x**2 *l/2 

# unused
def oldgrad_huber(x, l, d):
    if np.abs(x) > d:
        return l*np.sign(x)
    else:
        return l*x

########################################################################
##################### Postprocessing functions #########################
########################################################################

# rounds R values absolutely below d to zero
def threshold(R, d): 
    #print(f"{np.sum(np.abs(R)<d)} entries of R are close to 0 (below {d}) before applying high pass filter.")
    return R-(np.abs(R)<d)*R

# prints number of non zero coordinates of R for each column, rank.
def print_info(R):
    d=len(R)
    print(f"Dimension is {d}.")
    R=R.T
    k=1
    for col in R:
        print(f"Column {k} has {np.sum(col!=0)} non-zero values")
        k+=1

    print(f"Matrix R has rank {np.linalg.matrix_rank(R)}")
    return 0

# plot values
def plot(vals, maxy):
    plt.plot(vals)
    plt.axis([0,len(vals),0,maxy])
    plt.savefig("values")
    plt.show()
    return 0

########################################################################
######################## Starting functions ############################
########################################################################

# Defines the starting R at the beginning of the optimization.
# Many choices are available. R=0 and R=rand +/-1 are sensible choices.
# TODO: try random bernoulli +-1 matrix many times
def start(d):
    
    # saved R0
    #R0=np.load('R3.npy')
    
    # all ones R0
    #R0=np.ones((d,3))
    
    # all zeros R0
    R0=np.zeros((d,3))

    # brute-force solution R0
    #chart=np.array([17,72,73])
    #R0[chart,:]=np.eye(3)
    return R0

def read(filename):
    df=pd.read_csv(filename)
    v=np.array(df["target"])
    A=np.array(df.take(np.arange(1,99), axis=1))
    return A, v

########################################################################
########################### main functions #############################
########################################################################

# returns x = argmin(||ARx-v||)
# this is very fast since AR is n-by-3, so lstsq just inverts a 3-by-3 matrix and does matrix multiplication
def optx(A,v,R):
    B=A@R
    x=np.linalg.lstsq(B,v, rcond=-1)[0]
    if np.linalg.norm(x)==0:
        print("WARNING: the matrix R is zero. The objective value is discontinuous at this points.")
        print("Setting x=1 to prevent immobilization.")
        # this choice gives an initial direction of optimization if R=0,
        # (R,x) = (0,0) is a stationnary point around which the objective value is discontinuous
        return np.array([1,1,1])
    return x

# rescales columns of R w.r.t. L^infty norm.
# this does not change the image of AR, but it helps guide smaller coefficients to 0.
def rescale(R):
    R=R.T
    for col in R:
        n=np.linalg.norm(col, ord=np.inf)
        if n!=0:
            col /= n
    return R.T

# huber loss (see wiki).
# l is the weight on the L¹ norm in the loss function,
# d is the threshold at which the L¹ switches to a quadratic (differentiable) function around 0.
# returns a matrix to be summed to get the loss.
def huber(R, l, d):
    return (l*np.abs(R) + l*d**2/2 / l*d)*(np.abs(R)>d) + (R**2 *l/2)*(np.abs(R)<=d)

# huber gradient (simple derivative of huber loss)
def grad_huber(R, l, d):
    return (l*np.sign(R))*(np.abs(R)>d) + (l*R)*(np.abs(R)<=d)

# objective value f to minimize
# f = d(AR,v) + huber(R)
def f(A,v,R,x):
    f=np.linalg.norm(A@R@x-v)**2
    
    global l
    global delta
    f+=np.sum(huber(R,l,delta))
    return f

# returns d(AR,v) = min_x || ARx - v ||
def dist(A,v,R):
    x=optx(A,v,R)
    return np.linalg.norm(A@R@x-v)**2

# euclidian gradient of f w.r.t. R
def eu_gradient(A,v,R,x):
    n=3
    u=(A@(R@x)-v).T@A
    G=u[:,np.newaxis]*np.ones(n)
    G=2*G*x

    m=len(R)
    global l
    global delta
    G+=grad_huber(R, l, delta)
    return G

# Line search step.
# This is the backtracking line search by Armijo (1966), see wiki.
# The constant 0.1 is a free parameter in this method, and I chose it randomly.
def step(A,v,R,x,G,laststep):
    if laststep>-1e-10:
        s=-1e-7
    else:
        s=2*laststep
    t=(0.1)*np.linalg.norm(G)**2
    k=0
    while f(A,v,R,x)-f(A,v,R+s*G,x) < -s*t:
        #print(k)
        k+=1
        s /= 2
        if s > -1e-15:
                return s
    return s

# Stop conditions in gradient descent.
def check_stop(vals,k,opt_steps,tol):
    maxsteps=opt_steps
    print(f"{k} / {maxsteps}")
    if(np.abs(vals[k]-vals[k+1]) < tol):
        print("Stop due to tolerance.")
        return True
    return k>=maxsteps 

# Main optimizing function.
# The visited matrices and points (R,x) are kept in memory,
# but only that of the previous step are used.
# A and v are data points and target vector,
# R0 is the starting point for R.
def main(A, v, R0, opt_steps, tol):
    n, d=A.shape
    assert len(v)==n
    r1, r2=R0.shape
    assert r2==3 and r1==d
    R0=rescale(R0)/2 # isgood

    x0=optx(A,v,R0)
    M=[R0]
    p=[x0]
    vals=[f(A,v,R0,x0)]
    k=0
    s=-1e-2 # starting step
    stop=False
    while not stop:
        x=p[k]
        R=M[k]
        
        G=eu_gradient(A,v,R,x)

        s=step(A,v,R,x,G,s)
        print(s)
        assert s<=0, "step size is positive?"
        
        newR=rescale(R+s*G)*(1e-1) # this is a great idea

        newx=optx(A,v,newR)

        M.append(newR)
        p.append(newx)
        vals.append(f(A,v,newR,newx))
        
        if vals[k+1]>vals[k]:
            print("aïe")
        stop=check_stop(vals,k,opt_steps,tol)
        
        k=k+1
    R=M[k]
    x=p[k]
    return R, x, np.array(vals)

########################################################################
######################### script to optimize ###########################
########################################################################

# reading data
A, v= read("Regression.csv")
n, d=A.shape

# starting R
R0=start(d)

# global constants to manipulate
l=100
delta=1e-5

# optimization
t=time.time()
R,x,vals=main(A=A,v=v,R0=R0,tol=0e-15,opt_steps=1e3)
t=(time.time()-t)
print(f"Took {t} seconds.")

# save R before high pass filter
np.save("R3", R)

# plot
plot(vals, maxy=2*l)

# high pass filter and rescale for R
print(f"Distance to v before high pass filter on R: {dist(A,v,R)}.")
R = threshold(R, 1e-3)
R = rescale(R)

# print stuff yo

#print(R)

print(f"Objective values: {vals}")

print(f"Distance to v after high pass filter on R: {dist(A,v,R)}.")

print_info(R)


