import numpy as np
import scipy as sp
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

class featureSelection:

    def __init__(self, huber_slope=100, huber_cutoff=1e-5, opt_step=1e3, tol=0, R_start=None, num_features=3):
        self.l=huber_slope
        self.delta=huber_cutoff
        self.R=R_start
        self.opt_step=int(opt_step)
        self.tol=tol
        self.num_features=num_features
        if R_start != None:
            self.num_features=R_start.shape[1]

    ########################################################################
    ######################## Callable functions ############################
    ########################################################################

    def fit(self,A,v):
        self.n, self.d=A.shape
        self.A=A
        self.v=v

        R0=self.R
        if R0 == None:
            R0=self.start(A,v)

        t=time.time()
        self.R, self.x, self.vals = self.gradient_descent(A, v, R0, self.opt_step, self.tol)
        t=(time.time()-t)
        print(f"Fit took {t} seconds.")

        return self

    def predict(self, newA):
        return newA@(self.R@self.x)

    ########################################################################
    ##################### Postprocessing functions #########################
    ########################################################################

    # rounds R values absolutely below d to zero
    def threshold(self, R, t): 
        #print(f"{np.sum(np.abs(R)<t)} entries of R are close to 0 (below {t}) before applying high pass filter.")
        return R-(np.abs(R)<=t)*R

    # prints number of non zero coordinates of R for each column, rank.
    def print_info(self):
        R=self.R
        d=self.d
        print(f"Dimension is {d}.")
        R=R.T
        k=1
        for col in R:
            print(f"Column {k} has {np.sum(col!=0)} non-zero values")
            k+=1

        print(f"Matrix R has rank {np.linalg.matrix_rank(R)}")
        
        print(f"The distance from the feature space to the labels is {self.dist(self.A,self.v,self.R)}.")

        return 0

    # plot values and save figure.
    def plot(self, save=False, maxy=None):
        if maxy==None: maxy=np.max([2,self.l**2])
        vals=self.vals
        plt.plot(vals)
        plt.axis([0,len(vals),0,maxy])
        if save: plt.savefig("values")
        plt.show()
        return 0

    def biplot(self, save=False):
        thresholds, distances, nonzero=self.thresholds, self.distances, self.nonzero

        plt.figure(1)

        ax1=plt.subplot(211)
        ax1.plot(thresholds, distances)
        ax1.title.set_text("d(AR,v)")

        ax2=plt.subplot(212)
        ax2.plot(thresholds, nonzero)
        ax2.set_ybound(upper=20)
        ax2.title.set_text("Number of nonzero elements")

        plt.show()
        if save: plt.savefig("biplot")
        return 0
    # increases the threshold applied to R little by little,
    # and recording d(AR, v) and the number of nonzero elements,
    # until the whole matrix is reduced to zero.
    def rolling_threshold(self):
        print("Applying rolling threshold:")
        print("Small values of R are set to 0 and a new distance in computed.")
        print("R is overwitten to be the one minimizing the distance to the labels.")

        A=self.A
        R=self.R
        v=self.v

        thresholds=[]
        distances=[]
        nonzero=[]
        matrices=[]
        d=len(R)
        t=0
        while np.any(R!=0):
            matrices.append(R)

            thresholds.append(t)
           
            nonzero.append(np.sum(R!=0))
            
            distances.append(self.dist(A,v,R))
        
            t=np.min(np.abs(R[np.nonzero(R)]))
            R=self.threshold(R,t)

        self.thresholds=thresholds
        self.distances=distances
        self.nonzero=nonzero
        i=np.argmin(distances)
        self.R=matrices[i]
        self.x=self.optx(A,v,self.R)
        return matrices[i]

    ########################################################################
    ######################## Starting functions ############################
    ########################################################################

    # Defines the starting R at the beginning of the optimization.
    # Many choices are available. R=0 and R=rand +/-1 are sensible choices.
    # TODO: try random bernoulli +-1 matrix many times
    def start(self,A,v):
        d=self.d

        # all ones R0
        #R0=np.ones((d,3))
        
        # all zeros R0
        #R0=np.zeros((d,self.num_features))
        
        # least squares solution giving a 1-dim best approximation
        #R0 = x[:,np.newaxis]*np.ones(self.num_features)
        x=np.linalg.lstsq(A,v, rcond=-1)[0]
        R0=np.zeros((d,self.num_features))
        R0[:,0]=x
        R0=self.retract(R0, 0, 0)
        return R0

    ########################################################################
    ########################### main functions #############################
    ########################################################################

    # returns x = argmin(||ARx-v||)
    # this is very fast since AR is n-by-3, so lstsq just inverts a 3-by-3 matrix and does matrix multiplication
    def optx(self,A,v,R):
        B=A@R
        x=np.linalg.lstsq(B,v, rcond=-1)[0]
        if np.linalg.norm(x)==0:
            print("WARNING: the matrix R is zero. The objective value is discontinuous.")
            print("Setting x=1 to prevent immobilization.")
            # this choice gives an initial direction of optimization if R=0,
            # (R,x) = (0,0) is a stationnary point around which the objective value is discontinuous
            return np.ones(self.num_features)
        return x

    # rescales columns of R w.r.t. L^infty norm.
    # this does not change the image of AR, but it helps guide smaller coefficients to 0.
    def rescale(self, R):
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
    def huber(self, R, l, d):
        return (l*np.abs(R) + l*d**2/2 - l*d)*(np.abs(R)>d) + (R**2 *l/2)*(np.abs(R)<=d)

    # huber gradient (simple derivative of huber loss)
    def grad_huber(self, R, l, d):
        return (l*np.sign(R))*(np.abs(R)>d) + (l*R)*(np.abs(R)<=d)

    # objective value f to minimize
    # f = d(AR,v) + huber(R)
    def f(self,A,v,R,x):
        f=np.linalg.norm(A@R@x-v)**2
        f+=np.sum(self.huber(R,self.l,self.delta))
        return f

    # returns d(AR,v) = min_x || ARx - v ||
    def dist(self,A,v,R):
        x=self.optx(A,v,R)
        return np.linalg.norm(A@R@x-v)

    # euclidian gradient of f w.r.t. R
    def eu_gradient(self,A,v,R,x):
        k=self.num_features
        u=(A@(R@x)-v).T@A
        G=u[:,np.newaxis]*np.ones(k)
        G=2*G*x

        m=len(R)
        GH=self.grad_huber(R, self.l, self.delta)
        G+=GH
        return G
    
    # projects M onto tangent space at R
    def orth_proj(self, R, M):
        return 1/2 * (M - R@M.T@R)

    def riem_gradient(self,A,v,R,x):
        G = self.orth_proj(R, self.eu_gradient(A,v,R,x))
        return self.orth_proj(R, self.eu_gradient(A,v,R,x))

    def retract(self,R,G,step):
        Q=np.linalg.qr(R+step*G)[0]
        return Q

    # Line search step.
    # This is the backtracking line search by Armijo (1966), see wiki.
    # The constant 0.1 is a free parameter in this method, and I chose it randomly.
    def step(self, A,v,R,x,G,laststep):
        if laststep>-1e-10:
            s=-1e-7
        else:
            s=2*laststep
        t=(0.1)*np.linalg.norm(G)**2
        k=0
        while self.f(A,v,R,x)-self.f(A,v,R+s*G,x) < -s*t:
            #print(k)
            k+=1
            s /= 2
            if s > -1e-15:
                    return s
        return s

    # Stop conditions in gradient descent.
    def check_stop(self,G,vals,k,opt_steps,tol):
        maxsteps=opt_steps
        print(f"Step: {k} / {maxsteps}")
        if(np.linalg.norm(G) < tol):
            print("Stop due to gradient tolerance.")
            return True
        # outcommented for now
        #if(np.abs(vals[k] - vals[k+1]) < tol):
        #    print("Stop due to value tolerance.")
        #    return True
        return k>=maxsteps 

    # Main optimizing function.
    # The visited matrices and points (R,x) are kept in memory,
    # but only that of the previous step are used.
    # A and v are data points and target vector,
    # R0 is the starting point for R.
    def gradient_descent(self, A, v, R0, opt_steps, tol):
        n, d=A.shape
        assert len(v)==n
        r1, r2=R0.shape
        assert r2==self.num_features and r1==d
        #R0=self.rescale(R0) # not needed

        x0=self.optx(A,v,R0)
        M=[R0]
        p=[x0]
        vals=[self.f(A,v,R0,x0)]
        k=0
        s=-1e-2 # starting step, important choice
        stop=False
        while not stop:
            x=p[k]
            R=M[k]
            
            #G=self.eu_gradient(A,v,R,x)
            G=self.riem_gradient(A,v,R,x)

            s=self.step(A,v,R,x,G,s)
            #print(s)
            assert s<=0, "step size is positive?"
            
            # scales to entries between -1/10 and 1/10,
            # this should help the convergence of entries to 0.
            # The number should be played with to truly understand its effect.
            # This isn't very rigorous because dividing by the norm is a retraction for L² but not L^infty, I think.
            # Also, if we retract, the gradient should also be projected to get the Riemaniann gradient on the manifold.
            #newR=self.rescale(R+s*G)/1
            #newR=self.threshold(R,self.tol)
            newR=self.retract(R,G,s)

            newx=self.optx(A,v,newR)

            M.append(newR)
            p.append(newx)
            vals.append(self.f(A,v,newR,newx))

            # because of the rescaling, it's not uncommon for val to increase
            #if vals[k+1]>vals[k]:
            #    print("aïe")
            
            stop=self.check_stop(G,vals,k,opt_steps,tol)
            
            k=k+1
        R=M[k-1]
        x=p[k]
        return R, x, np.array(vals)

    ##################################################################
    ######################### Unused stuff ###########################
    ##################################################################

    # unused
    # ideally we would use an optimal step,
    # but the Huber loss makes it hard to compute exactly.
    # 
    def oldstep(self, A,v,R,x,G,s):
        l=self.l
        delta=self.delta
        alpha=np.linalg.norm(A@G@x)**2 
        beta=2*(A@G@x).T@(A@R@x-v) + l*np.sum(G)
        if alpha!=0:
            return -beta/alpha/2
        else:
            return -1e-8

    # unused
    # replaces f(A,v,R,x)-f(A,v,R+s*G,x) and is more efficient
    # the formula is wrong though lol
    def fdiff(self, A,v,R,x,G,s):
        l=self.l
        r=s**2 * np.linalg.norm(A@(G@x))**2 + 2*s*(A@(G@x)).T@(A@(R@x)-v) + l*s*np.sum(G)
        print(-r, f(A,v,R,x)-f(A,v,R+s*G,x))
        print(np.all(R+s*G>=0))
        return -r
