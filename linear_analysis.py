#!/usr/bin/python
import numpy as np
import sys
import scipy.linalg as la
import scipy.optimize as op
import matplotlib.pyplot as plt

def center(raw):

    return raw - np.mean(raw, axis=0)

def slice(dependent, raw_data):

    try:
        #find column index inputted by user and convert it to an integer
        index = int(dependent)

    except NameError:
        print "Cannot convert input to integer"
        raise
   
    A = np.array([False]*raw_data.shape[1])
    A[index] = True
    #return split data
    return W[:, -A], W[:, A]
    
def fit(X, Y):

    return la.solve(X.T.dot(X), X.T.dot(Y))

def RSS(X, Y, M):

    XM = X.dot(M)
    XM.shape = Y.shape
    return ((Y-XM).T).dot(Y-XM)[0][0]

def fractionOfVariance(X, Y, M):

    return ((Y.T).dot(Y) - RSS(X, Y, M))/((Y.T).dot(Y))

def PCA(N, W):

    return la.eigh((W.T.dot(W))/(N-1))    

def F(X, Y, M, t):

    return (0.5*RSS(X, Y, M) + t*la.norm(M, 1))

def lasso_step(X, Y, t):

    return op.minimize((lambda M: F(X, Y, M, t)), np.zeros(X.shape[1])).x

def lasso(X, Y):
    
    retval = []
    t_val = []
    max_t = int(find_max_t(X, Y))  #int(max(abs(X.T.dot(Y))))*5/12 

    min_t = 0
    iterations = 10
    step = (max_t - min_t)/iterations

    for t in range(min_t, max_t, step):
        retval.append(lasso_step(X, Y, t))
        t_val.append(t)

    return retval, t_val
    
def find_max_t(X, Y, bifurcations=10):

    # Find a reasonable value of T_max for the lasso loop, using a simple
    #binary-bracketing method.  The resolution is 2**(-bifurcations), so the def    #ault will be accurate to 1/1024. This assumes that your lasso_step takes th    #e form Mt =  lasso_step(X,Y,t) 

    #initial bracket to search 
    T_left=0.0
    T_right=np.max(np.abs(X.T.dot(Y)))  ## this is PLENTY big to give M near 0.

    for i in range(bifurcations):
        T_mid = (T_right + T_left)/2.0
        try:
            M_mid = lasso_step(X,Y,T_mid)
            if np.allclose(M_mid, 0.0):
                T_right = T_mid
            else:
                T_left = T_mid ### NOTE!  You may as well save M_mid to your trajectory in this case!
        except:
            print "lasso_step failed at t={}, so it must be too big. Treating as Mt==0".format(T_mid)
            T_right = T_mid
    print "T_right = ", T_right
    return T_right

def pic(t, m):

    plt.plot(t, m)
    plt.savefig("diabetesplot.png")


if __name__ == '__main__':

    try:
        data_file = open(sys.argv[1])

    except IndexError:
        print "\n\nThere was a syntax error in your input. Please try agan and use this syntax:\n ./linear_analysis.py nameOfDataFile modelType dependentResponseColumn tVal\n\nNote: modelType capabilities: pca, fit, lasso\ndependentResponseColumn should be an integer that indicates which column (indexed from 0) to be used as the dependent/response variable and is only applicable to fit and lasso\ntVal is applicable only to lasso - returns linear model given by lasso at the specified tVal\n"
        sys.exit(0)

    try:
        raw_data = np.loadtxt(data_file)
    except IOError:
        print "Could not open data file... please check your file and try again"
        sys.exit(0)

    print "\nSome quick info on your data:"
    print "Shape: ", raw_data.shape 
    print "Number of bytes: ", raw_data.nbytes ,"\n"

    #check if analysisType input is valid
    try:
        modelType = sys.argv[2]
    except IndexError:
        print "You did not enter modelType. Please try again - your syntax should look something like:\n ./linear_analysis.py myData.txt lasso 5(only for lasso or fit) 7.8(only for lasso)\nALL ARGUMENTS SHOULD BE SEPARATED BY A SPACE-"
        sys.exit(0) 

    #assign centered data to array W
    W = center(raw_data)
    #assign number of rows in matrix W to array N
    N = len(W)

    try: 
        dependent = sys.argv[3]
        X, Y = slice(dependent, raw_data)   
    except IndexError:
        print "No number inputted, cannot compute X and Y - OK if model type is pca"
    except ValueError:
        print "An integer is expected here, cannot compute X and Y"

    if modelType == "fit":

        print "FIT:\nComputes coefficients of the least-squares linear fit of the data,where the user-inputted column is the dependent variable and the other columns are independent variables\n"
        try:
            M = fit(X,Y) 
        except NameError:
            print "Dependent variable column not specified... cannot compute fit. Please try again."
            sys.exit(0)

        #generate n x vals for n M vals 
        x_list = ['x' + str(i) for i in range(len(M))]

        print "Coefficients of the least-squares linear fit of your data with column ", dependent ," as the dependent variable:"
        print M

        print "The Residual Sum of Squares (RSS) of your data:"
        print RSS(X, Y, M)

        print "The Fraction-of-Variance-Explained:" 
        print fractionOfVariance(X, Y, M)

        print "Complete equation for the model:"
        for i in range(len(x_list)):
            print str(M[i][0]) + '(' +  x_list[i] + '-' + str(np.mean(X[i])) + ')',
            if i+1 != len(x_list):
                print '+',
        print
               
    elif modelType == "pca":

        print "PCA:\nComputes eigenvalues and their corresponding eignevectors, and produces scatterplots in a grid showing the relarions among original variables\n"
        evalues, evectors = PCA(N, W)
        print "eigenvalues from greatest to least:"
        #[::-1] flips the vector
        print evalues[::-1]
        print "Absolute largest eignenvalue: ", evalues[-1]
        print "Its associated eigenvector: ", evectors[-1]
       
        myfig = plt.figure()
        myfig.add_subplot(4,4,1).scatter(evectors[0], evectors[0])
        myfig.add_subplot(4,4,2).scatter(evectors[0], evectors[1])
        myfig.add_subplot(4,4,3).scatter(evectors[0], evectors[2])
        myfig.add_subplot(4,4,4).scatter(evectors[0], evectors[3])
        myfig.savefig("diabetesscatter.png")
       
        #PRODUCE SCATTERPLOTS HERE

    elif modelType == "lasso":
 
        print "LASSO:\nRuns lasso minimization for a range of t between 0 and Tinfinity, where Tinfinity is the largest absolute value i X.T(Y). Note that if tVal is specified in the user input, the program then runs lasso minimization at the user defined tVal. Also produces a graph of how coefficients vary with t, and a grpah of the Fraction-of-Variance-Explained so the user can assess the best model\n"

        if len(sys.argv) >= 5:

            t0 = float(sys.argv[4])

            try:
                M = lasso_step(X, Y, t0)

            except NameError:
                print "Dependent variable column not specified... cannot compute fit. Please try again."
                
            print M
            print "The Residual Sum of Squares (RSS) of your data:"
            print RSS(X, Y, M)

            print "The Fraction-of-Variance-Explained:"
            print fractionOfVariance(X, Y, M)

            x_list = ['x' + str(i) for i in range(len(M))]
            print "Complete equation for the model:"
            for i in range(len(x_list)):
                print str(M[i]) + '(' +  x_list[i] + '-' + str(np.mean(X[:][i])) + ')',
                if i+1 != len(x_list):
                    print '+',
            print
          
        else:

            try:
                M, t = lasso(X, Y)
            except NameError:
                 print "Dependent variable column not specified... cannot compute fit. Please try again."
            M = np.array(M)
            t = np.array(t)
            pic(t, M)
            M = np.mean(M, axis=0)
            print M
            print "Fraction-of-Variance-Explained: ", fractionOfVariance(X, Y, M)
            
    else:
        try:
            raise TypeError
        except TypeError: 
            print "You did not enter a valid model type. Valid model types are pca, fit, and lasso."
