
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd # Singular value decomposition package


def pca_plotter(data, threshold: float):

    """
    Function: plots the PCs from the data to see the cummulative variance

    Inputs:

    data -> The data which PCs will be based from

    threhsold -> float, that will display the ratio of variance that is satisfying.

    Outputs:

    Plot that shows the individual PCs variance explained as well as the cummulative variance.
    
    
    """

    U,S,V = svd(data,full_matrices=False)

    # Compute variance explained by principal components

    rho = (S*S) / (S*S).sum() # The way to compute the individual rho,

    # Plotting variance explained.
    plt.figure()
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.show()










