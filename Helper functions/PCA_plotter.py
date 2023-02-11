
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from scipy.linalg import svd # Singular value decomposition package

from mpl_toolkits import mplot3d # plot 3D plots


def pca_variance_plotter(data, threshold: float):

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



def pca_point_plotter3d(pca_data,
                    class_data,
                    pc_list,
                    class_names):
    
    """
    
    Inputs:

    pca_data -> The transformed data by doing a SVD
    class_data -> The vector, that contain each entries' class
    pc_list -> list of the pc numbers that will be plotted.
    class_names -> the name of the classes
    
    """

    plt.figure(figsize = (16, 9)) # maybe f2 = plt.figure(....)

    C = len(class_names) # number of classes we have.

    plt.title("3D PCA plot")

    ax = plt.axes(projection ="3d")

    for c in range(C):
        
        class_mask = class_data == c

        ax.scatter3D(pca_data.iloc[class_mask,pc_list[0]], # plotting the first index in x-axis
                    pca_data.iloc[class_mask,pc_list[1]], # plotting the second index in y-axis
                    pca_data.iloc[class_mask,pc_list[2]]) # plotting the third index in z-axis
    
    plt.legend(class_names)

    ax.set_xlabel('PC{0}'.format(pc_list[0]+1))
    ax.set_ylabel('PC{0}'.format(pc_list[1]+1))
    ax.set_zlabel('PC{0}'.format(pc_list[2]+1))

    plt.show()









