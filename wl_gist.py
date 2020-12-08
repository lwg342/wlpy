# %% Importing Libs
import matplotlib.pyplot as plt
import numpy as np
# %% Matrix Plots


def heatmap(S, title='Matrix_S', path=None):
    """
    Plot the heatmap of a matrix S
    """
    from datetime import datetime
    timestr = datetime.now().strftime('%Y%m%d%H')
    fig = plt.figure()
    cax = fig.add_subplot(111).matshow(S, cmap='inferno')
    fig.colorbar(cax)
    plt.figtext(0.5, 0.01, 'Heatmap of' + '-'.join(title.split()),  wrap=True,
                horizontalalignment='center', fontsize= 10)
    if path != None:
        fig.savefig(path + 'Heatmap-' + title + '-' + timestr + '.eps', format= 'eps')
    fig.show()
    

def heatmap_dictionary(dict_matrices, shape = [1,1], path = None, title = ''):
    from datetime import datetime
    timestr = datetime.now().strftime('%Y%m%d%H')
    fig, ax = plt.subplots(shape[0], shape[1], figsize = [15,15])
    j = 0
    for i in dict_matrices.keys():
        # print(i)
        cax = ax.flatten()[j]
        current_mat = cax.matshow(dict_matrices.get(i))
        fig.colorbar(current_mat, ax = ax.ravel()[j])
        cax.set_title('Heatmap of ' + '-'.join(i.split()),  wrap=True, horizontalalignment='center', fontsize=10)
        j = j + 1
    if path != None:
        fig.savefig(path + 'Heatmap-' + title + '-' + timestr + '.eps', format= 'eps')
    plt.tight_layout()
    fig.show()


# %% Define the function that draws a network


# def show_graph_with_labels(adjacency_matrix):
#     rows, cols = np.where(adjacency_matrix == 1)
#     gr.add_edges_from(edges)
#     nx.draw(gr, node_size=300, with_labels=True)
#     plt.savefig('graph')
#     plt.show()

# %% Manipulating dataframe



# %%

def linear_probit(endog, params, add_constant = 1):
    """
    Return: normal.cdf(X*beta)
    Endog: A list of k N-dimensional endogenous variables including the intercept
    params: the coefficients beta
    add_intercept: do we need to add an intercept to the endogenous variables. 
    """
    from scipy.stats import norm as normal_rv
    # assert((len(endog) + add_constant) == params.shape[0])
    if add_constant == 1:
        intercept_term = np.ones(endog[0].shape)
        X = np.array([intercept_term] + endog)
    else:
        X = np.array(endog)
    Phi = normal_rv.cdf(X.transpose()@ params)
    return Phi
        

# %%
def generalized_thresholding(S : np.ndarray, T : np.ndarray, method='hard threshold') -> np.ndarray:
    """
    H = [h_ij] where h_ij(s_ij, t_ij) computes the generalized shrinkaged estimates
    This function applies generalized thresholding to a matrix M \n
    M : input matrix \n
    method: specify the name of the generalized thresholding function\n
    """       
    if method == 'hard threshold':
        H = S
        H[np.where(np.abs(H) < T)] = 0
    if method == 'soft threshold':
        H = S
        H[np.where(np.abs(H) < T)] = 0
        H[np.where(np.abs(H) >= T)] = H[np.where(np.abs(H) >= T)] - \
            T[np.where(np.abs(H) >= T)]
    else: 
        print('unfinished')
        H = None
    return H
# %%
