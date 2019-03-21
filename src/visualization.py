from src import constants as c
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from scipy.stats import norm
from torch import from_numpy


def imscatter(x, y, ax, data, zoom):
    """
    x, y : Dataframe columns of x, y coordinates
    ax: matplotlib axes object
    data: PyTorch Dataset
    zoom: display scaling for images
    indices: indices of Dataset
    """
    images = []
    for i in range(len(data)):
        x0, y0 = x[i], y[i]
        img = data[i][0]
        img = img.numpy().transpose(1,2,0)
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    
def get_latent_vector(image, model):
    """
    model: VAE with sampling and encode methods
    """
    return model.sampling(*model.encode(image.unsqueeze(0).to(c.device)))

def plot_latent_space(model, zdim=2, dimensions=[0,1], resolution=15):
 
    u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, resolution),
                                   np.linspace(0.05, 0.95, resolution)))
    z_grid = norm.ppf(u_grid)

    sampled = z_grid.reshape(resolution*resolution, 2)
    result = np.zeros((resolution*resolution, zdim))
    result[:sampled.shape[0], dimensions[0]] = sampled[:,0]
    result[:sampled.shape[0], dimensions[1]] = sampled[:,1]
   
    x_decoded = model.decode(from_numpy(result).to(c.device).float())
    x_decoded = x_decoded.reshape(resolution, resolution, 3, c.image_size, c.image_size)
    return x_decoded


def latent_interpolation(start, end, model, num_samples=10):
    """
    start: starting image
    end: ending image
    model: VAE with decode() method
    """
    latent_start = get_latent_vector(start, model)
    latent_end = get_latent_vector(end, model)
    alphas = np.linspace(0,1,num_samples)
    images = []
    for i in range(num_samples):
        result = model.decode(latent_start*(1-alphas[i])+latent_end*alphas[i])
        result = result.cpu().detach().numpy().squeeze().transpose(1,2,0) 
        images.append(result)
        
    return images

def latent_interpolation_by_dimension(start, end, model, zdim, num_samples=10):
    """
    start: starting image
    end: ending image
    model: VAE with decode() method
    """
    latent_start = get_latent_vector(start, model)
    latent_end = get_latent_vector(end, model)
    alphas = np.linspace(0,1,num_samples)
    images = []

    for dim in range(10):
        tmp = latent_start.clone()
        dimension = []
        for alpha in alphas:
            tmp[:,dim] = latent_start[:, dim] * (1-alpha) + latent_end[:, dim] * alpha
            result = model.decode(tmp)
            result = result.cpu().detach().numpy().squeeze().transpose(1,2,0)
            dimension.append(result)
        images.append(dimension)
        latent_start[:,dim]= latent_end[:, dim]
        
    return images

def plot_pca(n, dataframe):
    """
    n: number of pca-components
    dataframe: pandas dataframe where pca components are stored as pcX
    """
    fig = plt.figure(figsize=(10,10))
    fig.tight_layout()
    
    for i, j in combinations(range(n), 2):
        ax = fig.add_subplot(n-1, n-1, i*(n-1)+j)
        x = dataframe['pc{}'.format(i+1)]
        y = dataframe['pc{}'.format(j+1)]
        scatter = ax.scatter(x, y, c=dataframe['Tool'], cmap='bwr', alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('PC{}'.format(i+1))
        ax.set_ylabel('PC{}'.format(j+1))

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='No Tool',
                              markerfacecolor='b', alpha=0.2, markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Tool',
                              markerfacecolor='r', alpha=0.2, markersize=15)
                      ]
    fig.legend(handles=legend_elements, loc = (0.1, 0.1))
        
    return fig
    