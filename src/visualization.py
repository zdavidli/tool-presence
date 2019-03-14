from src import constants as c
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np


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
    
def latent_interpolation(start, end, model, num_samples=10):
    """
    start: starting image
    end: ending image
    model: VAE with decode() method
    """
    latent_start = model.sampling(*model.encode(start.unsqueeze(0).to(c.device)))
    latent_end = model.sampling(*model.encode(end.unsqueeze(0).to(c.device)))
    alphas = np.linspace(0,1,num_samples)
    images = []
    for i in range(num_samples):
        result = model.decode(latent_start*(1-alphas[i])+latent_end*alphas[i])
        result = result.cpu().detach().numpy().squeeze().transpose(1,2,0) 
        images.append(result)
        
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
    