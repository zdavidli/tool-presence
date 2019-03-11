from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np


def imscatter(x, y, ax, data, zoom, indices):
    """
    x, y : Dataframe columns of x, y coordinates
    ax: matplotlib axes object
    data: PyTorch Dataset
    zoom: display scaling for images
    indices: indices of Dataset
    """
    images = []
    for i in range(len(indices)):
        x0, y0 = x[i], y[i]
        img = data[indices[i]][0]
        img = img.numpy().transpose(1,2,0)
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()