from matplotlib import pylab as plt
from matplotlib.patches import Polygon


### Segmentation ###

def show_image_and_overlay(image, mask, color='red', alpha=0.2):
    """
    :param image: the image to display
    :param mask: mask border e.g list of x,y representing the border of the polygon or matrix of zeros and ones
                 representing the mask
    :param color: color of the overlay (applicable only if we get border and not mask)
    :param alpha: alpha value
    :return:
    """
    if isinstance(mask, type([])):
        overlay = Polygon(mask, color=color, alpha=alpha)
        plt.imshow(image)
        plt.gca().add_patch(overlay)
    else:
        plt.imshow(image)
        plt.imshow(mask, alpha=alpha)

def apply_mask_on_image(image, mask, intensity=200):
    return image + mask * intensity

