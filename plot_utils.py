import numpy as np
import matplotlib.pyplot as plt


def plot_pair(image_pairs):
    plt.figure(figsize=(9,2))
    plt.gray()
    imgs = image_pairs[0][0].cpu().detach().numpy()
    recon = image_pairs[0][1]
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9: break

        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item)

    plt.show()