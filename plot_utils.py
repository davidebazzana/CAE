import numpy as np
import matplotlib.pyplot as plt


def plot_in_out_labels(images):
    plt.figure(figsize=(9,3))
    plt.gray()
    input = images[0][0]
    output = images[0][1]
    labels = images[0][2]
    for i, item in enumerate(input):
        if i >= 9: break
        ax = plt.subplot(3, 9, i+1)
        ax.set(ylabel="Input")
        ax.label_outer()
        plt.imshow(item[0])

    for i, item in enumerate(output):
        if i >= 9: break

        ax = plt.subplot(3, 9, 9+i+1)
        ax.set(ylabel="Output")
        ax.label_outer()
        plt.imshow(item[0])

    for i, item in enumerate(labels):
        if i >= 9: break

        ax = plt.subplot(3, 9, 18+i+1)
        ax.set(ylabel="Labeling")
        ax.label_outer()
        plt.imshow(item)

    plt.show()


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