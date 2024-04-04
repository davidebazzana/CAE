import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import torch
from einops import rearrange
from sklearn.cluster import KMeans

THRESHOLD = 0.0001

def calc_ari_score(labels_true, labels_pred, with_background):
    ari = 0
    labels_true = labels_true.numpy()
    # print(f'labels_true: {labels_true["pixelwise_instance_labels"].numpy().shape}')
    # print(f'labels_pred: {labels_pred.shape}')
    for idx in range(labels_pred.shape[0]):
        if with_background:
            area_to_eval = np.where(
                labels_true[idx] > -1
            )  # Remove areas in which objects overlap.
        else:
            area_to_eval = np.where(
                labels_true[idx] > 0
            )  # Remove background & areas in which objects overlap.

        ari += adjusted_rand_score(
            labels_true[idx][area_to_eval], labels_pred[idx][area_to_eval]
        )
    return ari / labels_pred.shape[0]


def object_discovery(model, z: torch.Tensor, number_of_objects: int):
    prediction_labels = np.zeros(
        (z.shape[0], 32, 32)
    )
    output = model.output_layer(z).cpu()
    z = z.cpu()
    for image_idx in range(z.shape[0]):
        output_magnitude = rearrange(output[image_idx], "c h w -> c (h w)")[0]
        output_magnitude_min = torch.min(output_magnitude, dim=0).values
        output_magnitude_max = torch.max(output_magnitude, dim=0).values
        # Normalization -> [0, 1]
        output_magnitude = torch.div(torch.sub(output_magnitude, output_magnitude_min), torch.sub(output_magnitude_max, output_magnitude_min))
        output_magnitude = torch.unsqueeze(output_magnitude, dim=1)
        background_masking = torch.where(output_magnitude > THRESHOLD, 1.0, 0.0)
        img_phase = z[image_idx].angle()
        img_phase = rearrange(img_phase, "c h w -> (h w) c")
        # Hadamard product
        img_phase = torch.mul(img_phase, background_masking)
        k_means = KMeans(n_clusters=(number_of_objects + 1), random_state=0, n_init=10).fit(img_phase)
        cluster_img = rearrange(k_means.labels_, "(h w) -> h w", h=32, w=32)
        prediction_labels[image_idx] = cluster_img

    return prediction_labels