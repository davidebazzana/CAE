import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

def calc_ari_score(batch_size, labels_true, labels_pred, with_background, is_dataset_4Shapes):
    ari = 0
    if is_dataset_4Shapes:
        labels_true = labels_true["pixelwise_instance_labels"]
    labels_true = labels_true.numpy()
    # print(f'labels_true: {labels_true["pixelwise_instance_labels"].numpy().shape}')
    # print(f'labels_pred: {labels_pred.shape}')
    for idx in range(batch_size):
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
    return ari / batch_size