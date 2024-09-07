import numpy as np


def compute_iou(cpu_masks, i, j):
    """ Returns soft IOU between mask_i and mask_j.
        See Eqn B.6 in Appendix B: Event proposals
        via amortized inference - segmentation neural network
        > test procedure
    """
    numerator = np.minimum(cpu_masks[i, :, :], cpu_masks[j, :, :]).sum()
    denominator = 1e-10 + np.maximum(cpu_masks[i, :, :], cpu_masks[j, :, :]).sum()
    return numerator/denominator


def iou_filter(iou_matrix, iou_threshold):
    """ Returns a list of indexes, which masks pass the (single) IOU threshold """
    # Filter masks based on overlap. Masks are ordered in terms of score
    idx_to_keep = [0]
    for i in range(1, iou_matrix.shape[0]):
        keep = True
        for j in idx_to_keep:
            iou = iou_matrix[i, j]
            if iou >= iou_threshold:
                keep = False
                break
        if keep:
            idx_to_keep.append(i)
    return idx_to_keep


def iou_filter_by_class(iou_matrix, scores, pred_classes, score_thresholds, iou_threshold):
    """ Returns a list of indexes, which masks pass the (class-specific) IOU threshold """
    idx_to_keep = [0]
    for i in range(1, iou_matrix.shape[0]):
        keep = scores[i] > score_thresholds[pred_classes[i]]
        for j in idx_to_keep:
            iou = iou_matrix[i, j]
            if iou >= iou_threshold:
                keep = False
                break
        if keep:
            idx_to_keep.append(i)
    return idx_to_keep


def create_iou_matrix(cpu_masks):
    """ Return matrix with IOUs for all pairs of masks in cpu_masks """
    n_masks = cpu_masks.shape[0]
    iou_matrix = np.zeros((n_masks, n_masks))
    for i in range(n_masks):
        for j in range(i, n_masks):
            iou_matrix[i, j] = compute_iou(cpu_masks, i, j)
    iou_matrix = iou_matrix + iou_matrix.T
    np.fill_diagonal(iou_matrix, 1.0)
    return iou_matrix


def create_f0pred_iou_matrix(predictions, harmonic_class_idx=2):
    """ Create an IOU matrix based on F0 predictions, for the IoU between two harmonic events """
    iou_matrix_f0 = create_iou_matrix(predictions["f0"].cpu().numpy())
    iou_matrix_binary = create_iou_matrix(predictions["ibm"].numpy())
    is_harmonic = predictions["classes"] == harmonic_class_idx
    both_harmonic = (is_harmonic[None, :] * is_harmonic[:, None]).numpy()
    iou_matrix = both_harmonic * iou_matrix_f0 + (1 - both_harmonic) * iou_matrix_binary
    return iou_matrix
