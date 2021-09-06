import torch
import matplotlib.pyplot as plt

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(points, scores, dist_thres=50 / 2.5, top_k=50):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        points: (tensor) The location preds for the img, Shape: [num_priors,2].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        dist_thres: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.zeros_like(scores).long()
    if points.numel() == 0:
        return keep
    v, indices = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    top_k = min(top_k, len(indices))
    indices = indices[-top_k:]  # indices of the top-k largest vals

    # keep = torch.Tensor()
    count = 0
    while indices.numel() > 0:
        idx = indices[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = idx
        count += 1
        if indices.numel() == 1:
            break
        indices = indices[:-1]  # remove kept element from view
        target_point = points[idx, :]
        # load bboxes of next highest vals
        remaining_points = points[indices, :]
        dists = torch.norm(target_point - remaining_points, dim=1)  # store result in distances
        # keep only elements with an dists > dist_thres
        indices = indices[dists > dist_thres]
    return keep, count

def loss_curve(path, epochs, train_loss, test_loss):
    train_loss = history.history['train_loss']
    test_loss = history.history['test_loss']
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, test_loss, 'b', label='Testing loss')
    plt.title('Training and Tesing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

