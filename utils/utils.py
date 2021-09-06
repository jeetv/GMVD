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

def loss_curve(path, epochs, train_loss, test_loss, train_prec, test_prec, train_recall, test_recall, train_acc, test_acc):
    label_train = "train loss"
    label_test = "test loss"

    fig = plt.figure()
    ax1 = fig.add_subplot(221, title="Training and Tesing loss")
    ax1.plot(epochs, train_loss, 'g', label=label_train)
    ax1.plot(epochs, test_loss, 'b', label=label_test)
    ax1.set_xlabel("Epochs")
    ax1.set_xlabel("Loss")

    ax2 = fig.add_subplot(222, title="Training and Tesing Precision")
    ax2.plot(epochs, train_prec, 'g', label='train prec')
    ax2.plot(epochs, test_prec, 'b', label='test prec')
    ax2.set_xlabel("Epochs")
    ax2.set_xlabel("Precision")

    ax3 = fig.add_subplot(223, title="Training and Tesing Recall")
    ax3.plot(epochs, train_recall, 'g', label='train recall')
    ax3.plot(epochs, test_recall, 'b', label='test recall')
    ax3.set_xlabel("Epochs")
    ax3.set_xlabel("Recall")

    ax4 = fig.add_subplot(224, title="Training and Tesing Accuracy")
    ax4.plot(epochs, train_acc, 'g', label='train acc')
    ax4.plot(epochs, test_acc, 'b', label='test acc')
    ax4.set_xlabel("Epochs")
    ax4.set_xlabel("Accuracy")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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

