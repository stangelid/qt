import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class ResidualCoefficientScheduler():
    '''Computes the current residual coefficient for warmup annealing
       based on the current batch counter and epoch (both 0-indexed).
       The class is initialized with the warmup epochs, annealing epochs
       and total number of batches.
    '''
    def __init__(self, warmup_epochs, annealing_epochs, nbatches, min_coeff=0.0):
        self.warmup_epochs = warmup_epochs
        self.annealing_epochs = annealing_epochs
        self.nbatches = nbatches
        self.min_coeff = min_coeff

    def get_residual_coefficient(self, batch_idx, epoch_idx):
        if epoch_idx < self.warmup_epochs:
            return 1.0
        residual = 1.0 \
                 - (batch_idx \
                   + float(self.nbatches) * (epoch_idx - self.warmup_epochs)) \
                 / (self.annealing_epochs * self.nbatches) * (1.0 - self.min_coeff)

        return max(self.min_coeff, residual)


def lr_calc(epoch):
    '''Computes learning rate for current epoch, if learning rate
    drop-off is enabled'''
    if epoch < args.lr_drop_epoch \
            or epoch >= args.lr_drop_epoch + args.lr_rtrn_epochs:
        return args.lr_decay ** epoch
    else:
        return (args.lr_decay ** epoch) \
             - (args.lr_decay ** epoch) \
             * (1 - (epoch - args.lr_drop_epoch) / args.lr_rtrn_epochs)
