import numpy as np


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def LearningRateScheduler(loss_his, optimizer, lr_base):

    num_shock = np.sum((loss_his[:-1] - loss_his[1:]) < 0)
    if num_shock > 0.45 * loss_his.shape[0]:
        lr_new = lr_base*0.8
        adjust_learning_rate(optimizer, lr_new)
    # elif num_shock < 0.02 * loss_his.shape[0]:
    #     lr_new = lr_base*1.5
    #     adjust_learning_rate(optimizer, lr_new)
    else:
        lr_new = lr_base
    print('*** lr {} -> {}  ***'.format(lr_base, lr_new))
    print('num_shock', num_shock)
    return lr_new
