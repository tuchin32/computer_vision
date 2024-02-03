################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
# exp_name   = 'sgd_pre_da' # name of experiment
exp_name = 'vgg13_v9'

# Model Options
# model_type = 'resnet18' # 'mynet' or 'resnet18'
model_type = 'mynet'

# Learning Options
# epochs     = 50           # train how many epochs
epochs     = 80
# batch_size = 32           # batch size for dataloader 
batch_size = 256
# use_adam   = False        # Adam or SGD optimizer
use_adam   = True
# lr         = 1e-2         # learning rate
lr         = 1e-3
# milestones = [16, 32, 45] # reduce learning rate at 'milestones' epochs
milestones = 1