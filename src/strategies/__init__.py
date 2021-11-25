from .opt_default import opt_default
from .mlp_default import mlp_default

"""
strategies = s1, s2, s3 ...
s_i = [stage-1, stage-2, stage-3]
stage-i = {
    update_params = [],
    losses_weight = [],
    lr = xxx,
    epoch = xxx,
    filter_loss = [
        (loss_name-1, strategy-1),
        (loss_name-2, strategy-2),
        ...
    ], # in all the samples, only the sample that satisfies all criteria can be selected
    select_loss = loss_name, # select the one with best criteria
}
"""

strategies = dict(
    opt_default = opt_default,
    mlp_default = mlp_default,
)