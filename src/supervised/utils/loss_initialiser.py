from pytorch_metric_learning.losses import TripletMarginLoss, NTXentLoss


def initialise_loss(name, loss_margin=0.2, temperature=1.0):
    if name == "triplet_margin":
        loss = TripletMarginLoss(margin=loss_margin)
    elif name == "ntxent":
        loss = NTXentLoss(temperature=temperature)
    else:
        raise NameError(f"The loss: {name} does not exist!")
    return loss
