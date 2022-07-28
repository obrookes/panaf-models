from pytorch_metric_learning.losses import TripletMarginLoss, NTXentLoss


def initialise_loss(name, temperature=1):
    if name == "triplet_margin":
        loss = TripletMarginLoss()
    elif name == "ntxent":
        loss = NTXentLoss(temperature=temperature)
    else:
        raise NameError(f"The loss: {name} does not exist!")
    return loss
