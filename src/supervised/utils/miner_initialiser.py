from pytorch_metric_learning.miners import TripletMarginMiner, BatchEasyHardMiner


def initialise_miner(name, margin=0.2, type_of_triplets='easy'):
    if name == "triplet_margin_miner":
        miner = TripletMarginMiner(margin=margin, type_of_triplets=type_of_triplets)
    elif name == "batch_easy_hard_miner":
        miner = BatchEasyHardMiner()
    else:
        raise NameError(f"The miner: {name} does not exist!")
    return miner
