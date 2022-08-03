from pytorch_metric_learning.miners import TripletMarginMiner, BatchEasyHardMiner, BatchHardMiner


def initialise_miner(name, miner_margin=0.2, type_of_triplets='easy'):
    if name == "triplet_margin_miner":
        miner = TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
    if name == "batch_hard_miner":
        miner = BatchHardMiner()
    elif name == "batch_easy_hard_miner":
        miner = BatchEasyHardMiner()
    else:
        raise NameError(f"The miner: {name} does not exist!")
    return miner
