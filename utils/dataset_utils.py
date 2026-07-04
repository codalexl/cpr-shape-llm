def simple_collator(data):
    """Required for trl PPOTrainer to handle queries of different length"""
    return dict((key, [d[key] for d in data]) for key in data[0])