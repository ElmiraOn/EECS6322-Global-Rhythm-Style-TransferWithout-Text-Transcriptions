
def worker_init_fn(x):
    return np.random.seed((torch.initial_seed()) % (2**32))

def get_loader(hparams):
    """Build and return a data loader."""

    dataset = Utterances(hparams)

    my_collator = MyCollator(hparams)

    sampler = MultiSampler(len(dataset), hparams.samplier, shuffle=hparams.shuffle)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=hparams.batch_size,
                                  sampler=sampler,
                                  num_workers=hparams.num_workers,
                                  drop_last=True,
                                  pin_memory=False,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=my_collator)
    return data_loader

data_loader = get_loader()