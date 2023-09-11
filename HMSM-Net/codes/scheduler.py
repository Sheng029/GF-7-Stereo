def schedule(epoch):
    lr = 0.0001 * 0.5 ** int(epoch / 10)
    return lr
