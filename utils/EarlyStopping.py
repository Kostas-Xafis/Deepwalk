class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.min_delta = delta
        self.cur_val = float("inf")

    def __call__(self, train_loss: int) -> bool:
        if abs(train_loss - self.cur_val) <= self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
        self.cur_val = train_loss
        return self.early_stop