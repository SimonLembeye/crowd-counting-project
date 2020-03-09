from abc import ABC, abstractmethod

class abcTrainer(ABC):

    def __init__(self, dataloader, net, loss, optimizer, device, validation_frequency=1, max_epoch=100):
        self.train_loader, self.val_loader, self.test_loader = dataloader()
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.best_mae = 1e20
        self.best_mse = 1e20
        self.epoch = 0
        self.validation_frequency = validation_frequency
        self.max_epoch = max_epoch
        self.device = device


    def train(self):
        for epoch in range(0, self.max_epoch):
            self.epoch = epoch

            # training
            self.train_epoch()

            # validation
            if epoch%self.validation_frequency==0:
                self.validate()

        print(f'Train finished | best_mse: {self.best_mse} | best_mae: {self.best_mae}')


    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def validate(self):
        pass
