import numpy as np
import torch

from .abctrainer import abcTrainer
from losses import bayes_aleatoric_loss

class BayesTrainer(abcTrainer):
    def __init__(self, dataloader, net, loss, optimizer, device, post_prob, validation_frequency=1, max_epoch=100, aleatoric=False):
        super().__init__(dataloader, net, loss, optimizer, device, validation_frequency=validation_frequency, max_epoch=max_epoch, aleatoric=aleatoric)

        self.post_prob = post_prob

    def train_epoch(self):
        self.net.train()
        epoch_loss = 0

        for step, (inputs, points, targets, st_sizes) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                prob_list = self.post_prob(points, st_sizes)
                if self.aleatoric:
                    outputs, logvar = self.net(inputs, aleatoric=self.aleatoric)
                    loss = bayes_aleatoric_loss(self.loss, outputs, targets, logvar, prob_list)
                else:
                    outputs = self.net(inputs)
                    loss = self.loss(prob_list, targets, outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0) # batch size
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss += float(loss)

                print(f'epoch: {self.epoch} | step: {step} | gd_count: {gd_count} | prediction: {pre_count} | loss: {loss}')

        self.writer.add_scalar('train loss Bayes',
            epoch_loss,
            self.epoch)

    def validate(self):
        epoch_start = time.time()
        self.net.eval()  # Set model to evaluate mode
        epoch_res = []

        # Iterate over data.
        for inputs, count, name in self.val_loader:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            assert inputs.size(0) == 1 # 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                if self.aleatoric:
                    outputs, _ = self.net(inputs, aleatoric=self.aleatoric)
                else:
                    outputs = self.net(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)


        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        self.writer.add_scalar('val MAE Bayes',
                            mae,
                            self.epoch)
        self.writer.add_scalar('val MSE Bayes',
                        mse,
                        self.epoch)

        print('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            print("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                            self.best_mae,
                                                                                 self.epoch))
            torch.save(self.net.state_dict(), os.path.join(save_dir, 'best_model_bayes.pth'))
