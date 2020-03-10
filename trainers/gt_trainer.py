from torch.autograd import Variable
import torch

from .abctrainer import abcTrainer
from losses import GT_aleatoric_loss

LOG_PARA = 100.
seed = 1


class GTTrainer(abcTrainer):
    def __init__(self, dataloader, net, loss, optimizer, device, validation_frequency=1, max_epoch=100, aleatoric=False):
        super().__init__(dataloader, net, loss, optimizer, device, validation_frequency=validation_frequency, max_epoch=max_epoch, aleatoric=aleatoric)

    def train_epoch(self):
        self.net.train()
        epoch_loss = 0

        for step, data in enumerate(self.train_loader, 0):
            img, gt_map = data
            img = Variable(img).to(self.device)
            gt_map = Variable(gt_map).to(self.device)
            self.optimizer.zero_grad()

            if self.aleatoric:
                pred_density_map, logvar = self.net(img, aleatoric=self.aleatoric)
                loss = GT_aleatoric_loss(self.loss, pred_density_map, gt_map, logvar)
            else:
                pred_density_map = self.net(img)
                loss = self.loss(pred_density_map, gt_map)
            loss.backward()
            self.optimizer.step()

            gt_count = [int(gt_map[i].sum().data / LOG_PARA) for i in range(gt_map.size()[0])]
            pre_count = [int(pred_density_map[i].sum().data/LOG_PARA) for i in range(pred_density_map.size()[0])]
            epoch_loss += float(loss)

            print(f'epoch: {self.epoch} | step: {step} | count: {gt_count} | prediction: {pre_count} | loss: {loss}')

        self.writer.add_scalar('train loss GT',
            epoch_loss,
            self.epoch)


    def validate(self):
        epoch_start = time.time()
        self.net.eval()
        epoch_res = []

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).to(self.device)
                assert img.size(0) == 1
                gt_map = Variable(gt_map).to(self.device)

                if self.aleatoric:
                    pred_density_map, _ = self.net(img, aleatoric=aleatoric)
                else:
                    pred_density_map = self.net(img)

                pred_cnt = int(gt_map[0].sum().data / LOG_PARA)
                gt_count = int(pred_density_map[0].sum().data/LOG_PARA)
                res = gt_count - pred_cnt

                epoch_res.append(res)


        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        print('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        self.writer.add_scalar('val MAE GT',
                    mae,
                    self.epoch)
        self.writer.add_scalar('val MSE GT',
                        mse,
                        self.epoch)

        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            print("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                            self.best_mae,
                                                                                 self.epoch))
            torch.save(self.net.state_dict(), os.path.join(self.save_dir, 'best_model_gt.pth'))
