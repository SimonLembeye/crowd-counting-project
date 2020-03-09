import torch

from models import vgg19, CSRNet
from loaders import loading_data_GT,  loading_data_Bayes
from trainers import GTTrainer, BayesTrainer
from losses import PostProb, BayesLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

# args = None
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Train ')
#     parser.add_argument('--method', default='ground_truth',
#                         help='Method - ground_truth or bayes or aleatoric')
#     parser.add_argument('--model', default='csrnet',
#                         help='NN Model - csrnet or vgg19_extended')
#     parser.add_argument('--lr', default='5e-6',
#                         help='Learning Rate')
#     parser.add_argument('--use-bg', default='False',
#                         help='Use background for Bayes method')
#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     args = parse_args()
#     if args.method == "ground_truth":
#         gt_net = CSRNet().to(device)
#         loss = torch.nn.MSELoss().to(device)
#         optimizer = torch.optim.Adam(gt_net.parameters(), lr=args.lr)
#         gt_trainer = GTTrainer(loading_data_GT, gt_net, loss, optimizer, device, max_epoch=250)
#         gt_trainer.train()
#
#     elif args.method == "bayes":


# Launch GT Train !
# lr = 1e-6
# gt_net = CSRNet().to(device)
# loss = torch.nn.MSELoss().to(device)
# print(gt_net)
# optimizer = torch.optim.Adam(gt_net.parameters(), lr=lr)
# # optimizer = optim.SGD(gt_net.parameters(), lr=lr, momentum=0.95,weight_decay=5e-4)
#
# gt_trainer = GTTrainer(loading_data_GT, gt_net, loss, optimizer, device, aleatoric=True, max_epoch=250)
# gt_trainer.train()


lr = 1e-6
sigma = 10
use_background = False
background_ratio = 1
crop_size = 256
downsample_ratio = 8

bayes_net = vgg19().to(device)
post_prob = PostProb(sigma,
                   crop_size,
                   downsample_ratio,
                   background_ratio,
                   use_background,
                   device)
loss = BayesLoss(use_background, device)
optimizer = torch.optim.Adam(bayes_net.parameters(), lr=lr)
# optimizer = optim.SGD(gt_net.parameters(), lr=lr, momentum=0.95,weight_decay=5e-4)

bayes_trainer = BayesTrainer(loading_data_Bayes, bayes_net, loss, optimizer, device, post_prob, max_epoch=250, aleatoric=True)
bayes_trainer.train()
