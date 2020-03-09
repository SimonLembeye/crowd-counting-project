import torch
import argparse

from models import vgg19, CSRNet
from loaders import loading_data_GT,  loading_data_Bayes
from trainers import GTTrainer, BayesTrainer
from losses import PostProb, BayesLoss
from utils.parser import str2bool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--method', default='ground_truth',
                        help='Method - ground_truth or bayes')
    parser.add_argument('--model', default='csrnet',
                        help='NN Model - csrnet or vgg19_extended')
    parser.add_argument('--aleatoric', type=str2bool, default='false',
                        help='Boolean - use aleatoric loss')
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='Learning Rate')
    parser.add_argument('--use-bg', type=str2bool, default='false',
                        help='Use background for Bayes method')
    parser.add_argument('--epochs', type=int, default=20,
                        help='int - number of training epoch')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.method == "ground_truth":
        gt_net = CSRNet().to(device)
        loss = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(gt_net.parameters(), lr=int(args.lr))
        gt_trainer = GTTrainer(loading_data_GT, gt_net, loss, optimizer, device, aleatoric=args.aleatoric, max_epoch=args.epochs)
        gt_trainer.train()

    elif args.method == "bayes":
        sigma = 10
        use_background = args.use_bg
        background_ratio = 1
        crop_size = 256

        if args.model == "csrnet":
            bayes_net = CSRNet().to(device)
            downsample_ratio = 1
        elif args.model == "vgg19_extended":
            bayes_net = vgg19().to(device)
            downsample_ratio = 8
        else:
            print('model should be csrnet or vgg19_extended')

        post_prob = PostProb(sigma, crop_size, downsample_ratio, background_ratio, use_background, device)
        loss = BayesLoss(use_background, device)
        optimizer = torch.optim.Adam(bayes_net.parameters(), lr=args.lr)
        bayes_trainer = BayesTrainer(loading_data_Bayes, bayes_net, loss, optimizer, device, post_prob, aleatoric=args.aleatoric, max_epoch=args.epochs)
        bayes_trainer.train()

    else:
        print("model should be ground_truth or bayes")
