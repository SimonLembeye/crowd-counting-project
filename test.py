import argparse
import numpy as np

from loaders import loading_data_GT, loading_data_Bayes
from models import CSRNet, vgg19_extended

def test_gt(gt_net, aleatoric=False):
    _, _, test_dataloader = loading_data_GT()

    gt_net.load_state_dict(torch.load(os.path.join(save_dir, 'best_model_gt.pth'), device))
    gt_net.eval()
    errors = []

    for vi, data in enumerate(test_dataloader, 0):
        img, gt_map = data

        with torch.no_grad():
            img = Variable(img).to(device)
            assert img.size(0) == 1
            gt_map = Variable(gt_map).to(device)
            pred_density_map = gt_net(img)
            pred_cnt = int(gt_map[0].sum().data / LOG_PARA)
            gt_count = int(pred_density_map[0].sum().data/LOG_PARA)
            error = gt_count - pred_cnt
            print(vi, error, gt_count, pred_cnt)

            errors.append(error)


    errors = np.array(errors)
    mse = np.sqrt(np.mean(np.square(errors)))
    mae = np.mean(np.abs(errors))

    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)

def test_bayes(bayes_net, aleatoric=False):
    _, _, test_dataloader = loading_data_Bayes()

    bayes_net.load_state_dict(torch.load(os.path.join(save_dir, 'best_model_bayes.pth'), device))
    errors = []

    for inputs, count, name in test_dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1
        with torch.set_grad_enabled(False):
            outputs = bayes_net(inputs)
            error = count[0].item() - torch.sum(outputs).item()
            print(name, error, count[0].item(), torch.sum(outputs).item())
            errors.append(error)

    errors = np.array(errors)
    mse = np.sqrt(np.mean(np.square(errors)))
    mae = np.mean(np.abs(errors))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--method', default='ground_truth',
                        help='Method - ground_truth or bayes')
    parser.add_argument('--model', default='csrnet',
                        help='NN Model - csrnet or vgg19_extended')
    parser.add_argument('--aleatoric', type=str2bool, default='false',
                        help='Boolean - use aleatoric loss')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.method == "ground_truth":
        gt_net = CSRNet().to(device)
        test_gt(gt_net, aleatoric=args.aleatoric)

    elif args.method == "bayes":

        if args.model == "csrnet":
            bayes_net = CSRNet().to(device)
        elif args.model == "vgg19_extended":
            bayes_net = vgg19().to(device)
        else:
            print('model should be csrnet or vgg19_extended')

        test_bayes(bayes_net, aleatoric=args.aleatoric)

    else:
        print("model should be ground_truth or bayes")
