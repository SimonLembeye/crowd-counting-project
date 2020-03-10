# Crowd Counting Project
The aim of this project is to test different methods of crowd couting. We will find an implementation of the neural network and losses developed by the articles:
    - [1] CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes, CVPR 2018. [link](https://arxiv.org/abs/1802.10062 )
    - [2] Bayesian Loss for Crowd Count Estimation with Point Supervision. [link](https://arxiv.org/abs/1908.03684)
    - [3] What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? [link](https://arxiv.org/abs/1703.04977 )

## Data

These different methods have been tested on the ShanghaïTech Part A dataset.
We provide the well formatted data for this project [here](https://drive.google.com/open?id=1E3cc8yNBL5vZ8XGVOKuWY9w8BpO4EVbK).

## Project structure

The project is structured as following:

```code
.
├── loaders
|   └──  bayes_loader.py # Dataset and loader of the bayes method
|   └──  gt_laoder.py # Dataset and loader of the ground truth method
├── models
|   └──  csrnet.py # csrnet nn
|   └──  vgg19_extended.py # vgg19 with an extension
├──  losses
|   └──  aleatoric.py #  aleatoric  loss implementation
|   └──  bayes_loss.py
|   └──  port_prob.py # posterior probability of the bayes method
├──  trainers
|   └──  abctrainer.py # abstract class defining a trainer
|   └──  bayes_trainer.py
|   └──  gt_trainer.py
├──  utils
|   └──  loaders.py # helper functions for the loaders
|   └──  models.py # helper functions for the models
|   └──  parser.py
├── train.py # pipelines for training
├── test.y # pipelines for testing
```

## Launching

Train exemple :
`python commander.py --dataset gtsrb --name gtsrb_lenet_optsgd_lr1e-3_lrdecayPlateau0.5_bsz128 --batch-size 128 --optimizer sgd --scheduler ReduceLROnPlateau --lr 1e-3 --lr-decay 0.5 --step 15 --epochs 100 --arch lenet --model-name lenet5 --root-dir /data/ --num-classes 43 --workers 4 --crop-size 32 --criterion crossentropy --tensorboard`
Details can be found in train.py

Test exemple :
`python commander.py --dataset gtsrb --name gtsrb_lenet_optsgd_lr1e-3_lrdecayPlateau0.5_bsz128 --batch-size 128 --optimizer sgd --scheduler ReduceLROnPlateau --lr 1e-3 --lr-decay 0.5 --step 15 --epochs 100 --arch lenet --model-name lenet5 --root-dir /data/ --num-classes 43 --workers 4 --crop-size 32 --criterion crossentropy --tensorboard`
Details can be found in test.py

## Output

For each training session the weights of the best models (minimizing 2 * val_mse + val_mae) are stored in the best_model_weight folder. Everything else is printed.

### Tensorboard
In order the visualize metrics and results in tensorboard you need to launch it separately: `tensorboard --logdir=runs`. You can then access tensorboard in our browser at [localhost:6006](localhost:6006)
If you have performed multiple experiments, tensorboard will aggregate them in the same dashboard.
