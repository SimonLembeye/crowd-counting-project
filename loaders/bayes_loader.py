import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader

class BayesDataset(Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.root_path = root_path
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Pour CSRNet à vérifier si besoin de modif
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        else:
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""

        """
        Les keypoints correspondent aux coordonnées des têtes
        MAIS une troisième coordonnée a été calculée lors du preprocessing des données,
        elle correspont à "dis" et semble important pour calculer pas mal de choses
        """

        wd, ht = img.size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_cropBayes(ht, wd, self.c_size, self.c_size)
        img = F2.crop(img, i, j, h, w)

        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F2.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F2.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size


#bayes
downsample_ratio = 1 # Mettre à 8 pour le réseau du répo (à 1 pour CSRNet puisque on ne modifie pas la dim avec le réseau)
data_dir = "data/bayes"
#data_dir = "/home/simon/Bureau/framework-crowd-counting/processed_data_bcc/SHHA"
#data_dir = "/Users/VictoRambaud/dev/crowd_counting2/ProcessedData/SHHA"
crop_size = 256
is_gray = False

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


def loading_data_Bayes(batch_size = 5, num_workers = 8):
    datasets_bayes = {x: BayesDataset(os.path.join(data_dir, x),
                              crop_size,
                              downsample_ratio,
                              is_gray, x) for x in ['train', 'val', 'test']}

    dataloaders_bayes = {x: DataLoader(datasets_bayes[x],
                                collate_fn=(train_collate if x == 'train' else default_collate),
                                batch_size=(batch_size if x == 'train' else 1),
                                shuffle=(True if x == 'train' else False),
                                num_workers=num_workers,
                                pin_memory=(True if x == 'train' else False))
                                for x in ['train', 'val', 'test']}

    dataloaders_bayes_test = "To do"

    return dataloaders_bayes["train"], dataloaders_bayes["val"], dataloaders_bayes["test"]
