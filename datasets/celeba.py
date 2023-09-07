from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class CelebA(Dataset):
    def __init__(
            self,
            root: str,
            img_size: int,
            split: str = 'train',
            transform_type: str = 'default',
    ):
        assert split in ['train', 'valid', 'test', 'all']
        self.img_size = img_size
        self.split = split
        self.transform_type = transform_type
        self.transform = self.get_transform()

        self.celeba = dset.CelebA(root=root, split=split)

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, item):
        X, _ = self.celeba[item]
        if self.transform is not None:
            X = self.transform(X)
        return X

    def get_transform(self):
        transform = None
        flip_p = 0.5 if self.split in ['train', 'all'] else 0.0
        if self.transform_type in ['default', 'stylegan-like']:
            # https://github.com/NVlabs/stylegan/blob/master/dataset_tool.py#L484-L499
            cx, cy = 89, 121
            transform = T.Compose([
                T.Lambda(lambda x: TF.crop(x, top=cy-64, left=cx-64, height=128, width=128)),
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'resize':
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'crop140':
            transform = T.Compose([
                T.CenterCrop((140, 140)),
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        return transform


if __name__ == '__main__':
    dataset = CelebA(root='/data/CelebA/', img_size=64, split='train')
    print(len(dataset))
    dataset = CelebA(root='/data/CelebA/', img_size=64, split='valid')
    print(len(dataset))
    dataset = CelebA(root='/data/CelebA/', img_size=64, split='test')
    print(len(dataset))
    dataset = CelebA(root='/data/CelebA/', img_size=64, split='all')
    print(len(dataset))
