from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as T


class MNIST(Dataset):
    def __init__(
            self,
            root: str,
            img_size: int,
            split: str = 'train',
    ):
        assert split in ['train', 'test']
        self.img_size = img_size

        self.mnist = dset.MNIST(
            root=root,
            train=(split == 'train'),
            transform=self.get_transform(),
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, item):
        X, y = self.mnist[item]
        return X, y

    def get_transform(self):
        transform = T.Compose([
            T.Resize((self.img_size, self.img_size), antialias=True),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        return transform


if __name__ == '__main__':
    dataset = MNIST(root='~/data/MNIST/', img_size=32, split='train')
    print(len(dataset))
    dataset = MNIST(root='~/data/MNIST/', img_size=32, split='test')
    print(len(dataset))
