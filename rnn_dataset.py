import torch
import torchvision
import os
import torch.utils.data as data
from PIL import Image
import numpy as np

def show_batch(batch):
    im = torchvision.utils.make_grid(batch)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    
def data_view(data):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    print('Labels: ', labels)
    print('Batch shape: ', images.size())
    show_batch(images)

       
class RMNIST(data.Dataset):  
    def __init__(self, root, time, train, transform = None, target_transform=None, download=False):
        self.root = os.path. expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.time = time
        
        
        def load_data(filename):
            data, labels = torch.load(os.path.join(self.root, 'processed', filename))
            #print (data.shape)
            lbs_dict = {}
            for i in range(10):
                lbs_index = np.where(labels == i)
                lbs_index = list(lbs_index[0])
                lbs_dict[i] = lbs_index
            new_labellist = []
            data_matrix = []  
            for i in range(10):
                tmp_list = [i] * (len(lbs_dict[i])) 
                new_labellist += tmp_list
                for j in range(len(lbs_dict[i])):
                    if self.time == 3:
                        tmp = [lbs_dict[i][j], lbs_dict[i][(j  + 1) % len(lbs_dict[i])], lbs_dict[i][(j + 2) %len(lbs_dict[i])]]
                    if self.time == 2:
                        tmp = [lbs_dict[i][j], lbs_dict[i][(j  + 1) % len(lbs_dict[i])]]
                    data_matrix.append(tmp)
            new_data = torch.zeros((len(new_labellist), self.time, 28, 28), dtype = torch.uint8)
            for i in range(len(new_labellist)):
                for j in range(self.time):
                    new_data[i,j, :, :] = data[data_matrix[i][j], :, :]
            new_label = torch.LongTensor(new_labellist)
            return new_data, new_label
    
        if self.train:
            self.data, self.labels = load_data('training.pt')
        else:
            self.data, self.labels = load_data('test.pt')
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        new_img = torch.zeros((self.time, 1, 28, 28),dtype = torch.uint8)
        for i in range(self.time):
            tmp = img[i]
            #print ('tmp type', tmp.dtype)
    
            tmp = Image.fromarray(tmp.numpy(), mode='L')


            if self.transform is not None:
                new_img[i, :, :, :] = self.transform(tmp)
            else:
                new_img = torch.unsqueeze(tmp, 1)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return new_img, target
    
    def __len__(self):
        return len(self.data)

class MMNIST(data.Dataset):
    '''
    This loader is for method comparison. Which use multi-channel to compare
    '''
    def __init__(self, root, time, train = True, transform = None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.time = time

        def load_data(filename):
            data, labels = torch.load(os.path.join(self.root, 'processed', filename))
            #print (data.shape)
            lbs_dict = {}
            for i in range(10):
                lbs_index = np.where(labels == i)
                lbs_index = list(lbs_index[0])
                lbs_dict[i] = lbs_index
            new_labellist = []
            data_matrix = []  
            for i in range(10):
                tmp_list = [i] * (len(lbs_dict[i])) 
                new_labellist += tmp_list
                for j in range(len(lbs_dict[i])):
                    if self.time == 3:
                        tmp = [lbs_dict[i][j], lbs_dict[i][(j  + 1) % len(lbs_dict[i])], lbs_dict[i][(j + 2) %len(lbs_dict[i])]]
                    if self.time == 2:
                        tmp = [lbs_dict[i][j], lbs_dict[i][(j  + 1) % len(lbs_dict[i])]]
                    data_matrix.append(tmp)
            assert len(data_matrix) == len(new_labellist)
            new_data = torch.zeros((len(new_labellist), self.time, 28, 28), dtype = torch.uint8)
            for i in range(len(new_labellist)):
                for j in range(self.time):
                    new_data[i,j, :, :] = data[data_matrix[i][j], :, :]
            new_label = torch.LongTensor(new_labellist)
            return new_data, new_label
    
        if self.train:
            self.data, self.labels = load_data('training.pt')
        else:
            self.data, self.labels = load_data('test.pt')
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        
#        print (img.shape)
        new_img = torch.zeros((self.time, 28, 28),dtype = torch.uint8)
        for i in range(self.time):
            tmp = img[i]
            #print ('tmp type', tmp.dtype)
    
            tmp = Image.fromarray(tmp.numpy(), mode='L')


            if self.transform is not None:
                new_img[i, :, :] = self.transform(tmp)
            else:
                new_img[i, :, :] = tmp

            if self.target_transform is not None:
                target = self.target_transform(target)

        return new_img, target
    
    def __len__(self):
        return len(self.data)
    

class DMNIST(data.Dataset):
    def __init__(self, root, time, train, transform = None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.time = time

        def load_data(filename):
            loaded_data = torch.load(
                os.path.join(self.root, 'processed', filename))
            if len(loaded_data) == 2:
                return loaded_data
            else:
                clsname, data, labels = loaded_data
                if clsname != type(self).__name__:
                    raise RuntimeError("Expected {} data but found {}"
                                       .format(type(self).__name__, clsname, ))
                return data, labels

        if self.train:
            self.data, self.labels = load_data('training.pt')
        else:
            self.data, self.labels = load_data('test.pt')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.labels[index]
        else:
            img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        #print (img.dtype)
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        img_np = np.zeros((self.time, 1, 28, 28))
        for i in range(self.time):
            img_np[i,:, :, :] = img

        return img_np, target

    def __len__(self):
        return len(self.data)
    
class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str