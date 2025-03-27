#
from typing import Optional
#
import os
import random
#
import torch
from torch import Tensor
import torchvision
#from torchvision.io import read_image
#
from tqdm import tqdm


#
def get_random_perm(n: int, random_seed: Optional[int] = None) -> list[int]:
    #
    rd_perm: list[int] = list(range(n))
    #
    random.shuffle(rd_perm)
    #
    return rd_perm


#
class Dataset:

    #
    def __init__(self) -> None:

        #
        self.nb_train: int = 0
        self.nb_test: int = 0

        #
        self.x_train: Tensor = Tensor([0])
        self.y_train: Tensor = Tensor([0])
        #
        self.x_test: Tensor = Tensor([0])
        self.y_test: Tensor = Tensor([0])

    #
    def get_full_train(self, random_seed: Optional[int] = None) -> tuple[Tensor, Tensor]:

        #
        rd_permutation: list[int] = get_random_perm(self.nb_train)

        #
        return self.x_train[rd_permutation], self.y_train[rd_permutation]

    #
    def get_full_test(self, random_seed: Optional[int] = None) -> tuple[Tensor, Tensor]:

        #
        rd_permutation: list[int] = get_random_perm(self.nb_test)

        #
        return self.x_test[rd_permutation], self.y_test[rd_permutation]

    #
    def get_batch_train(self, batch_size: int, random_seed: Optional[int] = None) -> tuple[Tensor, Tensor]:

        #
        if batch_size > self.nb_train:

            #
            raise IndexError(f"Error : batch_size (={batch_size}) > data_size (={self.nb_train})")

        #
        rd_permutation: list[int] = get_random_perm(self.nb_train)

        #
        return self.x_train[rd_permutation[:batch_size]], self.y_train[rd_permutation[:batch_size]]

    #
    def get_batch_test(self, batch_size: int, random_seed: Optional[int] = None) -> tuple[Tensor, Tensor]:

        #
        if batch_size > self.nb_test:

            #
            raise IndexError(f"Error : batch_size (={batch_size}) > data_size (={self.nb_test})")

        #
        rd_permutation: list[int] = get_random_perm(self.nb_test)

        #
        return self.x_test[rd_permutation[:batch_size]], self.y_test[rd_permutation[:batch_size]]




#
class DatasetImages(Dataset):

    #
    def __init__(self, dataset_split_ratio: float = 0.75, seed: Optional[int] = None) -> None:

        #
        super().__init__()

        #
        self.dataset_split_ratio: float = dataset_split_ratio

        #
        self.nb_genres: int = 10
        self.nb_imgs_per_genre: int = 100

        #
        self.nb_train_per_genre: list[int] = []
        self.nb_test_per_genre: list[int] = []

        #
        self.train_images: list[str] = []
        self.train_labels: list[int] = []
        self.test_images: list[str] = []
        self.test_labels: list[int] = []

        #
        self.load_dataset()

    #
    def load_dataset(self) -> None:

        #
        self.nb_train = 0
        self.nb_test = 0
        #
        i_train: int = 0
        i_test: int = 0

        #
        base_path: str = "data/images_original/"

        #
        print("Loading dataset")

        #
        for (id_genre, genre) in enumerate(os.listdir(base_path)):

            #
            genre_path: str = f"{base_path}{genre}/"

            #
            imgs_files: list[str] = os.listdir(genre_path)

            #
            n: int = len(imgs_files)

            #
            nb_train: int = int( self.dataset_split_ratio * n )
            nb_test: int = n - nb_train

            #
            self.nb_train += nb_train
            self.nb_test += nb_test

            #
            self.nb_train_per_genre.append( nb_train )
            self.nb_test_per_genre.append( nb_test )

            #
            rd_permutation = get_random_perm(n)

            #
            for idx_img in rd_permutation[:nb_train]:

                #
                self.train_images.append( f"{genre_path}{imgs_files[idx_img]}" )
                self.train_labels.append( id_genre )

            #
            for idx_img in rd_permutation[nb_train:]:

                #
                self.test_images.append( f"{genre_path}{imgs_files[idx_img]}" )
                self.test_labels.append( id_genre )

        #
        loading_bar: tqdm.Tqdm = tqdm(total=self.nb_train + self.nb_test)

        #
        self.x_train = torch.zeros( (self.nb_train, 4, 288, 432) )
        self.y_train = torch.zeros( (self.nb_train, 1))
        self.x_test = torch.zeros( (self.nb_test, 4, 288, 432) )
        self.y_test = torch.zeros( (self.nb_test, 1))

        #
        for i_train in range(self.nb_train):

            #
            self.x_train[i_train] = torchvision.io.read_image( self.train_images[i_train] )
            self.y_train[i_train] = self.train_labels[i_train]

            #
            loading_bar.update(1)

        #
        for i_test in range(self.nb_test):

            #
            self.x_test[i_test] = torchvision.io.read_image( self.test_images[i_test] )
            self.y_test[i_test] = self.test_labels[i_test]

            #
            loading_bar.update(1)
