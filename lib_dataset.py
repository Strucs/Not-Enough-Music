#
from typing import Optional
#
import os
import random
#
import torch
from torch import Tensor
import torchvision  # type: ignore
import torchaudio  # type: ignore
#from torchvision.io import read_image
#
from tqdm import tqdm  # type: ignore


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
        self.class_names: list[str] = []

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
        self.class_names = [path for path in os.listdir(base_path) if not path.startswith(".")]

        #
        for (id_genre, genre) in enumerate(self.class_names):

            #
            genre_path: str = f"{base_path}{genre}/"

            #
            img_files: list[str] = [path for path in os.listdir(genre_path) if not path.startswith(".") and path.endswith(".png")]

            #
            n: int = len(img_files)

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
                self.train_images.append( f"{genre_path}{img_files[idx_img]}" )
                self.train_labels.append( id_genre )

            #
            for idx_img in rd_permutation[nb_train:]:

                #
                self.test_images.append( f"{genre_path}{img_files[idx_img]}" )
                self.test_labels.append( id_genre )

        #
        loading_bar: tqdm.Tqdm = tqdm(total=self.nb_train + self.nb_test)

        #
        self.x_train = torch.zeros( (self.nb_train, 3, 234, 216) )
        self.y_train = torch.zeros( (self.nb_train, 1))
        self.x_test = torch.zeros( (self.nb_test, 3, 234, 216) )
        self.y_test = torch.zeros( (self.nb_test, 1))

        #
        for i_train in range(self.nb_train):

            #
            self.x_train[i_train] = torchvision.io.read_image( self.train_images[i_train] )[:3, 54:388, 35:251]
            self.y_train[i_train] = self.train_labels[i_train]

            #
            loading_bar.update(1)

        #
        for i_test in range(self.nb_test):

            #
            self.x_test[i_test] = torchvision.io.read_image( self.test_images[i_test] )[:3, 54:388, 35:251]
            self.y_test[i_test] = self.test_labels[i_test]

            #
            loading_bar.update(1)


#
class DatasetImagesFiltered(Dataset):

    #
    def __init__(self, px_height_to_keep: int, dataset_split_ratio: float = 0.75, seed: Optional[int] = None) -> None:

        #
        super().__init__()

        #
        self.px_height_to_keep: int = px_height_to_keep

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
        self.class_names: list[str] = []

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
        self.class_names = [path for path in os.listdir(base_path) if not path.startswith(".")]

        #
        for (id_genre, genre) in enumerate(self.class_names):

            #
            genre_path: str = f"{base_path}{genre}/"

            #
            img_files: list[str] = [path for path in os.listdir(genre_path) if not path.startswith(".") and path.endswith(".png")]

            #
            n: int = len(img_files)

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
                self.train_images.append( f"{genre_path}{img_files[idx_img]}" )
                self.train_labels.append( id_genre )

            #
            for idx_img in rd_permutation[nb_train:]:

                #
                self.test_images.append( f"{genre_path}{img_files[idx_img]}" )
                self.test_labels.append( id_genre )

        #
        loading_bar: tqdm.Tqdm = tqdm(total=self.nb_train + self.nb_test)

        #
        bxt: int = 388
        byt: int = 251
        bx: int = 54
        by: int = 35
        by = 251 - self.px_height_to_keep

        w: int = bxt - bx
        h: int = byt - by

        #
        self.x_train = torch.zeros( (self.nb_train, 3, w, h) )
        self.y_train = torch.zeros( (self.nb_train, 1))
        self.x_test = torch.zeros( (self.nb_test, 3, w, h) )
        self.y_test = torch.zeros( (self.nb_test, 1))

        #
        for i_train in range(self.nb_train):

            #
            self.x_train[i_train] = torchvision.io.read_image( self.train_images[i_train] )[:3, bx:bxt, by:byt]
            self.y_train[i_train] = self.train_labels[i_train]

            #
            loading_bar.update(1)

        #
        for i_test in range(self.nb_test):

            #
            self.x_test[i_test] = torchvision.io.read_image( self.test_images[i_test] )[:3, 54:388, 35:251]
            self.y_test[i_test] = self.test_labels[i_test]

            #
            loading_bar.update(1)




class DatasetAudios(Dataset):

    #
    def __init__(self, dataset_split_ratio: float = 0.75, seed: Optional[int] = None) -> None:

        #
        super().__init__()

        #
        self.dataset_split_ratio: float = dataset_split_ratio

        #
        self.nb_genres: int = 10
        self.nb_audios_per_genre: int = 100

        #
        self.nb_train_per_genre: list[int] = []
        self.nb_test_per_genre: list[int] = []

        #
        self.train_audios: list[str] = []
        self.train_labels: list[int] = []
        self.test_audios: list[str] = []
        self.test_labels: list[int] = []

        #
        self.classes_names: list[str] = []

        #
        self.sampling_rate: int = -1

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
        base_path: str = "data/genres_original/"

        #
        print("Loading dataset")

        #
        self.class_names = [path for path in os.listdir(base_path) if not path.startswith(".")]

        #
        for (id_genre, genre) in enumerate(self.class_names):

            #
            genre_path: str = f"{base_path}{genre}/"

            #
            audio_files: list[str] = [path for path in os.listdir(genre_path) if not path.startswith(".") and path.endswith(".wav")]

            #
            n: int = len(audio_files)

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
                self.train_audios.append( f"{genre_path}{audio_files[idx_img]}" )
                self.train_labels.append( id_genre )

            #
            for idx_img in rd_permutation[nb_train:]:

                #
                self.test_audios.append( f"{genre_path}{audio_files[idx_img]}" )
                self.test_labels.append( id_genre )

        #
        loading_bar: tqdm.Tqdm = tqdm(total=self.nb_train + self.nb_test)

        #
        self.x_train = torch.zeros( (self.nb_train, 675808) )
        self.y_train = torch.zeros( (self.nb_train, 1))
        self.x_test = torch.zeros( (self.nb_test, 675808) )
        self.y_test = torch.zeros( (self.nb_test, 1))

        #
        for i_train in range(self.nb_train):

            #
            a, self.sampling_rate = torchaudio.load( self.train_audios[i_train], format="wav")
            self.x_train[i_train][:a.shape[1]] = a[0]
            self.y_train[i_train] = self.train_labels[i_train]

            #
            loading_bar.update(1)

        #
        for i_test in range(self.nb_test):

            #
            a, self.sampling_rate = torchaudio.load( self.test_audios[i_test] )
            self.x_test[i_test][:a.shape[1]] = a[0]
            self.y_test[i_test] = self.test_labels[i_test]

            #
            loading_bar.update(1)