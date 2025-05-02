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
from lib_device import get_device

#
from lib_plot import plot_rgb_image

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
    def __init__(self, dataset_split_ratio: float = 0.75, seed: Optional[int] = None, load_from_path: str = "data/images_original/") -> None:

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
        if load_from_path != "":
            #
            print(f"Load dataset from path : `{load_from_path}`")
            self.load_dataset( load_from_path )

    #
    def load_dataset(self, base_path: str) -> None:

        #
        self.nb_train = 0
        self.nb_test = 0
        #
        i_train: int = 0
        i_test: int = 0


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
    def __init__(self, px_height_to_keep: int, divisible_per: int = 18, dataset_split_ratio: float = 0.75, seed: Optional[int] = None, load_from_path: str = "data/images_original/") -> None:

        #
        super().__init__()

        #
        self.px_height_to_keep: int = px_height_to_keep
        self.divisible_per: int = divisible_per

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
        if load_from_path != "":
            print(f"Load dataset from path : `{load_from_path}`")
            self.load_dataset( load_from_path )

    #
    def load_dataset(self, base_path: str) -> None:

        #
        self.nb_train = 0
        self.nb_test = 0
        #
        i_train: int = 0
        i_test: int = 0

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
        byt: int = 388
        bxt: int = 251
        by: int = 54
        bx: int = 35
        # by = byt - self.px_height_to_keep

        #
        if self.px_height_to_keep > 0:

            bx = bxt - self.px_height_to_keep

        bxt -= (bxt - bx) % self.divisible_per
        byt -= (byt - by) % self.divisible_per

        w: int = bxt - bx
        h: int = byt - by

        #
        # print(f"DEBUG | w = {w} | h = {h} | bx = {bx} | bxt = {bxt} | by = {by} | byt = {byt}")

        #
        self.x_train = torch.zeros( (self.nb_train, 3, w, h), dtype=torch.float32 )
        self.y_train = torch.zeros( (self.nb_train, 1), dtype=torch.float32)
        self.x_test = torch.zeros( (self.nb_test, 3, w, h), dtype=torch.float32 )
        self.y_test = torch.zeros( (self.nb_test, 1), dtype=torch.float32)

        #
        for i_train in range(self.nb_train):

            #
            img = torchvision.io.read_image( self.train_images[i_train] )

            #
            # print(f"DEBUG | {img.shape}")

            #
            # plot_rgb_image(img[:3, :, :].permute(1, 2, 0).numpy())

            #
            self.x_train[i_train] = img[:3, bx:bxt, by:byt] / 255.0
            self.y_train[i_train] = self.train_labels[i_train]

            #
            # plot_rgb_image(self.x_train[i_train].permute(1, 2, 0).numpy())

            #
            loading_bar.update(1)

        #
        for i_test in range(self.nb_test):

            #
            self.x_test[i_test] = torchvision.io.read_image( self.test_images[i_test] )[:3, bx:bxt, by:byt] / 255.0
            self.y_test[i_test] = self.test_labels[i_test]

            #
            loading_bar.update(1)




class DatasetAudios(Dataset):

    #
    def __init__(self, dataset_split_ratio: float = 0.75, seed: Optional[int] = None, load_from_path: str = "data/genres_original/") -> None:

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
        if load_from_path != "":
            print(f"Load dataset from path : `{load_from_path}`")
            self.load_dataset( load_from_path )

    #
    def load_dataset(self, base_path: str) -> None:

        #
        self.nb_train = 0
        self.nb_test = 0
        #
        i_train: int = 0
        i_test: int = 0

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




#
def create_audio2vec_signal_dataset(in_dataset: DatasetAudios = DatasetAudios(), prob_self: float = 0.45, nb_train: int = 750, nb_test: int = 250) -> DatasetAudios:

    #
    result_dataset: DatasetAudios = DatasetAudios(load_from_path="")

    #
    print(f"DEBUG | train dataset sample shape : {in_dataset.x_train.shape}")

    #
    dim_x: int = in_dataset.x_test.shape[1]
    dtype = in_dataset.x_test.dtype

    #
    in_dataset.x_test = in_dataset.x_test.to( get_device() )
    in_dataset.y_test = in_dataset.y_test.to( "cpu" )
    in_dataset.x_train = in_dataset.x_train.to( get_device() )
    in_dataset.y_train = in_dataset.y_train.to( "cpu" )

    #
    result_dataset.x_test = torch.zeros( (nb_test, dim_x), dtype=dtype ).to( get_device() )
    result_dataset.y_test = torch.zeros( (nb_test, 1), dtype=torch.int ).to( "cpu" )
    result_dataset.x_train = torch.zeros( (nb_train, dim_x), dtype=dtype ).to( get_device() )
    result_dataset.y_train = torch.zeros( (nb_train, 1), dtype=torch.int ).to( "cpu" )
    #
    result_dataset.nb_train = nb_train
    result_dataset.nb_test = nb_test
    #
    result_dataset.class_names = ["2 different songs", "1 same song"]

    #
    X1: Tensor
    X2: Tensor

    #
    p: float
    dx1: int
    dx2: int
    i1: int
    i2: int

    #
    len_train: int = len(in_dataset.x_train) - 1
    len_test: int = len(in_dataset.x_test) - 1

    #
    print("Preparing train...")
    #
    for i in tqdm( range(nb_train) ):
        #
        if random.random() < prob_self:
            #
            X1 = in_dataset.x_train[ random.randint( 0, len_train ) ]
            #
            result_dataset.x_train[i] = X1
            result_dataset.y_train[i] = 1
        #
        else:
            #
            i1 = random.randint( 0, len_train )
            i2 = random.randint( 0, len_train )
            #
            X1 = in_dataset.x_train[ i1 ]
            X2 = in_dataset.x_train[ i2 ]
            #
            p = float( random.randint(40, 60) ) / 100.0
            #
            dx1 = int( dim_x * p )
            dx2 = dim_x - dx1
            #
            result_dataset.x_train[i][0: dx1] = X1[0: dx1]
            result_dataset.x_train[i][dx1: dx1+dx2] = X2[dx1: dx1+dx2]
            result_dataset.y_train[i] = 1 if i1 == i2 else 0

    #
    print("Preparing test...")
    #
    for i in tqdm( range(nb_test) ):
        #
        if random.random() < prob_self:
            #
            X1 = in_dataset.x_test[ random.randint( 0, len_test ) ]
            #
            result_dataset.x_test[i] = X1
            result_dataset.y_test[i] = 1
        #
        else:
            #
            i1 = random.randint( 0, len_test )
            i2 = random.randint( 0, len_test )
            #
            X1 = in_dataset.x_test[ i1 ]
            X2 = in_dataset.x_test[ i2 ]
            #
            p = float( random.randint(40, 60) ) / 100.0
            #
            dx1 = int( dim_x * p )
            dx2 = dim_x - dx1
            #
            result_dataset.x_test[i][0: dx1] = X1[0: dx1]
            result_dataset.x_test[i][dx1: dx1+dx2] = X2[dx1: dx1+dx2]
            result_dataset.y_test[i] = 1 if i1 == i2 else 0

    #
    in_dataset.x_test = in_dataset.x_test.to( "cpu" )
    in_dataset.y_test = in_dataset.y_test.to( "cpu" )
    in_dataset.x_train = in_dataset.x_train.to( get_device() )
    in_dataset.y_train = in_dataset.y_train.to( "cpu" )

    #
    result_dataset.x_test = torch.zeros( (nb_test, dim_x), dtype=dtype ).to( "cpu" )
    result_dataset.y_test = torch.zeros( (nb_test, 2), dtype=torch.int ).to( "cpu" )
    result_dataset.x_train = torch.zeros( (nb_train, dim_x), dtype=dtype ).to( "cpu" )
    result_dataset.y_train = torch.zeros( (nb_train, 2), dtype=torch.int ).to( "cpu" )

    #
    return result_dataset







#
def create_audio2vec_img_dataset(in_dataset: DatasetImages | DatasetImagesFiltered = DatasetImages(), prob_self: float = 0.45, nb_train: int = 5000, nb_test: int = 200) -> DatasetImages:

    #
    result_dataset: DatasetImages = DatasetImages(load_from_path="")

    #
    print(f"DEBUG | train dataset sample shape : {in_dataset.x_train.shape}")

    #
    dim_x: int = in_dataset.x_test.shape[1]
    dim_y: int = in_dataset.x_test.shape[2]
    dim_c: int = in_dataset.x_test.shape[3]
    dtype = in_dataset.x_test.dtype

    #
    in_dataset.x_test = in_dataset.x_test.to( get_device() )
    in_dataset.y_test = in_dataset.y_test.to( "cpu" )
    in_dataset.x_train = in_dataset.x_train.to( get_device() )
    in_dataset.y_train = in_dataset.y_train.to( "cpu" )

    #
    result_dataset.x_test = torch.zeros( (nb_test, dim_x, dim_y, dim_c), dtype=dtype ).to( get_device() )
    result_dataset.y_test = torch.zeros( (nb_test, 1), dtype=torch.int ).to( "cpu" )
    result_dataset.x_train = torch.zeros( (nb_train, dim_x, dim_y, dim_c), dtype=dtype ).to( get_device() )
    result_dataset.y_train = torch.zeros( (nb_train, 1), dtype=torch.int ).to( "cpu" )
    #
    result_dataset.nb_train = nb_train
    result_dataset.nb_test = nb_test
    #
    result_dataset.class_names = ["2 different songs", "1 same song"]

    #
    X1: Tensor
    X2: Tensor

    #
    p: float
    dx1: int
    dx2: int
    i1: int
    i2: int

    #
    len_train: int = len(in_dataset.x_train) - 1
    len_test: int = len(in_dataset.x_test) - 1

    #
    print("Preparing train...")
    #
    for i in tqdm( range(nb_train) ):
        #
        if random.random() < prob_self:
            #
            X1 = in_dataset.x_train[ random.randint( 0, len_train ) ]
            #
            result_dataset.x_train[i] = X1
            result_dataset.y_train[i] = 1
        #
        else:
            #
            i1 = random.randint( 0, len_train )
            i2 = random.randint( 0, len_train )
            #
            X1 = in_dataset.x_train[ i1 ]
            X2 = in_dataset.x_train[ i2 ]
            #
            p = float( random.randint(40, 60) ) / 100.0
            #
            dx1 = int( dim_x * p )
            dx2 = dim_x - dx1
            #
            result_dataset.x_train[i, 0: dx1, :, :] = X1[0: dx1, :, :]
            result_dataset.x_train[i, dx1: dx1+dx2, :, :] = X2[dx1: dx1+dx2, :, :]
            result_dataset.y_train[i] = 1 if i1 == i2 else 0

    #
    print("Preparing test...")
    #
    for i in tqdm( range(nb_test) ):
        #
        if random.random() < prob_self:
            #
            X1 = in_dataset.x_test[ random.randint( 0, len_test ) ]
            #
            result_dataset.x_test[i] = X1
            result_dataset.y_test[i] = 1
        #
        else:
            #
            i1 = random.randint( 0, len_test )
            i2 = random.randint( 0, len_test )
            #
            X1 = in_dataset.x_test[ i1 ]
            X2 = in_dataset.x_test[ i2 ]
            #
            p = float( random.randint(40, 60) ) / 100.0
            #
            dx1 = int( dim_x * p )
            dx2 = dim_x - dx1
            #
            result_dataset.x_test[i, 0: dx1, :, :] = X1[0: dx1, :, :]
            result_dataset.x_test[i, dx1: dx1+dx2, :, :] = X2[dx1: dx1+dx2, :, :]
            result_dataset.y_test[i] = 1 if i1 == i2 else 0

    #
    return result_dataset




