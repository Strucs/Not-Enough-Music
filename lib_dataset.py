#
from typing import Optional, Callable
#
import os
import random
#
import torch
from torch import Tensor
import torchvision  # type: ignore
import torchaudio  # type: ignore
import torchaudio.transforms as T  # type: ignore
from torch.utils.data import Dataset as TorchDataset
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
    def get_full_train(self) -> tuple[Tensor, Tensor]:

        #
        rd_permutation: list[int] = get_random_perm(self.nb_train)

        #
        return self.x_train[rd_permutation], self.y_train[rd_permutation]

    #
    def get_full_test(self) -> tuple[Tensor, Tensor]:

        #
        rd_permutation: list[int] = get_random_perm(self.nb_test)

        #
        return self.x_test[rd_permutation], self.y_test[rd_permutation]

    #
    def get_batch_train(self, batch_size: int) -> tuple[Tensor, Tensor]:

        #
        if batch_size > self.nb_train:

            #
            raise IndexError(f"Error : batch_size (={batch_size}) > data_size (={self.nb_train})")

        #
        rd_permutation: list[int] = get_random_perm(self.nb_train)

        #
        return self.x_train[rd_permutation[:batch_size]], self.y_train[rd_permutation[:batch_size]]

    #
    def get_batch_test(self, batch_size: int) -> tuple[Tensor, Tensor]:

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
    def __init__(self, dataset_split_ratio: float = 0.75, load_from_path: str = "data/images_original/") -> None:

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
        print(f"Load dataset from path : `{base_path}`...")

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
    def __init__(self, px_height_to_keep: int, divisible_per: int = 18, dataset_split_ratio: float = 0.75, load_from_path: str = "data/images_original/") -> None:

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
            #
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
        print(f"Load dataset from path : `{base_path}`...")

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

        #
        if self.px_height_to_keep > 0:

            bx = bxt - self.px_height_to_keep

        bxt -= (bxt - bx) % self.divisible_per
        byt -= (byt - by) % self.divisible_per

        w: int = bxt - bx
        h: int = byt - by

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
            self.x_train[i_train] = img[:3, bx:bxt, by:byt] / 255.0
            self.y_train[i_train] = self.train_labels[i_train]

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
    def __init__(self, dataset_split_ratio: float = 0.75, load_from_path: str = "data/genres_original/") -> None:

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
            #
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
        print(f"Load dataset from path : `{base_path}`...")

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
def create_audio2vec_signal_dataset(in_dataset: DatasetAudios, prob_self: float = 0.45, nb_train: int = 1000, nb_test: int = 250) -> DatasetAudios:

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
            result_dataset.x_train[i, 0: dx1] = X1[0: dx1]
            result_dataset.x_train[i, dx1: dx1+dx2] = X2[dx1: dx1+dx2]
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
            result_dataset.x_test[i, 0: dx1] = X1[0: dx1]
            result_dataset.x_test[i, dx1: dx1+dx2] = X2[dx1: dx1+dx2]
            result_dataset.y_test[i] = 1 if i1 == i2 else 0

    #
    in_dataset.x_test = in_dataset.x_test.to( "cpu" )
    in_dataset.y_test = in_dataset.y_test.to( "cpu" )
    in_dataset.x_train = in_dataset.x_train.to( "cpu" )
    in_dataset.y_train = in_dataset.y_train.to( "cpu" )

    #
    result_dataset.x_test = result_dataset.x_test.to( "cpu" )
    result_dataset.y_test = result_dataset.y_test.to( "cpu" )
    result_dataset.x_train = result_dataset.x_train.to( "cpu" )
    result_dataset.y_train = result_dataset.y_train.to( "cpu" )

    #
    return result_dataset







#
def create_audio2vec_img_dataset(in_dataset: DatasetImages | DatasetImagesFiltered, prob_self: float = 0.45, nb_train: int = 5000, nb_test: int = 200) -> DatasetImages:

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
    in_dataset.x_test = in_dataset.x_test.to( "cpu" )
    in_dataset.y_test = in_dataset.y_test.to( "cpu" )
    in_dataset.x_train = in_dataset.x_train.to( "cpu" )
    in_dataset.y_train = in_dataset.y_train.to( "cpu" )

    #
    result_dataset.x_test = result_dataset.x_test.to( "cpu" )
    result_dataset.y_test = result_dataset.y_test.to( "cpu" )
    result_dataset.x_train = result_dataset.x_train.to( "cpu" )
    result_dataset.y_train = result_dataset.y_train.to( "cpu" )

    #
    return result_dataset


"""
# --- Define Your Augmentation Pipeline ---
# Example for raw audio. Adapt as needed for spectrograms using torchvision.transforms
# You might need more sophisticated or combined augmentations
class AudioAugmentation:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        # Define a set of potential transformations
        self.transforms = [
            T.Vol(gain=random.uniform(0.5, 1.5), gain_type="amplitude"), # Random Gain
            T.PitchShift(sample_rate=sample_rate, n_steps=random.uniform(-2, 2)), # Pitch Shift
            # Add more transforms like TimeStretch, Noise addition, etc.
            # You can use torchaudio.prototype.functional for experimental features like noise
        ]
        # Simple time cropping / padding example
        self.target_length = 675808 # Match your original length

    def _apply_random_transforms(self, waveform: Tensor) -> Tensor:
        # Apply a subset of transforms randomly
        num_transforms_to_apply = random.randint(1, len(self.transforms))
        chosen_transforms = random.sample(self.transforms, num_transforms_to_apply)

        augmented_waveform = waveform
        for transform in chosen_transforms:
             # Need error handling in case a transform fails
            try:
                augmented_waveform = transform(augmented_waveform)
            except Exception as e:
                print(f"Warning: Transform {type(transform).__name__} failed: {e}")
                # Optionally skip this transform or handle differently
                pass # Continue with the next transform

        return augmented_waveform

    def _crop_or_pad(self, waveform: Tensor) -> Tensor:
         current_length = waveform.size(-1)
         if current_length > self.target_length:
             # Random crop
             start = random.randint(0, current_length - self.target_length)
             return waveform[..., start:start + self.target_length]
         elif current_length < self.target_length:
             # Random pad (e.g., with zeros)
             padding_needed = self.target_length - current_length
             pad_left = random.randint(0, padding_needed)
             pad_right = padding_needed - pad_left
             # Assuming mono audio (shape [channels, time])
             return torch.nn.functional.pad(waveform, (pad_left, pad_right))
         else:
            return waveform


    def __call__(self, waveform: Tensor) -> Tensor:
        # 1. Apply standard transforms
        augmented = self._apply_random_transforms(waveform)
        # 2. Apply cropping/padding
        augmented = self._crop_or_pad(augmented)
        return augmented
"""


# --- Improved Audio Augmentation (Time Domain) ---
class AudioAugmentation:
    def __init__(self, sample_rate: int, target_length: int = 675808):
        self.sample_rate = sample_rate
        self.target_length = target_length # Match your original length

        # Define a set of potential time-domain transformations with probabilities
        self.transforms = [
            (T.Vol(gain=random.uniform(0.2, 1.5), gain_type="amplitude"), 0.8), # Random Gain (80% chance)
            (T.PitchShift(sample_rate=sample_rate, n_steps=random.uniform(-4, 4)), 0.5), # Pitch Shift (50% chance)
            (T.TimeStretch(hop_length=128, n_freq=201, rate=random.uniform(0.8, 1.2)), 0.5), # Time Stretch (50% chance)
            (T.Roll(rolls=(int(random.randint(-self.target_length // 10, self.target_length // 10)),)), 0.3), # Random Roll (30% chance)
            # Add white noise
            (lambda w: w + torch.randn_like(w) * 0.005 * random.uniform(0.5, 1.5), 0.5), # Add Noise (50% chance)
            # Polarity Inversion
            (lambda w: w * -1 if random.random() > 0.5 else w, 0.2), # Polarity Inversion (20% chance)
            # Add more transforms as needed
        ]

    def _apply_random_transforms(self, waveform: Tensor) -> Tensor:
        augmented_waveform = waveform.clone() # Work on a copy

        for transform, prob in self.transforms:
            if random.random() < prob:
                try:
                    if isinstance(transform, T.TimeStretch):
                         # TimeStretch might change length, apply before fixed-length processing
                         augmented_waveform = transform(augmented_waveform.unsqueeze(0)).squeeze(0) # TimeStretch expects [channel, time]
                    elif isinstance(transform, T.PitchShift):
                         # PitchShift also expects [channel, time]
                         augmented_waveform = transform(augmented_waveform.unsqueeze(0), int(random.uniform(-4, 4))).squeeze(0) # Pass n_steps dynamically
                    elif isinstance(transform, T.Vol):
                         augmented_waveform = transform(augmented_waveform, random.uniform(0.2, 1.5)).squeeze(0) # Pass gain dynamically
                    elif isinstance(transform, T.Roll):
                         augmented_waveform = transform(augmented_waveform, (int(random.randint(-self.target_length // 10, self.target_length // 10)),)).squeeze(0) # Pass rolls dynamically
                    else:
                         # Assume other transforms work on [time] or [channel, time] and handle accordingly
                         augmented_waveform = transform(augmented_waveform)

                except Exception as e:
                    print(f"Warning: Transform {type(transform).__name__} failed: {e}")
                    # Optionally skip this transform or handle differently
                    pass # Continue with the next transform

        return augmented_waveform

    def _crop_or_pad(self, waveform: Tensor) -> Tensor:
         current_length = waveform.size(-1)
         if current_length > self.target_length:
             # Random crop
             start = random.randint(0, current_length - self.target_length)
             return waveform[start:start + self.target_length] # Assuming shape is [time]
         elif current_length < self.target_length:
             # Random pad (e.g., with zeros)
             padding_needed = self.target_length - current_length
             pad_left = random.randint(0, padding_needed)
             pad_right = padding_needed - pad_left
             # Assuming mono audio (shape [time])
             return F.pad(waveform, (pad_left, pad_right))
         else:
            return waveform


    def __call__(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform (Tensor): Input waveform, expected shape [time].
        Returns:
            Tensor: Augmented and length-adjusted waveform, shape [time].
        """
        # Ensure input is 1D [time] or [channel, time]
        if waveform.ndim == 2 and waveform.shape[0] == 1: # [1, time]
             waveform = waveform.squeeze(0) # Convert to [time]
        elif waveform.ndim != 1:
             raise ValueError(f"Expected waveform shape [time] or [1, time], but got {waveform.shape}")


        # Apply random transforms
        augmented = self._apply_random_transforms(waveform)

        # Apply cropping/padding to target length
        augmented = self._crop_or_pad(augmented)

        return augmented



# --- Image Spectrogram Augmentation ---
class ImageSpectrogramAugmentation:
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: int = 0,
        f_max: Optional[int] = None,
        target_spectrogram_shape: tuple[int, int] = (128, 1024), # (n_mels, time_frames)
        time_mask_param: int = 80, # Max width of time masks
        freq_mask_param: int = 40, # Max width of freq masks
        num_time_masks: int = 1, # Number of time masks
        num_freq_masks: int = 1 # Number of freq masks
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.target_spectrogram_shape = target_spectrogram_shape # (height, width) or (n_mels, time_frames)

        # Spectrogram transformation pipeline
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0, # Power of 2 for Mel spectrogram (magnitude squared)
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0) # Convert to dB scale

        # Spectrogram Augmentations (SpecAugment style)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)

        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks


    def _apply_spectrogram_augmentations(self, spectrogram: Tensor) -> Tensor:
        augmented_spectrogram = spectrogram.clone() # Work on a copy

        # Apply time masking
        for _ in range(self.num_time_masks):
            augmented_spectrogram = self.time_masking(augmented_spectrogram)

        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            augmented_spectrogram = self.freq_masking(augmented_spectrogram)

        # Add other spectrogram specific augmentations here if needed
        # e.g., Mixup on spectrograms (more complex)

        return augmented_spectrogram


    def _crop_or_pad_spectrogram(self, spectrogram: Tensor) -> Tensor:
        """
        Crops or pads the spectrogram to the target shape.
        Expected input shape: [channels, n_mels, time_frames] (after mel_spectrogram and db)
        Target shape: [channels, target_n_mels, target_time_frames]
        """
        # Assuming input spectrogram shape is [1, n_mels, time_frames]
        current_n_mels, current_time_frames = spectrogram.shape[-2:]
        target_n_mels, target_time_frames = self.target_spectrogram_shape

        # Pad/crop frequency dimension (n_mels) - usually not needed if n_mels is fixed
        if current_n_mels != target_n_mels:
            # Simple padding or raise error if dimensions mismatch unexpectedly
            if current_n_mels < target_n_mels:
                pad_n_mels = target_n_mels - current_n_mels
                spectrogram = F.pad(spectrogram, (0, 0, 0, pad_n_mels)) # Pad (..., dim-2_start, dim-2_end, dim-1_start, dim-1_end)
            else:
                 # Crop frequency if necessary - potentially undesirable
                 spectrogram = spectrogram[:, :target_n_mels, :]


        # Pad/crop time dimension (time_frames)
        current_time_frames = spectrogram.shape[-1] # Get updated shape after n_mels adjust
        if current_time_frames > target_time_frames:
             # Random crop time
             start = random.randint(0, current_time_frames - target_time_frames)
             spectrogram = spectrogram[:, :, start:start + target_time_frames]
        elif current_time_frames < target_time_frames:
             # Random pad time
             padding_needed = target_time_frames - current_time_frames
             pad_left = random.randint(0, padding_needed)
             pad_right = padding_needed - pad_left
             spectrogram = F.pad(spectrogram, (pad_left, pad_right)) # Pad (dim-1_start, dim-1_end)
        # If current_time_frames == target_time_frames, do nothing.


        return spectrogram



    def __call__(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform (Tensor): Input waveform, expected shape [time].
        Returns:
            Tensor: Augmented spectrogram, shape [channels, n_mels, time_frames].
                    (Channels will be 1 for standard MelSpectrogram).
        """
        # Ensure input is 1D [time] or [channel, time]
        if waveform.ndim == 2 and waveform.shape[0] == 1: # [1, time]
             waveform = waveform.squeeze(0) # Convert to [time]
        elif waveform.ndim != 1:
             raise ValueError(f"Expected waveform shape [time] or [1, time], but got {waveform.shape}")

        # 1. Convert waveform to Mel Spectrogram
        # MelSpectrogram expects [channel, time], so add channel dim
        mel_spec = self.mel_spectrogram(waveform.unsqueeze(0)) # Shape: [1, n_mels, time_frames]

        # 2. Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec) # Shape: [1, n_mels, time_frames]

        # 3. Apply spectrogram augmentations
        augmented_mel_spec_db = self._apply_spectrogram_augmentations(mel_spec_db) # Shape: [1, n_mels, time_frames]

        # 4. Crop or pad spectrogram to target shape
        final_spectrogram = self._crop_or_pad_spectrogram(augmented_mel_spec_db)

        # Optional: If your model expects a 3-channel input (like some image models),
        # you might repeat the single channel:
        # final_spectrogram = final_spectrogram.repeat(3, 1, 1)

        return final_spectrogram



# --- SimCLR Dataset ---
# This dataset takes an existing dataset (like DatasetAudios)
# and returns two augmented views for each sample.
class SimCLRDataset(TorchDataset):
    def __init__(self, base_dataset: torch.Tensor, transform: Callable):
        """
        Args:
            base_dataset (torch.Tensor): The underlying data (e.g., dataset.x_train).
                                         Assumes shape [num_samples, features...]
            transform (callable): The augmentation function to apply.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        print(f"SimCLRDataset created with {len(self.base_dataset)} samples.")
        if len(self.base_dataset) == 0:
             raise ValueError("Base dataset is empty!")


    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Returns two different augmented views of the same sample.
        Detaches tensors after augmentation before returning.
        """
        # Ensure the sample is loaded correctly
        try:
             sample = self.base_dataset[index]
        except IndexError:
             print(f"Error: Index {index} out of bounds for base_dataset with length {len(self.base_dataset)}")
             # Handle error appropriately, maybe return None or raise?
             # For now, let's raise it to make it clear.
             raise IndexError(f"Index {index} out of bounds.")


        # Apply the transform pipeline twice independently
        # Add error handling around transform calls as they can fail
        try:
             view1 = self.transform(sample)
        except Exception as e:
             print(f"Error during transform view1 for index {index}: {e}")
             # Handle error: maybe skip this sample, return dummy data, or re-raise
             raise e # Re-raise for now


        try:
             view2 = self.transform(sample)
        except Exception as e:
             print(f"Error during transform view2 for index {index}: {e}")
             # Handle error
             raise e # Re-raise for now


        # Detach tensors before returning them to the DataLoader
        # This prevents the collate function from seeing tensors that require grad
        # from potentially complex augmentation histories. Gradients will be
        # calculated correctly when these views are fed into the model later.
        # Also ensure they are contiguous in memory, which can sometimes help collation.
        return view1.detach().contiguous(), view2.detach().contiguous()





