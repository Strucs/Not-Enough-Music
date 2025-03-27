#
import torch
#
import lib_dataset as ld



#
data: ld.DatasetAudios = ld.DatasetAudios()


#
x_train, y_train = data.get_batch_train(42)

#
print(f"x_train {x_train.shape} y_train {y_train.shape}")

print(f"y_train = {y_train}")




