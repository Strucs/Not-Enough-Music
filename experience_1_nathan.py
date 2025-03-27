#
import torch
from transformers import ASTForAudioClassification

#
import lib_dataset as ld



#
data: ld.DatasetImages = ld.DatasetImages()

#
model: ASTForAudioClassification = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", attn_implementation="sdpa", torch_dtype=torch.float32)


#
x_train, y_train = data.get_batch_train(42)

#
print(f"x_train {x_train.shape} y_train {y_train.shape}")

print(f"y_train = {y_train}")


model.eval()

predict: Tensor = model(x_train[0])

print(f"predict : {predict}\n\n\n | {predict.shape}")

