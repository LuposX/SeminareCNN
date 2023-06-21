# SeminareCNN
> This repo is about the Seminar "Grundlagen des maschinellen Lernens 1 & 2" I did at the Karlsruhe Institute of Technology in the summer semester 2023 where I had the Topic "Convolutional Neural Networks".

## Description

My Topic was "Convolutional Neural Networks" and I had the task to write a Paper and present a Presentation about this topic.
This Repository contains all the material I used for both of them.

The trainings and inference Code for my self-trained CNN for digit recognition can be found in form of jupyter file.

## Models

I trained four different model, below I will shortly describe them:
- CNN, the Baseline Standard Model based on a Convolutional Neural Network.
- CNN_NOISE_RC, the Baseline model, but trained on altered data, where random noise and random cropping was used.
- FC_NN, a model without convolutional layers using only fully connected layers, but habing the same size as the CNN Model.
- FC_NN_NOISE_RC, the fully connected model with random Noise and Random Cropping.

You can find the Model checkpoints and code, in the respective folders.

### Architetcure of the Baseline Model

```
CNN(
  (hidden0): Sequential(
    (0): Conv2d(1, 16, kernel_size=(4, 4), stride=(1, 1))
    (1): LeakyReLU(negative_slope=0.2)
  )
  (hidden1): Sequential(
    (0): Conv2d(16, 64, kernel_size=(4, 4), stride=(1, 1))
    (1): LeakyReLU(negative_slope=0.2)
  )
  (hidden2): Sequential(
    (0): Conv2d(64, 16, kernel_size=(4, 4), stride=(1, 1))
    (1): LeakyReLU(negative_slope=0.2)
  )
  (hidden3): Sequential(
    (0): Linear(in_features=5776, out_features=200, bias=True)
    (1): LeakyReLU(negative_slope=0.2)
  )
  (hidden4): Sequential(
    (0): Linear(in_features=200, out_features=10, bias=True)
    (1): Softmax(dim=None)
  )
)
````


## Loading Model & Inference

```python
# We need one extra dimesnion to emulate batch size.
data = data.view(1, 1, 28, 28)

model = net.load_from_checkpoint(checkpoint_folder + "/1L_RN_RC-epoch=4.ckpt")
model.eval()

with torch.no_grad():
   y_hat = model(data.cuda())

print("Results for Inference")
print("---------------------------")
print(" ")
print("Probabilities for Predicted-Labels: ", y_hat)
print(" ")
print("Predicted Label: ", y_hat.argmax())
````

## Results

Cross-Entropy-Loss during training        |  Validation Accuracy on the test Set
:-------------------------:|:---------------------------------------------------:
![](train_loss.jpeg)  |  ![](val_acc.jpeg)
