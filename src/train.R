library(torch)
library(torchvision)

library(yaml)
library(dplyr)
library(purrr)
library(magick)

params <- yaml.load_file('params.yaml')
train_dir <- params$train$input

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
print(device)

train_transforms <- function(img) {
  img %>%
    magick_loader %>%
    transform_resize(size = c(448, 448))    %>%
    # data augmentation
    transform_random_resized_crop(size = c(224, 224)) %>%
    # first convert image to tensor
    transform_to_tensor() %>%
    # then move to the GPU (if available)
    (function(x) x$to(device = device)) %>%
#    # data augmentation
#    transform_color_jitter() %>%
#    # data augmentation
#    transform_random_horizontal_flip() %>%
    # normalize according to what is expected by resnet
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}
     
valid_transforms <- function(img) {
  img %>%
    magick_loader %>%
    transform_resize(size = c(448, 448))    %>%
    # data augmentation
    transform_center_crop(size = c(224, 224)) %>%
    # first convert image to tensor
    transform_to_tensor() %>%
    # then move to the GPU (if available)
    (function(x) x$to(device = device)) %>%
#    # data augmentation
#    transform_color_jitter() %>%
#    # data augmentation
#    transform_random_horizontal_flip() %>%
    # normalize according to what is expected by resnet
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}


train_ds <- image_folder_dataset(
  train_dir,
  transform = train_transforms)
class_names <- train_ds$classes

batch_size <- params$train$batch_size
train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)


model <- model_resnet18(pretrained = TRUE)

model$parameters %>% purrr::walk(function(param) param$requires_grad_(FALSE))

# Replace output layer
num_features <- model$fc$in_features
model$fc <- nn_linear(in_features = num_features, out_features = length(class_names))

# Configure device
model <- model$to(device = device)

# Prepare training

criterion <- nn_cross_entropy_loss()
optimizer <- optim_sgd(model$parameters, lr = 0.05, momentum = 0.9)

# Scheduler init

num_epochs <- params$train$num_epochs

scheduler <- optimizer %>% 
  lr_one_cycle(max_lr = 0.05, epochs = num_epochs, steps_per_epoch = train_dl$.length())

#Training loop

train_batch <- function(b) {
  optimizer$zero_grad()
  output <- model(b[[1]])
  loss <- criterion(output, b[[2]]$to(device = device))
  loss$backward()
  optimizer$step()
  scheduler$step()
  loss$item()
}

for (epoch in 1:num_epochs) {
  model$train()
  train_losses <- c()
  for (b in enumerate(train_dl)) {
    loss <- train_batch(b)
    train_losses <- c(train_losses, loss)
  }
  cat(sprintf("\nLoss at epoch %d: training: %3f\n", epoch, mean(train_losses)))
}

torch_save(model, params$train$model)
