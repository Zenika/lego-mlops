library(torch)
library(torchvision)
library(yaml)
library(magick)
library(jsonlite)
library(ggplot2)

params <- yaml.load_file('params.yaml')
test_dir <- params$evaluate$test_dataset
model_location <- params$evaluate$model

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
print(device)

model <- torch_load(model_location)
model$eval()

criterion <- nn_cross_entropy_loss()

test_batch <- function(b) {
  output <- model(b[[1]])
  labels <- b[[2]]$to(device = device)
  loss <- criterion(output, labels)
  
  test_losses <<- c(test_losses, loss$item())
  # torch_max returns a list, with position 1 containing the values
  # and position 2 containing the respective indices
  predicted <- torch_max(output$data(), dim = 2)[[2]]
  total <<- total + labels$size(1)
  # add number of correct classifications in this batch to the aggregate
  correct <<- correct + (predicted == labels)$sum()$item()

  observations <<- c(observations, as.numeric(labels))
  predictions <<- c(predictions, as.numeric(predicted))
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



test_losses <- c()
observations <- c()
predictions <- c()
total <- 0
correct <- 0


test_ds_for_valid <- image_folder_dataset(test_dir, transform = valid_transforms)
batch_size <- 16
test_dl_for_valid <- dataloader(test_ds_for_valid, batch_size = batch_size)

for (b in enumerate(test_dl_for_valid)) {
  test_batch(b)
}

# Metrics
message(paste("Mean loss is", mean(test_losses)))
accuracy <- correct / total
message(paste("Accuracy is", accuracy))

metric_file <- file(params$evaluate$metrics)
writeLines(toJSON(list(accuracy = accuracy), auto_unbox=TRUE), metric_file)
close(metric_file)

# Confusion matrix
cm <- table(as.factor(observations), as.factor(predictions), dnn = c('observations', 'predictions'))
cm_as_df <- as.data.frame(cm)

png(params$evaluate$confusion_matrix)
ggplot(cm_as_df[cm_as_df$Freq > 0,], mapping = aes(x=observations, y=predictions)) +
    geom_tile(aes(fill = Freq)) +
    geom_text(aes(label = Freq)) +
    scale_fill_gradient(low = "blue", high = "red") +
    theme_bw() +
    theme(legend.position = "none")
dev.off()
