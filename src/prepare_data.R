library(yaml)

params <- yaml.load_file('params.yaml')
data_dir <- params$prepare_data$input
train_dir <- params$prepare_data$train_output
index <- read.csv(file.path(data_dir, 'index.csv'))
if (!dir.exists(train_dir)) {
    message("Creating train directory")
    dir.create(train_dir, recursive = TRUE)
}

test_dir <- params$prepare_data$test_output
test_index <- read.csv(file.path(data_dir, 'test.csv'))
if (!dir.exists(test_dir)) {
    message("Creating test directory")
    dir.create(test_dir, recursive = TRUE)
}

link_image <- function(path, class_id, output_dir) {
    src_path <- file.path(data_dir, path)
    dest_dir <- file.path(output_dir, class_id)
    dest_path <- file.path(dest_dir, basename(src_path))
    if(!dir.exists(dest_dir)) {
        message(paste('Creating', dest_dir))
        dir.create(dest_dir, recursive = TRUE)
    }
    message(paste('Linking', src_path, 'to', dest_path))
    R.utils::createLink(link = dest_path, src_path)
}

suppressMessages(mapply(link_image, index$path, index$class_id, train_dir))
suppressMessages(mapply(link_image, test_index$path, test_index$class_id, test_dir))
