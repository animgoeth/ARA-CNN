import numpy
import os

from PIL import Image

FLOAT_EPS = 1e-5

DOWNLOAD_PATH = 'https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip'
ARCHIVE_FILE_NAME = DOWNLOAD_PATH.split('/')[-1]
DATASET_FOLDER_NAME = ARCHIVE_FILE_NAME.replace('.zip', '')
TRAIN_RATIO = 0.9


def download_and_extract_dataset():
    os.system('wget %s' % DOWNLOAD_PATH)
    os.system('unzip %s' % ARCHIVE_FILE_NAME)


def split_dataset():
    class_names = os.listdir(DATASET_FOLDER_NAME)
    test_path, train_path = prepare_split_target_folders()

    for class_name in class_names:
        current_class_path = os.path.join(DATASET_FOLDER_NAME, class_name)
        current_class_images = list((filter(lambda filename: filename.endswith('tif'), os.listdir(current_class_path))))
        number_of_images_in_class = len(current_class_images)

        sampling_permuatation = numpy.random.permutation(number_of_images_in_class)
        train_index_bound = int(TRAIN_RATIO * number_of_images_in_class)

        category_test_path, category_train_path = prepare_class_folder(class_name, test_path, train_path)

        for current_index in range(train_index_bound):
            copy_image_to_destination_folder(current_class_images,
                                             sampling_permuatation,
                                             current_index,
                                             current_class_path,
                                             category_train_path)

        for current_index in range(train_index_bound, number_of_images_in_class):
            copy_image_to_destination_folder(current_class_images,
                                             sampling_permuatation,
                                             current_index,
                                             current_class_path,
                                             category_test_path)


def copy_image_to_destination_folder(current_class_images, sampling_permutation, current_index, current_class_path, destination_path):
    current_image_name = current_class_images[sampling_permutation[current_index]]
    current_image_path = os.path.join(current_class_path, current_image_name)

    current_image_dst_path = os.path.join(destination_path, current_image_name).replace('.tif', '.jpg')

    image = Image.open(current_image_path)
    image.save(current_image_dst_path)


def prepare_class_folder(class_name, test_path, train_path):
    train_path_for_class = os.path.join(train_path, class_name)
    test_path_for_class = os.path.join(test_path, class_name)
    custom_mkdir(train_path_for_class)
    custom_mkdir(test_path_for_class)
    return test_path_for_class, train_path_for_class


def prepare_split_target_folders():
    train_path = os.path.join('./', "train")
    test_path = os.path.join('./', "test")
    custom_mkdir('train')
    custom_mkdir('test')
    return test_path, train_path


def custom_mkdir(train_path):
    if not os.path.exists(train_path):
        os.mkdir(train_path)


def cleanup():
    os.system('rm %s' % ARCHIVE_FILE_NAME)
    os.system('rm -rf %s' % DATASET_FOLDER_NAME)


if __name__ == "__main__":
    download_and_extract_dataset()
    split_dataset()
    cleanup()
