import numpy
import os

from keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import load_img, img_to_array
from keras.regularizers import l2

from keras import backend as K


class EmptyImagesFolder(Exception):
    pass


def load_all_images_from_folder(folder_path, target_size, color_mode, image_extension="jpg"):
    # Getting list of images filepath
    list_of_files = os.listdir(folder_path)
    list_of_images_path = [os.path.join(folder_path, path) for path in list_of_files if path.endswith(image_extension)]

    if len(list_of_images_path) == 0:
        raise EmptyImagesFolder

    # Getting list of images
    list_of_images = [img_to_array(load_img(path=path, grayscale=(color_mode == "grayscale"), target_size=target_size))
                      for path in list_of_images_path]

    return numpy.array(list_of_images)


def import_folder_to_numpy_array(folder_path, target_size, color_mode, binary=False, class_dict=None):
    # Preparations of accumulative variables
    classes = os.listdir(folder_path)
    arrays_list = list()
    class_list = list()

    # Crawling through folder to get all images
    for class_name in classes:
        class_dir_path = os.path.join(folder_path, class_name)
        array_of_images = load_all_images_from_folder(
            folder_path=class_dir_path,
            target_size=target_size,
            color_mode=color_mode)
        arrays_list.append(array_of_images)

        if binary:
            class_name = '02_NOT_TUMOR'

        class_list.append(class_name)

    # Preparations of class_dict if it not exist
    if class_dict is None:
        class_dict = dict()
        for nb, class_name in enumerate(class_list):
            class_dict[class_name] = nb

    # Preparations of labels
    labels = list()
    for class_nb, class_name in enumerate(class_list):
        labels = labels + [class_dict[class_name]] * arrays_list[class_nb].shape[0]

    # Final joining to numpy arrays
    images_array = numpy.concatenate(arrays_list)
    labels_array = numpy.array(labels)

    return images_array, labels_array


def single_class_accuracy(interesting_class_id):
    """
    This function creates a Keras metric with single class accuracy for a class provided.
    :param interesting_class_id: Number of class.
    :return: A Keras metric function.
    """
    def fn(y_true, y_pred):
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        positive_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
        true_mask = K.cast(K.equal(y_true, interesting_class_id), 'int32')
        acc_mask = K.cast(K.equal(positive_mask, true_mask), 'float32')
        class_acc = K.mean(acc_mask)
        return class_acc

    return fn


def build_stem_cnn_block(
        input_tensor_,
        filter_nb,
        filter_size=(7, 7),
        strides=(1, 1),
        freeze_batch=False,
        alpha=0.1, l2_param=0.0001, pooling=True, pooling_size=(2, 2)):
    """
    This function is responsible for creation of a stem cnn block.
    :param input_tensor_: An input tensor to block.
    :param filter_nb: Number of block filters.
    :param filter_size: A size of filter.
    :param strides: Filter strides.
    :param freeze_batch: Flag if BatchNormalization should be freezed when Keras is in training phase.
    :param alpha: Alpha parameter to LeakyRelu
    :param l2_param: L2 regularization term.
    :param pooling: Flag if pooling should be applied.
    :param pooling_size: A size of pooling (if applied)
    :return: An output tensor.
    """
    block_acc_tensor = Conv2D(
        filter_nb, filter_size, padding="same",
        strides=strides,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_param))(input_tensor_)
    block_acc_tensor = BatchNormalization()(block_acc_tensor) if not freeze_batch \
        else NonTrainableBatchNormalization()(block_acc_tensor)
    block_acc_tensor = LeakyReLU(alpha=alpha)(block_acc_tensor)
    if pooling:
        block_acc_tensor = MaxPooling2D(pooling_size)(block_acc_tensor)
    return block_acc_tensor


class NonTrainableBatchNormalization(BatchNormalization):
    """
    This class makes possible to freeze batch normalization while Keras is in training phase.
    """
    def call(self, inputs, training=None):
        return super(NonTrainableBatchNormalization, self).call(inputs, training=False)


def get_dataset_path(dataset_name):
    """
    A helper function for getting a path to a dataset.
    :param dataset_name: A name of dataset for which path will be created.
    :return: A path to a dataset.
    """
    result_path = os.getcwd()
    result_path = os.path.join(result_path, "datasets")
    result_path = os.path.join(result_path, dataset_name)

    return result_path

