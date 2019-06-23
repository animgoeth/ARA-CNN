from config import COLOR_TYPE, CLASS_DICT, IMAGE_SIZE, EPS
from keras.models import load_model
from model_utils import single_class_accuracy, import_folder_to_numpy_array, get_dataset_path
from ara_cnn import build_simple_cnn_model_with_dropout

import keras.backend as K
import numpy


def load_dropout_model_instance(model_path):
    """
    This function is responsible for loading a dropout model instance from file.
    :param model_path: Path to dropout model.
    :return: A dropout model instance.
    """
    return load_model(model_path, custom_objects={"fn": single_class_accuracy(0)})


def copy_weights(src_model, dst_model):
    """
    A helper function for copying weights.
    :param src_model: A source model for copying.
    :param dst_model:  A destination model to copying.
    """
    for src_layer, dst_layer in zip(src_model.layers, dst_model.layers):
        dst_layer.set_weights(src_layer.get_weights())


def build_variational_model(dropout_model_instance):
    """
    This function creates a variational version of dropout model.
    :param dropout_model_instance: A dropout model instance to be copied to variational dropout model.
    :return: A new variational copy of a dropout model.
    """
    variational_model = build_simple_cnn_model_with_dropout(freeze_batch=True)
    copy_weights(dropout_model_instance, variational_model)
    return variational_model


def build_variational_inference_function(variational_inference_model):
    """
    This function is responsible for building a function for variational inference.
    :param variational_inference_model: A base model for variational inference.
    :return: A variational inference function.
    """
    input_to_model = variational_inference_model.input
    output_from_model = variational_inference_model.output[0]
    variational_base_function = K.function(
        inputs=[input_to_model, K.learning_phase()],
        outputs=[output_from_model])

    def _variational_inference_function(inputs, nb_of_samples):
        """
        A function for variational inference.
        :param inputs: Inputs for variational inference.
        :param nb_of_samples: Number of variational samples.
        :return: A sample table with shape (number_of_samples, ..).
        """
        samples = [variational_base_function([inputs, 1]) for _ in range(nb_of_samples)]
        return numpy.concatenate(samples)

    return _variational_inference_function


def build_variational_inference_saliency_module(variational_inference_model_, agg_function):
    input_to_model = variational_inference_model_.input
    output_from_model = variational_inference_model_.output[0]
    output_to_saliency = agg_function(output_from_model)
    gradient = K.gradients(loss=output_to_saliency, variables=input_to_model)[0]
    variational_base_function = K.function(
        inputs=[input_to_model, K.learning_phase()],
        outputs=[gradient])

    def _variational_inference_saliency_function(inputs, nb_of_samples):
        """
        A function for variational inference.
        :param inputs: Inputs for variational inference.
        :param nb_of_samples: Number of variational samples.
        :return: A sample table with shape (number_of_samples, ..).
        """
        samples = [variational_base_function([inputs, 1]) for _ in range(nb_of_samples)]
        return numpy.concatenate(samples)

    return _variational_inference_saliency_function


def compute_table_uncertainty(distributions, measure_name):
    """
    This function computes the acquisition function for distributions.
    :param distributions: A table of distributions.
    :param measure_name: Uncertainty measure - either Entropy or BALD.
    :return: A table of uncertainty values.
    """
    def _compute_table_entropy(table):
        """
        A helper function for computing entropy for a single distributions table.
        :param table: A table of distributions.
        :return: A table of entropies.
        """
        table = numpy.clip(table, EPS, 1 - EPS)
        return (-table * numpy.log(table)).sum(axis=1)
    mean_entropy = numpy.array([_compute_table_entropy(table) for table in distributions]).mean(axis=0)
    entropy_of_mean = _compute_table_entropy(distributions.mean(axis=0))

    if measure_name == 'Entropy':
        return entropy_of_mean # Entropy
    elif measure_name == 'BALD':
        return entropy_of_mean - mean_entropy # BALD

