"""
Engagement classification model implementation
"""

# pylint: disable=no-name-in-module,import-error
from .utils import get_engagement_classification_weights

TF_AVAILABLE = True
try:
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import (
        Activation,
        Dense,
        Input,
        Lambda,
        Multiply,
        Permute,
        RepeatVector,
        Reshape,
    )
    from tensorflow.keras.models import Model
except ImportError:
    TF_AVAILABLE = False


def _single_attention_model(n_classes, weights, feature_vector_dim, samples=None):
    """
    Build a single attention classification model.

    Args:
        n_classes (int): The number of output classes.
        weights (str): Path to the pre-trained model weights.
        feature_vector_dim (int): The dimensionality of the input feature vectors.
                                  Must be 2560; otherwise, a ValueError is raised.
        samples (int, optional): The number of samples in the input sequence. Defaults to None.

    Returns:
        Model: A Keras model implementing attention-based classification.

    Raises:
        ValueError: If the feature vector dimension is not 2560.
    """
    if TF_AVAILABLE is False:
        raise RuntimeError("Cannot create engagement model because tensorflow is not installed")
    if feature_vector_dim != 2560:
        raise ValueError("Unsupported feature vector dim. Maybe you use unsupported model.")
    inputs = Input(
        shape=(samples, feature_vector_dim), name="image_set"
    )  # (batch, samples, features)
    e = Dense(1, activation="linear", name="e")(inputs)
    e = Reshape([samples], name="alignment")(e)
    alpha = Activation("softmax", name="alpha")(e)

    alpha_repeated = Permute([2, 1], name="alpha_repeated")(
        RepeatVector(feature_vector_dim, name="repeat")(alpha)
    )

    c = Multiply(name="c")([inputs, alpha_repeated])
    x = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(feature_vector_dim,), name="context")(
        c
    )

    x = Dense(units=512, activation="relu", name="hidden_FC")(x)  # (batch, units) #128 64

    pred = Dense(n_classes, activation="softmax")(x)  # (batch, classes)
    model = Model(inputs=inputs, outputs=pred)

    model.load_weights(weights)

    return model


def get_engagement_model(feature_vector_dim, number_of_frames):
    """
    Load and return the engagement classification model.

    Args:
        feature_vector_dim (int): The dimensionality of the input feature vectors.
        number_of_frames (int): The number of frames in the input sequence.

    Returns:
        Model: A Keras model for engagement classification.
    """
    weights_path = get_engagement_classification_weights()
    return _single_attention_model(2, weights_path, feature_vector_dim, number_of_frames)
