"""
Engagement classification model converter to ONNX and torch
"""

# pylint: disable=no-name-in-module,import-error,duplicate-code
import os
import pathlib
from collections import OrderedDict

import h5py
import numpy as np
import onnx
import torch
import torch.nn.functional as F
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
from torch import nn

FILE_DIR = pathlib.Path(__file__).parent.resolve()


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


def _single_attention_model_torch(n_classes, weights, feature_vector_dim, samples=None):
    """
    Build a torch single attention classification model.

    Args:
        n_classes (int): The number of output classes.
        weights (str): Path to the pre-trained model weights.
        feature_vector_dim (int): The dimensionality of the input feature vectors.
                                  Must be 2560; otherwise, a ValueError is raised.
        samples (int, optional): The number of samples in the input sequence. Defaults to None.

    Returns:
        Model: A Torch model implementing attention-based classification.

    Raises:
        ValueError: If the feature vector dimension is not 2560.
    """

    class SingleAttentionModel(nn.Module):
        """
        Creates torch instance of single attention model.
        """

        def __init__(self, n_classes, feature_vector_dim, samples=None):
            """
            Single attention classification model in PyTorch.

            Args:
                n_classes (int): The number of output classes.
                feature_vector_dim (int): The dimensionality of the input feature vectors.
                samples (int, optional): The number of samples in the input sequence. Defaults
                                         to None.
            """
            super().__init__()

            if feature_vector_dim != 2560:
                raise ValueError("Unsupported feature vector dim. Maybe you use unsupported model.")

            self.feature_vector_dim = feature_vector_dim
            self.samples = samples

            self.e = nn.Linear(feature_vector_dim, 1)  # Linear layer to generate attention scores
            self.hidden_fc = nn.Linear(feature_vector_dim, 512)  # Hidden fully connected layer
            self.output_fc = nn.Linear(512, n_classes)  # Output layer

        def forward(self, x):
            """
            Forward pass of the model.

            Args:
                x (torch.Tensor): Input tensor of shape (batch, samples, feature_vector_dim)
            Returns:
                torch.Tensor: Output predictions of shape (batch, n_classes)
            """
            e = self.e(x).squeeze(-1)  # Shape: (batch, samples)
            alpha = F.softmax(e, dim=1)  # Shape: (batch, samples)

            alpha_repeated = alpha.unsqueeze(-1).repeat(
                1, 1, self.feature_vector_dim
            )  # Shape: (batch, samples, feature_vector_dim)
            c = (
                x * alpha_repeated
            )  # Element-wise multiplication (batch, samples, feature_vector_dim)

            context = c.sum(dim=1)  # Sum over samples -> (batch, feature_vector_dim)

            x = F.relu(self.hidden_fc(context))  # Hidden fully connected layer
            pred = F.softmax(self.output_fc(x), dim=1)  # Final classification layer

            return pred

        def load_keras_weights(self, h5_path):
            """Load weights directly from a Keras .h5 file without saving intermediate .pth file"""

            def recursive_extraction(h5_obj, prefix=""):
                for layer_name in h5_obj.keys():
                    layer_group = h5_obj[layer_name]
                    if isinstance(layer_group, h5py.Group):  # If it's a subgroup, recurse
                        recursive_extraction(layer_group, layer_name + "/")
                    else:  # If it's a dataset, store it
                        weights[prefix + layer_name] = np.array(layer_group)

            weights = {}
            with h5py.File(h5_path, "r") as f:
                recursive_extraction(f)

            # Create a PyTorch-compatible state_dict
            state_dict = OrderedDict()

            # ** Mapping Keras weights to PyTorch **
            try:
                state_dict["e.weight"] = torch.tensor(
                    weights["e/kernel:0"].T, dtype=torch.float32
                )  # Linear layer: (out_dim, in_dim) → (in_dim, out_dim)
                state_dict["e.bias"] = torch.tensor(weights["e/bias:0"], dtype=torch.float32)

                state_dict["hidden_fc.weight"] = torch.tensor(
                    weights["hidden_FC/kernel:0"].T, dtype=torch.float32
                )
                state_dict["hidden_fc.bias"] = torch.tensor(
                    weights["hidden_FC/bias:0"], dtype=torch.float32
                )

                state_dict["output_fc.weight"] = torch.tensor(
                    weights["dense/kernel:0"].T, dtype=torch.float32
                )
                state_dict["output_fc.bias"] = torch.tensor(
                    weights["dense/bias:0"], dtype=torch.float32
                )

                # Load weights into model
                self.load_state_dict(state_dict)
                print("Weights loaded successfully from Keras .h5 file!")

            except KeyError as e:
                print(f"Error: Missing key in .h5 file: {e}")

    model = SingleAttentionModel(n_classes, feature_vector_dim, samples)

    model.load_keras_weights(weights)
    model.eval()

    return model


def get_engagement_model(feature_vector_dim: int, number_of_frames: int):
    """
    Load and return the engagement classification model.

    Args:
        feature_vector_dim (int): The dimensionality of the input feature vectors.
        number_of_frames (int): The number of frames in the input sequence.

    Returns:
        Model: A Keras model for engagement classification.
    """
    weights_path = os.path.join(FILE_DIR, "engagement_single_attention.h5")
    model = _single_attention_model(2, weights_path, feature_vector_dim, number_of_frames)
    model.summary()
    input_data = np.random.random((1, 128, 2560))
    _ = model(input_data)
    return model


def get_engagement_torch_model(feature_vector_dim: int, number_of_frames: int):
    """
    Load and return the torch engagement classification model.

    Args:
        feature_vector_dim (int): The dimensionality of the input feature vectors.
        number_of_frames (int): The number of frames in the input sequence.

    Returns:
        Model: A Keras model for engagement classification.
    """
    weights_path = os.path.join(FILE_DIR, "engagement_single_attention.h5")
    model = _single_attention_model_torch(2, weights_path, feature_vector_dim, number_of_frames)
    return model


def engagement_to_onnx_converter(onnx_filename: str) -> onnx.ModelProto:
    """
    Converts and exports an engagement recognition model to ONNX format if it does not already
    exist.

    Args:
        onnx_filename (str): The file path where the ONNX model will be saved.

    Returns:
        None
    """
    if not os.path.exists(onnx_filename):
        model = get_engagement_model(2560, 128)

        model.export(onnx_filename, format="onnx")
    else:
        print("SKIP Engagement onnx model creation")

    loaded_model = onnx.load(onnx_filename)
    onnx.checker.check_model(loaded_model)
    print("The ONNX engagement model is valid.")


def engagement_to_torch_converter(torch_filename: str) -> None:
    """
    Converts an ONNX engagement recognition model to a TorchScript model and saves it.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model to be converted.
        torch_filename (str): The file path where the TorchScript model will be saved.

    Returns:
        None
    """
    if os.path.exists(torch_filename):
        print("SKIP Engagement torch model creation")
    pytorch_model = get_engagement_torch_model(2560, 128)
    input_shape = (1, 128, 2560)
    model_example = torch.rand(*input_shape)
    traced_script_module = torch.jit.trace(pytorch_model, model_example)
    traced_script_module.save(torch_filename)
