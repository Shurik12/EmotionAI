import source.audio_analysis_utils.utils as utils
import source.config as config
import source as source
import source.audio_analysis_utils.preprocess_data as data
from config.settings import Config

import torch
import numpy as np
import os


def predict(input_file, model_save_path=config.AUDIO_MODEL_SAVE_PATH):
    print(f"audio_file: {input_file}")

    best_hyperparameters = utils.load_dict_from_json(config.AUDIO_BEST_HP_JSON_SAVE_PATH)
    print(f"Best hyperparameters, {best_hyperparameters}")

    # Extract features
    extracted_mfcc = utils.extract_mfcc(
        os.path.join(Config.UPLOAD_FOLDER, input_file),
        N_FFT=best_hyperparameters['N_FFT'],
        NUM_MFCC=best_hyperparameters['NUM_MFCC'],
        HOP_LENGTH=best_hyperparameters['HOP_LENGTH']
    )
    print(f"extracted_mfcc: {extracted_mfcc.shape}")

    # Reshape to make sure it fit pytorch model
    extracted_mfcc = np.repeat(extracted_mfcc[np.newaxis, np.newaxis, :, :], 3, axis=1)
    print(f"Reshaped extracted_mfcc: {extracted_mfcc.shape}")

    # Convert to tensor
    extracted_mfcc = torch.from_numpy(extracted_mfcc).float().to(config.device)

    print (1)

    # Load the model
    model = torch.load(model_save_path, weights_only=False, map_location=torch.device("cpu"))
    model.to(config.device).eval()

    print (2)

    prediction = model(extracted_mfcc)

    print (3)

    prediction = torch.nn.functional.softmax(prediction, dim=1)

    print (4)

    prediction_numpy = prediction[0].cpu().detach().numpy()
    print (prediction_numpy)
    print(f"prediction: {prediction_numpy}")

    # Get the predicted label
    predicted_label = max(prediction_numpy)
    emotion = config.EMOTION_INDEX[prediction_numpy.tolist().index(predicted_label)]
    print(f"Predicted emotion: {emotion} {round(predicted_label, 2)}")

    ret_string = utils.get_softmax_probs_string(prediction_numpy, list(config.EMOTION_INDEX.values()))
    q = {category: f"{score:.2f}" for category, score in zip(list(config.EMOTION_INDEX.values()), prediction_numpy)}
    print(f"\n\n\nPrediction labels:\n{ret_string}")

    return emotion, round(predicted_label, 2), q
