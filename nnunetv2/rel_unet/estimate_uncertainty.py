#imports
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import cv2
import nnunetv2
from nnunetv2.rel_unet.uncertainty_utils import *
import os
import shutil

# This script is used to compute the uncertainty score for a given dataset, after prediction was done.

#proba_dir ='' ### the folder with the checkpoints folders (output of previous script)
#raw_path ='' ##path to the folder with the dataset the user wants to predict ( input of previous script)
#labels = ''## optional - path to the labels of the dataset'
#score_type = '' ## optional - the score type to use for the uncertainty score. default is 'class_entropy' - other options are 'total_entropy' and 't_test'.
#outpot_pred_path = '' ## optional - path to the folder where the predictions will be saved. default is 'proba_dir + '/unnunet_pred''

def calculate_ece(probabilities, labels, n_bins=15):
    """Calculate the Expected Calibration Error (ECE).
    
    Args:
        probabilities (np.ndarray): Array of predicted probabilities.
        labels (np.ndarray): Array of true labels.
        n_bins (int): Number of bins for the calibration.
        
    Returns:
        float: ECE score.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = np.logical_and(probabilities > bin_lower, probabilities <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin] == 1)
            avg_confidence_in_bin = np.mean(probabilities[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def estimate_uncertainty(proba_dir, raw_path, score_type, labels, outpot_pred_path=False):
    dice_list = []
    uncertainty_scores = []
    ece_scores = []  # List to store ECE scores
    name_list = [name.split('.')[0].replace('_0000','') for name in os.listdir(raw_path)]

    for image_name in name_list:
        #compute p values map for the image
        # Map all dir in the folder - those are the checkpoints.
        checkpoint_list = [checkpoint for checkpoint in os.listdir(proba_dir) if os.path.isdir(proba_dir + '/' + checkpoint)]

        class0_array = [] # background
        class1_array = [] # foreground
        # For each checkpoint we will have a list of probability maps for each class.

        for checkpoint in checkpoint_list:
            # Load probs from .npz file
            prediction_file = np.load(proba_dir + '/' + checkpoint + '/' + image_name + '.npz', allow_pickle=True)
            class0_array.append(prediction_file['probabilities'][0, :, :, :])
            class1_array.append(prediction_file['probabilities'][1, :, :, :])

        # Convert the class arrays to numpy arrays.
        class0_array = np.array(class0_array)
        class1_array = np.array(class1_array)
        mask = (np.mean(class1_array, axis=0).T > 0.5).astype(np.uint8)
        map = np.zeros_like(mask)
        print(mask.shape, "---------------")
        if score_type == 't_test':
            p_values_map = T_test_on_single_image(class0_array, class1_array, plot_results = False)
            uncertainty_score = uncertainty_from_mask_and_valmap(p_values_map ,  mask)
            map = p_values_map

        elif score_type == 'class_entropy':
            class_entropy_map = entropy_map_fun(np.mean(class1_array,axis = 0), np.mean(class0_array,axis = 0))
            uncertainty_score =  uncertainty_from_mask_and_valmap(class_entropy_map ,  mask)
            map = class_entropy_map
        elif score_type == 'total_entropy':
            #append class one and class two on axis 0
            np.concatenate((class0_array, class1_array), axis = 0)
            total_entropy_map = -np.sum(class0_array * np.log(class0_array), axis=0)
            uncertainty_score =  uncertainty_from_mask_and_valmap(total_entropy_map ,  mask) * 0.01
            map = total_entropy_map

        uncertainty_scores.append(uncertainty_score)

        # ECE calculation
        probabilities = np.mean(class1_array, axis=0).flatten()
        true_labels = mask.flatten()
        ece_score = calculate_ece(probabilities, true_labels)
        ece_scores.append(ece_score)

        if labels:
            original_label_file = labels + '/' + image_name + '.nii.gz'
            label_data = nib.load(original_label_file)
            label = load_niigii_file(labels + '/' + image_name + '.nii.gz')
            temp_dice = dice(mask, label)
            dice_list.append(temp_dice)
        
        if not outpot_pred_path:
            outpot_pred_path = proba_dir + '/unnunet_pred'
        if not os.path.exists(outpot_pred_path):
            os.makedirs(outpot_pred_path)

        #copy prediction mask
        predicted_mask = proba_dir + '/checkpoint_best/' + image_name + '.nii.gz'
        #copy prediction mask to output folder
        shutil.copy(predicted_mask, outpot_pred_path + '/' + image_name + '_predicted_mask.nii.gz')
        
        #save the uncertainty map
        map_nii = nib.Nifti1Image(map, label_data.affine.T)
        nib.save(map_nii, outpot_pred_path + '/' + image_name + '_uncertainty_map.nii.gz')
    
    # Save the uncertainty scores, with the image names, if dice available also dice scores.
    uncertainty_df = pd.DataFrame({'image_name': name_list, 'uncertainty_score': uncertainty_scores, 'ece_score': ece_scores})
    if labels:
        uncertainty_df['dice_score'] = dice_list
    uncertainty_df.to_csv(outpot_pred_path + '/uncertainty_scores.csv', index=False)
    return uncertainty_df

def estimate_uncertainty():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proba_dir', type=str, default='', help='path to the folder with the checkpoints folders (output of previous script)')
    parser.add_argument('--raw_path', type=str, default='', help='path to the folder with the dataset the user wants to predict (input of previous script)')
    parser.add_argument('--labels', type=str, default='', help='optional - path to the labels of the dataset')
    parser.add_argument('--score_type', type=str, default='class_entropy', help='optional - the score type to use for the uncertainty score. default is class_entropy - other options are total_entropy and t_test')
    parser.add_argument('--outpot_pred_path', type=str, default='', help='optional - path to the folder where the predictions will be saved. default is proba_dir + /unnunet_pred')
    args = parser.parse_args()

    estimate_uncertainty(args.proba_dir, args.raw_path, args.score_type , args.labels , args.outpot_pred_path)

if __name__ == '__main__':
    estimate_uncertainty()
