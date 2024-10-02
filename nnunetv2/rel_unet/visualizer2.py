import os
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

raw_path = r'/home/seyedsina.ziaee/datasets/v2_all/nnUNet_raw/Dataset002_Kits'
results_path = r'/home/seyedsina.ziaee/datasets/v2_all/nnUNet_results/Dataset002_Kits/nnUNetTrainer__nnUNetPlans__2d'

# ce_scores = pd.read_csv('/home/seyedsina.ziaee/datasets/v2_all/uncert_class_kidney/uncertainty_scores.csv')
ce_scores = pd.read_csv('/home/seyedsina.ziaee/datasets/v2_all/uncert_class_entropy2/uncertainty_scores.csv')
te_scores = pd.read_csv('/home/seyedsina.ziaee/datasets/v2_all/uncert_total_entropy2/uncertainty_scores.csv')
# tt_scores = pd.read_csv('/home/seyedsina.ziaee/datasets/v2_all/uncert_t_test/uncertainty_scores.csv')

#get images names
images = [x for x in os.listdir(raw_path + '/imagesTs2/') if x.endswith('.nii.gz')]
count   = 0
print(images)
for image in images:
    if image == "kits_00588_0000.nii.gz":
        try:
            count += 1
            # if count > 2:
            #     break
            #load image and raw data
            raw_image = nib.load(raw_path + '/imagesTs2/' + image).get_fdata()
            gt_mask = nib.load(raw_path + '/labelsTs2/' + image.replace("_0000","")).get_fdata()
            # pred_mask = nib.load('/home/seyedsina.ziaee/datasets/v2_all/uncert_class_kidney/' + image.replace("_0000","_predicted_mask")).get_fdata()
            # ce_map = nib.load('/home/seyedsina.ziaee/datasets/v2_all/uncert_class_kidney/' + image.replace("_0000","_uncertainty_map")).get_fdata()
            pred_mask = nib.load('/home/seyedsina.ziaee/datasets/v2_all/uncert_class_entropy2/' + image.replace("_0000","_predicted_mask")).get_fdata()
            ce_map = nib.load('/home/seyedsina.ziaee/datasets/v2_all/uncert_class_entropy2/' + image.replace("_0000","_uncertainty_map")).get_fdata()
            # tumor_map = nib.load('/home/seyedsina.ziaee/datasets/v2_all/temp/' + image.replace("_0000","_tumor_uncertainty_map")).get_fdata()
            # kidney_map = nib.load('/home/seyedsina.ziaee/datasets/v2_all/temp/' + image.replace("_0000","_kidney_uncertainty_map")).get_fdata()
            te_map = nib.load('/home/seyedsina.ziaee/datasets/v2_all/uncert_total_entropy2/' + image.replace("_0000","_uncertainty_map")).get_fdata()
            # tt_map = nib.load('/home/seyedsina.ziaee/datasets/v2_all/uncert_t_test/' + image.replace("_0000","_uncertainty_map")).get_fdata()

            #load uncertainty scores
            ce_score = ce_scores.loc[ce_scores['image_name'] == image.replace("_0000.nii.gz","")]['uncertainty_score'].values[0]
            te_score = te_scores.loc[te_scores['image_name'] == image.replace("_0000.nii.gz","")]['uncertainty_score'].values[0]
            # tt_score = tt_scores.loc[tt_scores['image_name'] == image.replace("_0000.nii.gz","")]['uncertainty_score'].values[0]
            dice_score = ce_scores.loc[ce_scores['image_name'] == image.replace("_0000.nii.gz","")]['dice_score'].values[0]
            
            #plot images in one row; raw image + ground truth + predicted mask , uncertainty map ce , uncertainty map te , uncertainty map tt
            #add titles ; image name + deice score, ce score, te score, tt score
            # if count > 20:
            #     break
            # if dice_score < 0.6 or ce_score > 0.4 or te_score > 0.85 or tt_score > 0.05:
            depth = raw_image.shape[0]
            print("working on image:", image)
            for slice_inx in range(depth):
                # count += 1
                fig, axs = plt.subplots(1, 3, figsize = (15,20)) #create figure with 5 subplots
                #add colorbar
                # axs[0].imshow( raw_image[slice_inx], cmap='gray', vmin=0, vmax=3) #plot raw image
                # axs[0].imshow( gt_mask[slice_inx], cmap='Greens', alpha = 0.5, vmin=0, vmax=3) #plot ground truth mask
                # axs[0].imshow( pred_mask[slice_inx], cmap='Reds', alpha = 0.5, vmin=0, vmax=3) #plot ground truth mask
                axs[0].imshow( raw_image[slice_inx], cmap='gray') #plot raw image
                axs[0].imshow( gt_mask[slice_inx], cmap='Greens', alpha = 0.5) #plot ground truth mask
                axs[0].imshow( pred_mask[slice_inx], cmap='Reds', alpha = 0.5) #plot ground truth mask
                axs[0].legend(handles = [Patch(facecolor='#54322f',label='Predicted Tumor'), Patch(facecolor='#9d6850', label='Predicted Kidney'), Patch(facecolor='#d99484',label='Predicted kidney without groundtruth'), Patch(facecolor='#b4c5b0',label='kidney groundtruth'), Patch(facecolor='#5e6e5a',label='tumor groundtruth')])  # add legend
                axs[0].set_title(image.split("_0000.")[0] + ' Dice: ' + "%.3f" % dice_score) #add title to subplot
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                
                
                axs[1].imshow(np.transpose(ce_map[:, :, slice_inx]) , cmap='hot') #plot ce map
                axs[1].set_title('Class entropy, Score: ' + "%.3f" % ce_score) #add title to subplot
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                
                im2 = axs[2].imshow(np.transpose(te_map[:, :, slice_inx]) , cmap='hot') #plot te map
                axs[2].set_title('Total entropy, Score: ' + "%.3f" % te_score) #add title to subplot
                axs[2].set_xticks([])
                axs[2].set_yticks([])
                
                cbar = fig.colorbar(im2, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
                cbar.set_label('Uncertainty (Higher value = More Uncertainty, Lower value = Less Uncertainty)')


                num = image.split("_")[1]
                os.makedirs(f"/home/seyedsina.ziaee/datasets/v2_all/temp_visualization2/kits_{num}_uncertainty_maps", exist_ok=True)
                plt.savefig(f"/home/seyedsina.ziaee/datasets/v2_all/temp_visualization2/kits_{num}_uncertainty_maps/slice_{slice_inx:04d}.png")
                plt.close()
        except Exception as e:
            print(e)
