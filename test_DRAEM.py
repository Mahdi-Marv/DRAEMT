import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


import matplotlib.pyplot as plt
import os

def plot_images_and_save(dataloader, subclass_name, shrink_factor, grid_size=(5, 4), target_total_images=20):
    images_collected = []
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 10))

    # Iterate over the dataloader and collect images until you reach the desired number or exhaust the dataloader
    for batch in dataloader:
        images_batch = batch['image']
        for img in images_batch:
            images_collected.append(img)
            if len(images_collected) >= target_total_images:
                break
        if len(images_collected) >= target_total_images:
            break

    # Plotting
    for i, ax in enumerate(axs.flat):
        if i >= len(images_collected):  # Ensure we do not go out of bounds
            break
        # Adjust the permute as necessary based on your image tensor dimensions
        ax.imshow(images_collected[i].permute(1, 2, 0).cpu().numpy())
        ax.axis('off')

    # Adjust subplot parameters to give specified padding
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'{subclass_name} with shrink factor: {shrink_factor}', fontsize=16)

    # Ensure the folder exists
    folder_path = f'/kaggle/working/plots/{subclass_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the figure
    fig_path = os.path.join(folder_path, f'image_grid_{shrink_factor}.png')
    plt.savefig(fig_path)
    plt.close(fig)  # Close the figure to free memory

# Note: Ensure that the images are moved to CPU and converted to numpy arrays if you're using a GPU for training.



def test(obj_names, mvtec_path, checkpoint_path, base_model_name):
    shrink_factors = [1, 0.8, 0.98, 0.9, 0.95, 0.85]

    factor_stats = {}

    for factor in shrink_factors:
        factor_stats[factor] = {
            'total_image_roc_auc': [],
            'total_pixel_roc_auc': [],
            'ap_image': [],
            'ap_pixel': []
        }

    for obj_name in obj_names:
        img_dim = 256
        run_name = base_model_name + "_" + obj_name + '_'

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name + ".pckl"), map_location='cuda:0'))
        model.cuda()
        model.eval()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(
            torch.load(os.path.join(checkpoint_path, run_name + "_seg.pckl"), map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()

        for factor in shrink_factors:

            dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=256, shrink_factor=factor)
            dataloader = DataLoader(dataset, batch_size=1,
                                    shuffle=False, num_workers=0)

            plot_images_and_save(dataloader, obj_name, factor)

            total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
            total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
            # print('len dataset: ', len(dataset))
            mask_cnt = 0

            anomaly_score_gt = []
            anomaly_score_prediction = []

            display_images = torch.zeros((16, 3, 256, 256)).cuda()
            display_gt_images = torch.zeros((16, 3, 256, 256)).cuda()
            display_out_masks = torch.zeros((16, 1, 256, 256)).cuda()
            display_in_masks = torch.zeros((16, 1, 256, 256)).cuda()
            cnt_display = 0
            display_indices = np.random.randint(len(dataloader), size=(16,))

            for i_batch, sample_batched in enumerate(dataloader):

                gray_batch = sample_batched["image"].cuda()

                is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
                # print(is_normal)
                anomaly_score_gt.append(is_normal)
                true_mask = sample_batched["mask"]
                true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

                gray_rec = model(gray_batch)
                joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                if i_batch in display_indices:
                    t_mask = out_mask_sm[:, 1:, :, :]
                    display_images[cnt_display] = gray_rec[0]
                    display_gt_images[cnt_display] = gray_batch[0]
                    display_out_masks[cnt_display] = t_mask[0]
                    display_in_masks[cnt_display] = true_mask[0]
                    cnt_display += 1

                out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()

                out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                                   padding=21 // 2).cpu().detach().numpy()
                image_score = np.max(out_mask_averaged)

                anomaly_score_prediction.append(image_score)

                flat_true_mask = true_mask_cv.flatten()
                flat_out_mask = out_mask_cv.flatten()
                total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
                total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
                mask_cnt += 1

            anomaly_score_prediction = np.array(anomaly_score_prediction)
            anomaly_score_gt = np.array(anomaly_score_gt)
            auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
            ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

            total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
            total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
            total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
            auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
            ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
            factor_stats[factor]['ap_pixel'].append(ap_pixel)
            factor_stats[factor]['ap_image'].append(ap)
            factor_stats[factor]['total_pixel_roc_auc'].append(auroc_pixel)
            factor_stats[factor]['total_image_roc_auc'].append(auroc)

            print(obj_name, f'shrink factor: {factor}')
            print("AUC Image:  " + str(auroc))
            print("AP Image:  " + str(ap))
            print("AUC Pixel:  " + str(auroc_pixel))
            print("AP Pixel:  " + str(ap_pixel))
            print("==============================")


    for factor in shrink_factors:
        print(f'shrink factor: {factor}')
        print("AUC Image mean:  " + str(np.mean(factor_stats[factor]['total_image_roc_auc'])))
        print("AP Image mean:  " + str(np.mean(factor_stats[factor]['ap_image'])))
        print("AUC Pixel mean:  " + str(np.mean(factor_stats[factor]['total_pixel_roc_auc'])))
        print("AP Pixel mean:  " + str(np.mean(factor_stats[factor]['ap_pixel'])))

    # write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)

    args = parser.parse_args()

    obj_list = ['capsule',
                'bottle',
                'carpet',
                'leather',
                'pill',
                'transistor',
                'tile',
                'cable',
                'zipper',
                'toothbrush',
                'metal_nut',
                'hazelnut',
                'screw',
                'grid',
                'wood'
                ]

    with torch.cuda.device(args.gpu_id):
        test(obj_list, args.data_path, args.checkpoint_path, args.base_model_name)
