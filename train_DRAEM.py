import torch

import data_loader
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def test_model(model, model_seg):
    obj_auroc_image_list = []

    img_dim = 256

    dataset = data_loader.MVTecDRAEMTestDataset("/kaggle/input/mvtec-ad/toothbrush/test/",
                                                resize_shape=[img_dim, img_dim])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
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
        anomaly_score_gt.append(is_normal)
        # true_mask = sample_batched["mask"]
        # true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

        gray_rec = model(gray_batch)
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        if i_batch in display_indices:
            t_mask = out_mask_sm[:, 1:, :, :]
            display_images[cnt_display] = gray_rec[0]
            display_gt_images[cnt_display] = gray_batch[0]
            display_out_masks[cnt_display] = t_mask[0]
            # display_in_masks[cnt_display] = true_mask[0]
            cnt_display += 1

        out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)

        anomaly_score_prediction.append(image_score)

        # flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        # total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)

    obj_auroc_image_list.append(auroc)
    print("AUC Image:  " + str(auroc))

    print("==============================")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(obj_names, args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = 'DRAEM_test'

    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.cuda()
    model.apply(weights_init)

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.cuda()
    model_seg.apply(weights_init)

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr},
        {"params": model_seg.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.8, args.epochs * 0.9], gamma=0.2,
                                               last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()

    dataset = MVTecDRAEMTrainDataset(args.data_path, args.anomaly_source_path, resize_shape=[256, 256])

    dataloader = DataLoader(dataset, batch_size=args.bs,
                            shuffle=True, num_workers=16)

    # dataloader = data_loader.get_isic_loader()

    n_iter = 0
    e_num = 0
    l = 0
    for epoch in tqdm(range(args.epochs), desc='Epochs Progress'):
        e_num += 1
        if e_num%5==0:
            test_model(model, model_seg)
        tqdm.write(f"Epoch: {epoch}")

        for i_batch, sample_batched in enumerate(tqdm(dataloader, desc=f'Batch Progress', leave=True, position=0)):
            gray_batch = sample_batched["image"].cuda()
            aug_gray_batch = sample_batched["augmented_image"].cuda()
            anomaly_mask = sample_batched["anomaly_mask"].cuda()

            gray_rec = model(aug_gray_batch)
            joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            l2_loss = loss_l2(gray_rec, gray_batch)
            ssim_loss = loss_ssim(gray_rec, gray_batch)

            segment_loss = loss_focal(out_mask_sm, anomaly_mask)
            loss = l2_loss + ssim_loss + segment_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            l = loss.item()

            if args.visualize and n_iter % 200 == 0:
                visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
            if args.visualize and n_iter % 400 == 0:
                t_mask = out_mask_sm[:, 1:, :, :]
                visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='batch_augmented')
                visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
                visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
                visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')
                visualizer.visualize_image_batch(t_mask, n_iter, image_name='mask_out')

            n_iter += 1

        scheduler.step()

        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl"))
        torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name + "_seg.pckl"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    if int(args.obj_id) == -1:
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
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)


