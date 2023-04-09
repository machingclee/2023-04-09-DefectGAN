from dis import dis
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import config

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import Discriminator, Generator
from utils import ConsoleLog, add_sn_, initialize_weights, gradient_penalty, view_intermediate_result
from dataset import defectDataloader, defect_dataset
from device import device
from config import DiscLambdas, GenLambdas, save_config
from config import ModelAndTrainingConfig as config

console_log = ConsoleLog(lines_up_on_end=1)
torch.autograd.set_detect_anomaly(True)


def train(gen, disc, start_epoch=1):
    data_gen = iter(defectDataloader)

    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=config.betas)
    opt_disc = optim.Adam(disc.parameters(), lr=config.lr, betas=config.betas)

    cce = nn.CrossEntropyLoss()
    mae = nn.L1Loss()

    n_batches = defect_dataset.get_num_of_batches()

    for epoch in range(config.epochs):
        epoch = epoch + start_epoch
        batch = 0
        data_gen = iter(defectDataloader)
        print("device in use:", device)

        # if epoch >= 0 and epoch <= 2:
        #     lambda_mask = 5.0
        # elif epoch >= 3 and epoch <= 5:
        #     lambda_mask = 0.5
        # else:
        #     lambda_mask = 0

        for defect_cls_index, normal_np, defect_np, _, spatial_cat_map in \
                tqdm(data_gen,
                     total=n_batches,
                     desc="Epoch {}".format(epoch),
                     bar_format=config.bar_format):
            batch += 1
            normal_cls_index = torch.as_tensor([0] * config.batch_size).type(torch.LongTensor).to(device)

            for normal, normal_index, defect_index in [
                (normal_np, normal_cls_index, defect_cls_index),
                (defect_np, defect_cls_index, normal_cls_index)
            ]:
                z_1 = torch.randn((config.batch_size,) + config.noise_dim, device=device)
                z_2 = torch.randn((config.batch_size,) + config.noise_dim, device=device)

                # naming convetion is from normal --> defect. Roles of normal and defect can be interchanged
                defect_overlays, n2d_masks = gen(normal, spatial_cat_map, z_1)
                gened_defects = normal * (1 - n2d_masks) + defect_overlays * n2d_masks

                # g_mask_similarity_loss = mae(n2d_masks, spatial_cat_map[int(defect_cls_index.detach().cpu())])

                restore_overlays, d2n_masks = gen(gened_defects, spatial_cat_map, z_2)
                restoration = gened_defects * (1 - d2n_masks) + restore_overlays * d2n_masks

                d_critic_on_gened, d_cls_logit_on_gened = disc(gened_defects)

                g_spatial_cat_similarity_loss = mae(0.5 * (n2d_masks + d2n_masks),
                                                    spatial_cat_map[:, [defect_cls_index.cpu().item()], :, :])

                g_wgan_gp_loss = -torch.mean(d_critic_on_gened)

                g_cls_loss_on_gened = cce(d_cls_logit_on_gened, defect_index)

                g_cycle_loss = mae(normal, restoration)

                g_mask_cycle_loss = mae(n2d_masks, d2n_masks)

                g_mask_vanishing_loss = -torch.log(torch.mean(
                    mae(n2d_masks, torch.zeros_like(n2d_masks, device=device)) +
                    mae(d2n_masks, torch.zeros_like(d2n_masks, device=device))
                ))

                g_mask_spatial_constraint_loss = torch.mean(
                    mae(n2d_masks, torch.zeros_like(n2d_masks, device=device)) +
                    mae(d2n_masks, torch.zeros_like(d2n_masks, device=device))
                )

                g_loss = (
                    GenLambdas.g_spatial_cat_similarity_loss *
                    g_spatial_cat_similarity_loss +

                    GenLambdas.g_cycle_loss
                    * g_cycle_loss +

                    GenLambdas.g_mask_cycle_loss
                    * g_mask_cycle_loss +

                    GenLambdas.g_mask_vanishing_loss
                    * g_mask_vanishing_loss +

                    GenLambdas.g_mask_spatial_constraint_loss
                    * g_mask_spatial_constraint_loss +

                    GenLambdas.g_cls_loss_on_gened
                    * g_cls_loss_on_gened +

                    GenLambdas.g_wgan_gp_loss
                    * g_wgan_gp_loss
                )

                opt_gen.zero_grad()
                g_loss.backward(inputs=list(gen.parameters()), retain_graph=True)
                opt_gen.step()

                d_critic_on_normal, d_cls_logic_on_normal = disc(normal)

                gp = gradient_penalty(disc, normal, gened_defects)

                d_wgan_gp_loss = torch.mean(d_critic_on_gened) - torch.mean(d_critic_on_normal) + 10 * gp
                cls_loss_on_real = cce(d_cls_logic_on_normal, normal_index)

                d_loss = (
                    DiscLambdas.d_wgan_gp_loss
                    * d_wgan_gp_loss +

                    DiscLambdas.cls_loss_on_real *
                    cls_loss_on_real
                )

                opt_disc.zero_grad()
                d_loss.backward(inputs=list(disc.parameters()))
                opt_disc.step()

            with torch.no_grad():
                console_log.print(
                    [
                        ("", ""),
                        ("d_loss", d_loss.item()),
                        ("- d_wgan_gp_loss", d_wgan_gp_loss.item()),
                        ("- d_cls_loss_on_real", cls_loss_on_real.item()),
                        ("", ""),
                        ("g_loss", g_loss.item()),
                        ("- g_spatial_cat_similarity_loss", g_spatial_cat_similarity_loss.item()),
                        ("- g_cycle_loss", g_cycle_loss.item()),
                        ("- g_mask_cycle_loss", g_mask_cycle_loss.item()),
                        ("- g_mask_vanishing_loss", g_mask_vanishing_loss.item()),
                        ("- g_mask_spatial_constraint_loss", g_mask_spatial_constraint_loss.item()),
                        ("- g_cls_loss_on_gened", g_cls_loss_on_gened.item()),
                        ("- g_wgan_gp_loss", g_wgan_gp_loss.item())
                    ])

            if (batch) % config.n_batch_to_save_checkpoint == 0:
                torch.save(
                    gen,
                    "{}/gen_epoch_{}_batch_{}.pt".format(
                        config.checkpoints_dir,
                        epoch,
                        batch
                    )
                )

                torch.save(
                    disc,
                    "{}/disc_epoch_{}_batch_{}.pt".format(
                        config.checkpoints_dir,
                        epoch,
                        batch
                    )
                )

            if (batch) % config.n_batch_to_sample_img == 0:
                if config.save_sample_image:
                    view_intermediate_result(
                        gen,
                        "epoch_{}_batch_{}.jpg".format(
                            epoch,
                            batch
                        ))

        console_log.clear_log_on_epoch_end()


if __name__ == "__main__":
    save_config()

    if config.gen_model_path is None:
        gen = Generator()
        add_sn_(gen)
        gen = gen.to(device, dtype=torch.float)
        initialize_weights(gen)
    else:
        gen = torch.load(config.gen_model_path)

    if config.disc_model_path is None:
        disc = Discriminator()
        add_sn_(disc)
        disc = disc.to(device, dtype=torch.float)
        initialize_weights(disc)
    else:
        disc = torch.load(config.disc_model_path)

    gen.train()
    disc.train()

    train(gen, disc, start_epoch=config.start_epoch)
