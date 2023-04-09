import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import config
import numpy as np

from dataset import defectDataloader
from device import device
from models import Generator
from pydash.objects import get, set_
from torch.nn.utils import spectral_norm
from config import ModelAndTrainingConfig as config

# TODO:  fix the error layer
plt.rcParams.update({'figure.max_open_warning': 0})


def _add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m


def add_sn_(model: nn.Module):
    model.apply(_add_sn)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight.data)


def swap_axes(np_img):
    return np_img.swapaxes(0, 1).swapaxes(1, 2)


def gradient_penalty(critic, real, fake):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    critic_inter, _ = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=critic_inter,
        grad_outputs=torch.ones_like(critic_inter),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
    return gradient_penalty


def view_intermediate_result(gen, image_name):
    plt.figure(figsize=(17, 10))
    for i in range(0, 1):
        with torch.no_grad():
            data_gen = iter(defectDataloader)
            defect_cls_index, normals, defects, defect_mask, spatial_cat_maps = next(data_gen)

            _, _, _, _, spatial_cat_map_ = next(data_gen)
            spatial_cat_map_randomSeg = spatial_cat_map_[0]

            spatial_cat_map = spatial_cat_maps[i]
            normal = normals[i]
            defect = defects[i]

            gen.eval()

            for row_count, (from_, spa_cat, description) in enumerate([
                (defect, spatial_cat_map, "from_defect"),
                (normal, spatial_cat_map, "from_normal"),
                (normal, spatial_cat_map_randomSeg, "from_random_segment")
            ]):
                z = torch.randn((1, ) + config.noise_dim, dtype=torch.float, device=device)

                top, m = gen(
                    from_[None, ...],
                    spa_cat[None, ...],
                    z)

                top_layer = swap_axes(np.squeeze(top.cpu().numpy()))
                m = swap_axes(m.cpu().numpy()[0])

                defect_index = int(defect_cls_index[i])

                from_ = swap_axes(from_.cpu().numpy())
                spa_cat = swap_axes(spa_cat.cpu().numpy())

                plt.subplot(3, 5, 5 * row_count + 1)
                plt.imshow(((from_ + 1) * 127.5).astype("uint8"))

                plt.subplot(3, 5, 5 * row_count + 2)
                plt.imshow(spa_cat[:, :, defect_index])

                plt.subplot(3, 5, 5 * row_count + 3)

                plt.imshow(((top_layer + 1) * 127.5).astype("uint8"))

                plt.subplot(3, 5, 5 * row_count + 4)
                plt.imshow(np.squeeze(m))

                plt.subplot(3, 5, 5 * row_count + 5)
                gened_defects = from_ * (1 - m) + top_layer * m
                plt.imshow(((gened_defects + 1) * 127.5).astype("uint8"))

            if image_name is not None:
                plt.savefig("{}/{}".format(config.result_img_dir, image_name),
                            dpi=80,
                            bbox_inches="tight")
                plt.savefig("{}/{}".format(config.result_img_dir, "latest_output.jpg"),
                            dpi=80,
                            bbox_inches="tight")

            gen.train()


class ConsoleLog():
    def __init__(self, lines_up_on_end=0):
        self.CLR = "\x1B[0K"
        self.lines_up_on_batch_end = lines_up_on_end
        self.record = {}

    def UP(self, lines):
        return "\x1B[" + str(lines + 1) + "A"

    def DOWN(self, lines):
        return "\x1B[" + str(lines) + "B"

    def on_print_end(self):
        print(self.UP(self.lines_of_log))
        print(self.UP(self.lines_up_on_batch_end))

    def print(self, key_values):
        lines_of_log = len(key_values)
        self.lines_of_log = lines_of_log

        print("".join(["\n"] * (self.lines_of_log)))
        print(self.UP(self.lines_of_log))

        for key, value in key_values:
            if key == "" and value == "":
                print()
            else:
                if key != "" and value != "":
                    prev_value = get(self.record, key, 0.)
                    curr_value = value
                    diff = curr_value - prev_value
                    sign = "+" if diff >= 0 else ""
                    print("{0: <35}: {1: <30}".format(key, value) + sign + "{:.5f}".format(diff) + self.CLR)
                    set_(self.record, key, value)

        self.on_print_end()

    def clear_log_on_epoch_end(self):
        # usually before calling this line, print() has been run, therefore we are at the top of the log.
        for _ in range(self.lines_of_log):
            # clear lines
            print(self.CLR)
        # ready for next epoch
        print(self.UP(self.lines_of_log))


if __name__ == "__main__":
    gen = Generator()
    gen.init_layers()
    gen = gen.to(device, dtype=torch.float)

    view_intermediate_result(gen, "test.jpg")
