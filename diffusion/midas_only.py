import sys
import os
import torch
import numpy as np
import gradio as gr
import argparse
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder

from txt2img import put_watermark
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.util import AddMiDaS

torch.set_grad_enabled(False)
# python3 scripts/midas_only.py -m outputs -d data13 -c configs/stable-diffusion/v2-midas-inference.yaml -ckpt checkpoints/512-depth-ema.ckpt

parser = argparse.ArgumentParser()
parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help='checkpoint path')
parser.add_argument('-c', '--config', type=str, default=None, help='config')
parser.add_argument('-m', '--mode', type=str, default="outputs", help='output or upsample_aug')
parser.add_argument('-d', '--data', type=str, default=None, help='data num')
args = parser.parse_args()
mode = args.mode
data = args.data
assert mode in ["outputs", "upsample_aug"]
print("mode: ", mode)
print("data: ", data)
if mode =="outputs":
    folder_path = "/hy-tmp/hy-tmp/generative-holograph/stablediffusion/"+mode+"/"+data+"/samples"
elif mode =="upsample_aug":
    folder_path = "/hy-tmp/hy-tmp/generative-holograph/stablediffusion/"+mode+"/"+data

save_path = "/hy-tmp/hy-tmp/generative-holograph/stablediffusion/"+mode+"/"+data+"/depth"



def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    return sampler


def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
        model_type="dpt_hybrid"
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    # sample['jpg'] is tensor hwc in [-1, 1] at this point
    midas_trafo = AddMiDaS(model_type=model_type)
    batch = {
        "jpg": image,
        "txt": num_samples * [txt],
    }
    batch = midas_trafo(batch)
    batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
    batch["jpg"] = repeat(batch["jpg"].to(device=device),
                          "1 ... -> n ...", n=num_samples)
    batch["midas_in"] = repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(
        device=device), "1 ... -> n ...", n=num_samples)
    return batch


def paint(sampler, image, prompt, t_enc, seed, scale, num_samples=1, callback=None,
          do_full_sample=False):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    seed_everything(seed)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    with torch.no_grad(),\
            torch.autocast("cuda"):
        batch = make_batch_sd(
            image, txt=prompt, device=device, num_samples=num_samples)
        z = model.get_first_stage_encoding(model.encode_first_stage(
            batch[model.first_stage_key]))  # move to latent space
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck]
            cc = model.depth_model(cc)
            depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                           keepdim=True)
            display_depth = (cc - depth_min) / (depth_max - depth_min)
            depth_image = Image.fromarray(
                (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
            cc = torch.nn.functional.interpolate(
                cc,
                size=z.shape[2:],
                mode="bicubic",
                align_corners=False,
            )
            depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                           keepdim=True)
            cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
            c_cat.append(cc)
            
    return depth_image


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded


def predict(input_image, prompt, steps, num_samples, scale, seed, eta, strength):
    init_image = input_image.convert("RGB")
    image = pad_image(init_image)  # resize to integer multiple of 32

    sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    do_full_sample = strength == 1.
    t_enc = min(int(strength * steps), steps-1)
    result = paint(
        sampler=sampler,
        image=image,
        prompt=prompt,
        t_enc=t_enc,
        seed=seed,
        scale=scale,
        num_samples=num_samples,
        callback=None,
        do_full_sample=do_full_sample
    )
    return result


sampler = initialize_model(args.config, args.checkpoint)

# Path to the folder with the images


os.makedirs(save_path, exist_ok=True)
prompt = ""
num_samples = 1
ddim_steps = 50
scale = 9.0
strength =0.9
seed = 5400
eta = 0.0

# Loop through the images in the folder and load each one
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image using PIL
        img_path = os.path.join(folder_path, filename)
        input_image = Image.open(img_path)
        depth = predict(input_image, prompt, ddim_steps, num_samples, scale, seed, eta, strength)
        depth.save(save_path+"/"+filename[:-4]+"_depth.png")
