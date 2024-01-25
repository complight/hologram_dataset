# Description
This repository contains scripts to generate holograms using images and their depth maps.

## Installing required libraries
Please make sure that you have the latest `odak` installed on your system.
You can find more information on installing `odak` in [this link](https://github.com/kaanaksit/odak).

## Preparing images and depth maps
Make sure that you have images and their depth maps in separate folders.
In our case, we keep a dataset directory in our home directory, where there are `images` and `depths` folders inside it.
For example, we have a directory called `~/datasets/div2k_w_depth_4k`, and there are `~/datasets/div2k_w_depth_4k/images` and `~/datasets/div2k_w_depth_4k/depths` folders inside it.
To create these folders, we downloaded `DIV2K` dataset from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip).
We used a monocular depth estimation model, [DenseDepth](https://github.com/ialhashim/DenseDepth), to predict the depth maps of our images.
Once we have the `images` and `depths` folder filled, we resize these images to the required resolution by our spatial light modulator.
Below, you find a script to resize all the images and depths, which we placed inside `~/datasets/div2k_w_depth_4k/`.

```bash
for f in `find . -name "*.png"`
do
    mogrify -resize 4094x2400! $f
done
```

Note that you can install `mogrify` with a simple `sudo apt-get install mogrify`.
Note that this codebase likes working with `png` files.

## Hologram generation
Now that we have images and their corresponding depths, we can go ahead and generate holograms using `realistic_defocus` library located in [this link](https://github.com/complight/realistic_defocus).
Imagine you have `hologram_dataset` repository under `~/repos/hologram_dataset`, in this case, make sure to clone `realistic_defocus` under `~/repos/realistic_defocus`.
For example, in `realistic_defocus`, you will find two settings file for two different spatial light modulators as `settings/holoeye.txt` and `settings/jasper.txt`.
If you are also using the same spatial light modulator, feel free to use them or modify them for your needs.

Our next step is to plug everything together, let's say you are interested in generating hologram at `4094 x 2400` resolution.
To do this, there are some variables that you have to carefully adjust for your case:

```bash
--settings_filename
--rgb_directory
--depth_directory
--key png
--output_dataset_directory
--generator
```

These are the variables that you want to adjust in that `compile_set_4k.py`. 
`settings_filename` should point to the location where your settings file is.
This is typically a settings file for `realistic_defocus`, such as `settings/jasper.txt` under `realistic_defocus`.
`rgb_directory` is the location where your `images` are.
`depth_directory` is the location where your `depths` are.
`key` defines the file extension of your images and depth files.
`generator` defines the generator that you want to use for generating your dataset (`holohdr` or `realistic_defocus`).
Finally, `output_dataset_directory` is the location where you will be saving all the holograms.
For remaining keys that we haven't explained here, please consult `python3 main.py --help`.


Now that you have configured everything as intended, you can start generating holograms by using the following syntax in your shell:

```bash
CUDA_VISIBLE_DEVICES=1 python3 compile.py --settings_filename ../realistic_defocus/settings/jasper.txt --rgb_directory ~/datasets/div2k_w_depth_4k/train/images/ --depth_directory ~/datasets/div2k_w_depth_4k/train/depths/ --key *.png --output_dataset_directory ~/datasets/holograms_4k/ --count_offset 0 --generator realistic_defocus
```

or for `holohdr`, you can use the following:

```bash
python3 compile.py --settings_filename ../holohdr/settings/jasper.txt --rgb_directory ~/datasets/div2k_w_depth_4k/train/images/ --depth_directory ~/datasets/div2k_w_depth_4k/train/depths/ --output_dataset_directory ~/datasets/holograms_jasper_rgb_conventional --key *.png --count_offset 0 --generator holohdr
```


Note that `CUDA_VISIBLE_DEVICES` helps you to choose the GPU that you want to use for this calculation.
Please also remember that the default GPU is `zero`, `CUDA_VISIBLE_DEVICES=0`.
In the above example, the second GPU is used.
For example, if you need to generate a hologram dataset in high-definition for holoeye spatilal light modulators, you can use the following command:

```bash
python3 compile.py --settings_filename ../realistic_defocus/settings/holoeye.txt --rgb_directory ~/datasets/div2k_w_depth_hd/train/images/ --depth_directory ~/datasets/div2k_w_depth_hd/train/depths/ --key *.png --output_dataset_directory ~/datasets/holograms_hd/ --count_offset 0 --generator realistic_defocus
```

or for `holohdr`, you can use the following:

```bash
python3 compile.py --settings_filename ../holohdr/settings/holoeye.txt --rgb_directory ~/datasets/div2k_w_depth_hd/train/images/ --depth_directory ~/datasets/div2k_w_depth_hd/train/depths/ --output_dataset_directory ~/datasets/holograms_hd_rgb_conventional --key *.png --count_offset 0 --generator holohdr 
```

## Support
We tested the explained workflow in an Ubuntu operating system.
If you have any questions regarding the descriptions and usage, please do not hesitate to use `issues` section for your queries.
