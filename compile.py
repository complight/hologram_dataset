import odak
import sys
import os
import argparse
import random
from tqdm import tqdm


__title__ = 'Hologram dataset generator'


def dir_path(raw_path):
    if not os.path.isdir(raw_path):
        raise argparse.ArgumentTypeError('"{}" is not an existing directory'.format(raw_path))
    return os.path.abspath(raw_path)


def main():
    settings_filename = './settings/jasper.txt'
    rgb_directory = '~/datasets/div2k_w_depth_4k/train/images/'
    depth_directory = '~/datasets/div2k_w_depth_4k/train/depths/'
    output_dataset_directory='~/datasets/holograms_4k/'
    key='*.png'
    count_offset = 0
    start_offset = 0
    peak_amplitude = 1.8
    generator = 'multi-color'
    parser = argparse.ArgumentParser(description = __title__)
    parser.add_argument(
                        '--settings_filename',
                        type = argparse.FileType('r'),
                        help = 'Filename for the settings file. Default is {}.'.format(settings_filename)
                       )
    parser.add_argument(
                        '--rgb_directory',
                        type = dir_path,
                        help = 'Folder location for RGB images. Default is {}.'.format(rgb_directory)
                       )
    parser.add_argument(
                        '--depth_directory',
                        type = dir_path,
                        help = 'Folder location for depth images. Default is {}.'.format(depth_directory)
                       )
    parser.add_argument(
                        '--key',
                        type = str,
                        help = 'File extension for RGB images and their depth files. Default is {}.'.format(key)
                       )
    parser.add_argument(
                        '--output_dataset_directory',
                        type = str,
                        help = 'Location where you want to save your holograms at the end. Default is {}.'.format(output_dataset_directory)
                       )
    parser.add_argument(
                        '--count_offset',
                        type = float,
                        help = 'If you set this value other than zero, the saved holograms will be use that as the starting id (e.g., data_0015.pt where count_offset is 15). Default is {}.'.format(count_offset)
                       )
    parser.add_argument(
                        '--peak_brightness',
                        type = float,
                        help = 'The peak brightness multiplier. Default is {}.'.format(peak_amplitude)
                       )
    parser.add_argument(
                        '--start_offset',
                        type = float,
                        help = 'If you set this value other than zero, the calculations will start from the Xth file (e.g., when x = 100, it starts from data_0100.pt but not data_0000.pt). Default is {}.'.format(start_offset)
                       )
    parser.add_argument(
                        '--generator',
                        type = str,
                        help = 'Which hologram generator would you like to use, `multi-color` or `conventional`. Default is {}.'.format(generator)
                       )
    args = parser.parse_args()
    if not isinstance(args.peak_brightness, type(None)):
        peak_amplitude = float(args.peak_brightness)
    if not isinstance(args.settings_filename, type(None)):
        settings_filename = str(args.settings_filename.name)
    if not isinstance(args.rgb_directory, type(None)):
        rgb_directory = str(args.rgb_directory)
    if not isinstance(args.depth_directory, type(None)):
        depth_directory = str(args.depth_directory)
    if not isinstance(args.output_dataset_directory, type(None)):
        output_dataset_directory = os.path.expanduser(str(args.output_dataset_directory))
    if not isinstance(args.key, type(None)):
        key = str(args.key)
    if not isinstance(args.count_offset, type(None)):
        count_offset = int(args.count_offset)
    if not isinstance(args.start_offset, type(None)):
        start_offset = int(args.start_offset)
    if not isinstance(args.generator, type(None)):
        generator = str(args.generator)
    print('Generator: ', generator)
    print('Settings filename: ', settings_filename)
    print('RGB images directory: ', rgb_directory)
    print('Depth directory: ', depth_directory)
    print('Count offset: ', count_offset, ' Start offset: ', start_offset)
    print('Output dataset: ', output_dataset_directory)
    process(
            settings_filename = settings_filename,
            rgb_directory = rgb_directory,
            depth_directory = depth_directory,
            key = key,
            output_dataset_directory = output_dataset_directory,
            count_offset = count_offset,
            start_offset = start_offset,
            peak_amplitude = peak_amplitude,
            generator = generator
           )


def process(
            settings_filename,
            rgb_directory,
            depth_directory,
            key,
            output_dataset_directory,
            count_offset,
            start_offset,
            peak_amplitude,
            generator
           ):
    os.chdir('../holohdr')
    print("Current working directory: {0}".format(os.getcwd()))
    settings = odak.tools.load_dictionary(settings_filename)
    rgb_files = odak.tools.list_files(path = rgb_directory, key = key)[start_offset::]
    depth_files = odak.tools.list_files(path = depth_directory, key = key)[start_offset::]
    odak.tools.check_directory(output_dataset_directory)
    random.seed()
    number = random.randint(0, 999999)
    output_directory = '{}{}_'.format(settings["general"]["output directory"], number)
    settings["general"]["output directory"] = output_directory
    if isinstance(rgb_files, type(None)):
        print('No RGB images found.')
        sys.exit()
    if isinstance(depth_files, type(None)):
        print('No depth images found.')
        sys.exit()
    print('RGB images to process: {}'.format(len(rgb_files)))
    print('Depth images to process: {}'.format(len(depth_files)))
    print('Random number: {}'.format(number))
    for rgb_id, rgb_filename in enumerate(rgb_files):
        final_filename = '{}/data_{:04d}.pt'.format(output_dataset_directory, rgb_id + count_offset)
        depth_filename = depth_files[rgb_id]
        settings["target"]["image filename"] = rgb_filename
        settings["target"]["depth filename"] = depth_filename
        settings["target"]["peak amplitude"] = peak_amplitude
        settings["general"]["method"] = generator
        description = 'RGB: {} Depth: {}'.format(rgb_filename, depth_filename)
        print(description)
        results_directory ='{}{}'.format(output_directory, settings["general"]["method"])
        odak.tools.check_directory(results_directory)
        output_data = '{}/data.pt'.format(results_directory)
        print('Output data: {}, clone to: {}'.format(output_data, final_filename))
        if os.path.exists(os.path.expanduser(final_filename)):
            saved_data = odak.learn.tools.torch_load(final_filename)
            laser_powers = saved_data['channel powers']
            laser_powers_filename = '{}/channel_power.pt'.format(results_directory)
            odak.learn.tools.save_torch_tensor(laser_powers_filename, laser_powers)
        else:
            laser_powers_filename = ''
        settings["target"]["laser power filename"] = laser_powers_filename
        print('Laser powers filename: {}'.format(settings["target"]["laser power filename"]))
        dictionary_filename = './settings/cache_{:06d}.txt'.format(number)
        odak.tools.save_dictionary(settings, dictionary_filename)
        cmd = [
               'python3',
               'main.py',
               '--settings',
               '{}'.format(dictionary_filename),
              ]
        odak.tools.shell_command(cmd)
        cmd = [
               'cp',
               '{}'.format(os.path.expanduser(output_data)),
               '{}'.format(os.path.expanduser(final_filename)),
              ]
        odak.tools.shell_command(cmd)
        for m in range(3):
            sys.stdout.write("\x1b[1A\x1b[2K")
    return True


if "__main__" == "__main__":
    sys.exit(main())
