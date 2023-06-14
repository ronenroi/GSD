import numpy as np
from PIL import Image
import glob, os
import matplotlib.pyplot as plt
def image_mosaic(data_path,save_path,Ncols=4):
    if 0:
        xml = glob.glob(os.path.join(os.path.join(data_path, "*.xml")))[0]
        with open(xml, 'r') as fp:
            # read all lines in a list
            lines = fp.readlines()
        for line in lines:
            # check if string present on a current line
            if line.find('<Images>') != -1:
                start_line = lines.index(line)
                break
        lines = lines[start_line:]

        image_paths = [f for f in glob.glob(os.path.join(data_path, "*f01*-ch1*.tiff"))]
        for image_path in image_paths:
            image_path_fields = sorted([f for f in glob.glob(image_path.replace('f01','*'))])
            patches = []
            image_positions = []

            for i_field, image_path_field in enumerate(image_path_fields):
                image_path_field_channels = sorted([f for f in glob.glob(image_path_field.replace('-ch1', '*'))])
                patch_channels = []
                for image_path_field_channel in image_path_field_channels:
                    patch_channels.append(np.array(Image.open(image_path_field_channel)))
                patch_channels = np.stack(patch_channels)
                patch_channels = np.swapaxes(np.array(patch_channels), 0, -1)
                # plt.imshow(patch_channels[...,1])
                # plt.show()
                patches.append(patch_channels)
                for line_index, line in enumerate(lines):
                    # check if string present on a current line
                    if line.find(image_path_field_channel.split('/')[-1]) != -1:
                        file_info = lines[line_index-2:line_index+33]
                        image_res_x=image_res_y=position_x=position_y=None
                        for line in file_info:
                            if line.find('<ImageResolutionX Unit="m">') != -1:
                                image_res_x = float(line.split('<ImageResolutionX Unit="m">')[-1].split('</ImageResolutionX>')[0])
                            elif line.find('<ImageResolutionY Unit="m">') != -1:
                                image_res_y = float(line.split('<ImageResolutionY Unit="m">')[-1].split('</ImageResolutionY>')[0])
                            elif line.find('<PositionX Unit="m">') != -1:
                                position_x = float(line.split('<PositionX Unit="m">')[-1].split('</PositionX>')[0])
                            elif line.find('<PositionY Unit="m">') != -1:
                                position_y = float(line.split('<PositionY Unit="m">')[-1].split('</PositionY>')[0])
                        pixl_x_position = int(np.round(position_x/image_res_x))
                        pixl_y_position = int(np.round(position_y/image_res_y))
                        image_positions.append((pixl_x_position,pixl_y_position))
                        print()

        np.save('image',np.array(patches))
        np.save('position',np.array(image_positions))
    else:
        patches = np.load('image.npy')
        image_positions = np.load('position.npy')

    patches = np.array(patches)
    image_positions = np.array(image_positions)
    image_positions[:, 1] *= -1
    image_positions[:, 0] -= np.min(image_positions[:, 0])
    image_positions[:, 1] -= np.min(image_positions[:, 1])
    x_size = int(np.max(image_positions[:, 0]) + patches.shape[1])
    y_size = int(np.max(image_positions[:, 1]) + patches.shape[2])
    mosaic_image = np.zeros((x_size,y_size,patches.shape[-1]))
    for patch, image_position in zip(patches,image_positions):
        print(np.sum(mosaic_image[image_position[0]:image_position[0]+patches.shape[1],
        image_position[1]:image_position[1]+patches.shape[2],:]))

        mosaic_image[image_position[0]:image_position[0]+patches.shape[1],
        image_position[1]:image_position[1]+patches.shape[2],:] = patch
    A = mosaic_image[:, :, 1:4].astype(np.uint8)
    plt.imshow(A)
    plt.show()

    im = Image.fromarray(A)
    im.show()
    im.save('r02c02.png')
    print()




if __name__ == "__main__":
    data_path = '/home/roironen/GSD/GSD/data/uri-gsd1a-liveexp19r2p2__2022-07-19T16_18_38-Measurement 1'
    save_path = '/home/roironen/GSD/GSD/data/processed_uri-gsd1a-liveexp19r2p2__2022-07-19T16_18_38-Measurement 1'
    image_mosaic(data_path,save_path)


