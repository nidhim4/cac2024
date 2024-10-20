from PIL import Image, ImageOps
import numpy as np

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


def is_valid_color(rgb):
    return not (rgb == (0, 0, 0) or rgb == (255, 255, 255))


def give_most_hex(file_path, code='hex'):
    my_image = Image.open(file_path).convert('RGB')
    size = my_image.size

    if size[0] >= 1200 or size[1] >= 1200:
        my_image = ImageOps.scale(my_image, factor=0.6)
    elif size[0] >= 800 or size[1] >= 800:
        my_image = ImageOps.scale(my_image, factor=0.5)
    elif size[0] >= 600 or size[1] >= 600:
        my_image = ImageOps.scale(my_image, factor=0.4)
    elif size[0] >= 400 or size[1] >= 400:
        my_image = ImageOps.scale(my_image, factor=0.2)

    my_image = ImageOps.posterize(my_image, 2)

    image_array = np.array(my_image)

    unique_colors = {}
    for column in image_array:
        for rgb in column:
            t_rgb = tuple(rgb)
            if t_rgb not in unique_colors:
                unique_colors[t_rgb] = 1
            else:
                unique_colors[t_rgb] += 1

    sorted_unique_colors = sorted(unique_colors.items(), key=lambda x: x[1], reverse=True)

    filtered_colors = [color for color, count in sorted_unique_colors if is_valid_color(color)]

    top_6 = filtered_colors[0:6]

    if code == 'hex':
        hex_list = [rgb_to_hex(rgb) for rgb in top_6]
        return hex_list
    else:
        return top_6