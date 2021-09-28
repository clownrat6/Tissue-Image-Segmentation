import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


def colorize_instance_map(instance_map):
    """using random rgb color to colorize instance map."""
    colorful_instance_map = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    instance_id_list = list(np.unique(instance_map))
    instance_id_list.remove(0)
    for instance_id in instance_id_list:
        r, g, b = [random.random() * 255 for i in range(3)]
        colorful_instance_map[instance_map == instance_id, :] = (r, g, b)

    return colorful_instance_map


def draw_semantic(save_folder, data_id, image, pred, label,
                  single_loop_results):
    """draw semantic level picture with FP & FN."""

    plt.figure(figsize=(5 * 2, 5 * 2 + 3))

    # prediction drawing
    plt.subplot(221)
    plt.imshow(pred)
    plt.axis('off')
    plt.title('Prediction', fontsize=15, color='black')

    # ground truth drawing
    plt.subplot(222)
    plt.imshow(label)
    plt.axis('off')
    plt.title('Ground Truth', fontsize=15, color='black')

    # image drawing
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(223)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Image', fontsize=15, color='black')

    canvas = np.zeros((*pred.shape, 3), dtype=np.uint8)
    canvas[label == 1, :] = (255, 255, 2)
    canvas[(pred == 0) * (label == 1), :] = (2, 255, 255)
    canvas[(pred == 1) * (label == 0), :] = (255, 2, 255)
    plt.subplot(224)
    plt.imshow(canvas)
    plt.axis('off')
    plt.title('FN-FP-Ground Truth', fontsize=15, color='black')

    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [(255, 255, 2), (2, 255, 255), (255, 2, 255)]
    label_list = [
        'Ground Truth',
        'FN',
        'FP',
    ]
    for color, label in zip(colors, label_list):
        color = list(color)
        color = [x / 255 for x in color]
        plt.plot(0, 0, '-', color=tuple(color), label=label)
    plt.legend(loc='upper center', fontsize=9, bbox_to_anchor=(0.5, 0), ncol=3)

    # results visulization
    aji = f'{single_loop_results["Aji"] * 100:.2f}'
    dice = f'{single_loop_results["Dice"] * 100:.2f}'
    recall = f'{single_loop_results["Recall"] * 100:.2f}'
    precision = f'{single_loop_results["Precision"] * 100:.2f}'
    temp_str = (f'Aji: {aji:<10}\nDice: '
                f'{dice:<10}\nRecall: {recall:<10}\nPrecision: '
                f'{precision:<10}')
    plt.suptitle(temp_str, fontsize=15, color='black')
    plt.tight_layout()
    plt.savefig(
        f'{save_folder}/{data_id}_monuseg_semantic_compare.png', dpi=300)


def draw_instance(save_folder, data_id, pred_instance, label_instance):
    """draw instance level picture."""

    plt.figure(figsize=(5 * 2, 5))

    plt.subplot(121)
    plt.imshow(colorize_instance_map(pred_instance))
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(colorize_instance_map(label_instance))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(
        f'{save_folder}/{data_id}_monuseg_instance_compare.png', dpi=300)
