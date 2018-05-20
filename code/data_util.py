import os
import shutil
import copy
import re
import xml.etree.ElementTree as ET
import skimage, skimage.io, skimage.color, skimage.draw, skimage.transform
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings

import PIL

class_map = {'background': 0, 'Masonry': 6, 'Trim': 3, 'Corner Trim': 10, 'Door': 8, 'Window': 7,
             'Shutter': 9, 'Siding Main': 2, 'Siding Accent': 1, 'Roofing': 4, 'Foreground': 5, 'Siding': 11}
color_table = np.array([[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0], [255, 0, 128],
                        [128, 255, 0], [128, 0, 255], [128, 128, 0], [128, 0, 128], [0, 128, 128], [255, 100, 100],
                        [100, 255, 100], [100, 100, 255]])


def process_mask_annotation(data_path, layer, class_id, anno_image):
    """ Process layers with mask image annotation. """
    mask_path = '{}/mask_{}.png'.format(data_path, layer.attrib['layerID'])
    mask_img = skimage.io.imread(mask_path, as_grey=True)
    bool_mask = mask_img > 0
    offx, offy = int(layer.attrib['maskX']), int(layer.attrib['maskY'])
    mw, mh = mask_img.shape[1], mask_img.shape[0]
    anno_image[offy:offy + mh, offx:offx + mw][bool_mask] = class_id
    return True


def process_contour_annotation(layer, class_id, anno_image):
    """ Process layers with contour annotation. """
    warnings.warn('Not implemented')
    return True


def process_ppoints_annotation(layer, class_id, anno_image):
    """ Process layers with perspective annotation. """
    ppts = np.empty([4, 2])
    for i in range(1, 5):
        if 'perspectiveP{}'.format(i) not in layer.attrib:
            # warnings.warn('Perpective coordinate not exists')
            return False
        pt = layer.attrib['perspectiveP{}'.format(i)].strip().split(',')
        ppts[i - 1] = [float(pt[0]), float(pt[1])]
    rids, cids = skimage.draw.polygon(ppts[:, 1], ppts[:, 0], shape=anno_image.shape[:2])
    anno_image[rids, cids] = class_id
    return True


def str_to_points(pts_str):
    """ Parse coordinates encoded inside a string. """
    pts = pts_str.strip().split(':')
    vert = np.empty([len(pts), 2], dtype=np.int)
    for i in range(len(pts)):
        coord = pts[i].split(',')
        vert[i] = [int(coord[0]), int(coord[1])]
    return vert


def process_polygon_annotation(layer, class_id, anno_image):
    """ Parse layers with polygon annotation. """
    binary_mask = np.zeros(anno_image.shape, dtype=np.bool)
    for polygon in layer.findall('polygon'):
        vert = str_to_points(polygon.attrib['points'])
        rids, cids = skimage.draw.polygon(vert[:, 1], vert[:, 0], shape=anno_image.shape[:2])
        binary_mask[rids, cids] = True
        for hole in polygon.findall('hole'):
            hole_vert = str_to_points(hole.attrib['points'])
            rids, cids = skimage.draw.polygon(hole_vert[:, 1], hole_vert[:, 0], shape=anno_image.shape[:2])
            binary_mask[rids, cids] = False
    anno_image[binary_mask] = class_id
    return True


def parse_annotation(data_path, class_map):
    """ Given an annotation xml file, return a image of the same size. """
    if not os.path.exists(os.path.join(data_path, 'project.xml')):
        warnings.warn('No project.xml found in ' + data_path)
        return None
    anno_xml = ET.parse(data_path + '/project.xml').getroot()
    width, height = int(anno_xml.attrib['wid']), int(anno_xml.attrib['hgt'])
    anno_image = np.zeros([height, width], dtype=np.int)
    for layer in anno_xml.find('layers'):
        try:
            class_name = 'background'
            if 'type' in layer.attrib:
                class_name = layer.attrib['type']
            elif 'layerName' in layer.attrib:
                class_name = layer.attrib['layerName']
            else:
                continue
            class_id = class_map[class_name]
            # The annotation can be in either mask image,
            if os.path.exists('{}/mask_{}.png'.format(data_path, layer.attrib['layerID'])):
                process_mask_annotation(data_path, layer, class_id, anno_image)
            elif layer.find('polygon') is not None:
                process_polygon_annotation(layer, class_id, anno_image)
            elif layer.find('contour') is not None:
                process_contour_annotation(layer, class_id, anno_image)
            elif 'perspectiveP1' in layer.attrib:
                process_ppoints_annotation(layer, class_id, anno_image)
        except Exception as e:
            print('Exception in {}: {}'.format(data_path, e))
    return anno_image


def annotation_to_image(annotation, ctable, draw_contour=True):
    """ Visualize the annotation mask with color table. """
    img = np.zeros([annotation.shape[0], annotation.shape[1], 3], dtype=np.uint8)
    for i in range(np.amax(annotation.ravel())):
        img[annotation == i] = ctable[i % ctable.shape[0]]
        if draw_contour:
            _, contours, _ = cv2.findContours((annotation == i).astype(np.uint8),
                                              cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 0, 0), 2)
    return img


def build_dataset_info(dataset_dir, class_map):
    """
    Scan a directory to samples and write the meta information into a json file. Each sample inside the "dataset_dir"
    should contain an image file "%05d.jpg" and an annotation file "%05d_anno.npy".
    """
    file_list = os.listdir(dataset_dir)
    sample_list = [s[:5] for s in file_list if re.match(r'[0-9]{5}\.jpg', s)]

    output_json = {'source': 'renoworks', 'class_map': class_map, 'samples': []}
    for sample in sample_list:
        if os.path.exists('{}/{}.jpg'.format(dataset_dir, sample)) and os.path.exists(
                '{}/{}_anno.npy'.format(dataset_dir, sample)):
            img = skimage.io.imread('{}/{}.jpg'.format(dataset_dir, sample))
            output_json['samples'].append({
                'id': sample,
                'image': '{}.jpg'.format(sample),
                'annotation': '{}_anno.npy'.format(sample),
                'source': 'N/A',
                'width': img.shape[1],
                'height': img.shape[0]
            })

    with open(dataset_dir + '/dataset.json', 'w') as f:
        json.dump(output_json, f)


def compute_dataset_statistics(data_dir, img_subsample=5, pixel_subsample=10):
    """
    This function computes mean and standard variance of all pixels in the entire dataset, including training,
    validation and testing. To speed up the process, not all images and pixels are used. This is controlled by
    "img_subsample" and "pixel_subsample". The result will be written into a json file under the dataset root.
    """
    all_pixel_list = []
    for phase in ['train', 'validation', 'test']:
        with open(os.path.join(data_dir, phase, 'dataset.json')) as f:
            dataset_info = json.load(f)
        for sample in dataset_info['samples'][::img_subsample]:
            img = skimage.io.imread(os.path.join(data_dir, phase, sample['image'])).astype(np.float) / 255.0
            img = img.reshape((img.shape[0] * img.shape[1], 3)).tolist()
            all_pixel_list.extend(img[::pixel_subsample])

    all_pixel_list = np.array(all_pixel_list)
    mean = np.average(all_pixel_list, axis=0)
    std = np.std(all_pixel_list, axis=0)

    out_json = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(os.path.join(data_dir, 'info.json'), 'w') as f:
        json.dump(out_json, f)
