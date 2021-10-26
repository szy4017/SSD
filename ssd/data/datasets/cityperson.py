import os
import torch.utils.data
import numpy as np
from PIL import Image
import json

from ssd.structures.container import Container


def get_valid_person(json_path):
    """Get the valid person object, which needs to satisfy the following conditions.
    *: The label of the object is pedestrian.
    *: The height of pedestrian is more than 50 pixel.
    *: The ridio of pedestrain is less than 0.35 (not sure)
    Args:
        json_path: the path of json file that will be decoded

    Retuens:
        valid_person: a dictionary and the keys of this dict is 'valid' and 'bbox'
        the value of 'valid' is a bool that valid == True means we get a valid person data
        the value of 'bbox_list' is a list that contains y_min, x_min, y_max, x_max
    """
    valid_person = dict()
    bbox_list = list()
    label_list = list()
    person_label = {'pedestrian': 1, 'rider': 2}
    with open(json_path, 'r') as label_json:
        annot = json.load(label_json)
    valid_index = 0
    for obj in annot['objects']:
        if obj['label'] == 'pedestrian':
            x, y, w, h = obj['bbox']
            if h > 60:
                x_vis, y_vis, w_vis, h_vis = obj['bboxVis']
                ratio = 1 - (w_vis * h_vis) / (w * h)
                if ratio < 0.35:
                    valid_index += 1
                    bbox = [y-1, x-1, y-1+h, x-1+w]
                    bbox_list.append(bbox)
                    label_list.append(person_label[obj['label']])
    if valid_index > 0:
        valid = True
    else:
        valid = False
    valid_person['valid'] = valid
    valid_person['bbox_list'] = bbox_list
    valid_person['label_list'] = label_list
    return valid_person


def get_data_list(img_path, json_path, split='train'):
    """Get the images and bbox labels lists, which contain people.
    Args:
        img_path: the path of images that contain people, actually the path is '/cityscape/leftImg8bit'
        label_path: the path of bbox labels that contain people, actually the path is '/cityscape/gtBboxCityPersons'

    Returns:
        valid image list and label list
    """

    img_list, json_list = list(), list()
    if split not in ['train', 'val']:
        return print("You should check the parameter of split!")
    else:
        i_dir = os.path.join(img_path, split)
        j_dir = os.path.join(json_path, split)
        city_list = sorted(os.listdir(j_dir))
        for city in city_list:
            i_city_dir = os.path.join(i_dir, city)
            j_city_dir = os.path.join(j_dir, city)
            js_list = sorted(os.listdir(j_city_dir))
            for js in js_list:
                label_json_path = os.path.join(j_city_dir, js)
                if get_valid_person(label_json_path)['valid'] == True:
                    json_name = label_json_path
                    img_name = os.path.join(i_city_dir, js).replace('gtBboxCityPersons.json', 'leftImg8bit.png')
                    img_list.append(img_name)
                    json_list.append(json_name)

        return img_list, json_list


class CitypersonDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'pedestrian')
    data_dir = '/home/szy/data/cityscape'
    #img_path = '/home/szy/data/cityscape/leftImg8bit'
    #json_path = '/home/szy/data/cityscape/gtBboxCityPersons'

    def __init__(self, data_dir, split, transform=None, target_transform=None, keep_difficult=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult
        self.img_path = os.path.join(data_dir, 'leftImg8bit')
        self.json_path = os.path.join(data_dir, 'gtBboxCityPersons')
        self.img_list, self.json_list = get_data_list(self.img_path, self.json_path, split=split)

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        boxes, labels, is_difficult = self._get_annotation(index)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(index)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        target = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, target, index

    def __len__(self):
        return len(self.img_list)

    def _get_annotation(self, image_id):
        annotation_file = self.json_list[image_id]
        bbox_list = get_valid_person(annotation_file)['bbox_list']
        label_list = get_valid_person(annotation_file)['label_list']
        boxes = np.array(bbox_list, dtype=np.float32)
        labels = np.array(label_list, dtype=np.int64)
        is_difficult = np.zeros(labels.size, dtype=np.int8)
        return boxes, labels, is_difficult

    def _read_image(self, image_id):
        image_file = self.img_list[image_id]
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image



if __name__ == '__main__':
    img_path = '/home/szy/data/cityscape/leftImg8bit'
    json_path = '/home/szy/data/cityscape/gtBboxCityPersons'
    img_list, json_list = get_data_list(img_path, json_path)
    print(img_list)
    data_dir = '/home/szy/data/cityscape'
    cityperson = CitypersonDataset(data_dir, split='train')
    image, target, index = cityperson.__getitem__(0)
    print('done')