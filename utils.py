import numpy as np
import torch
import h5py
import os
import nibabel as nib
import json
import random
import cv2
#import tensorflow as tf

from typing import Dict, Any, List
from scipy import ndimage
from torch.nn.functional import softmax


class Params:
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """ "
        Save dict to json file

        Parameters
        ----------
        json_path : string
            Path to save location
        """
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Load parameters from json file

        Parameters
        ----------
        json_path : string
            Path to json file
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """ "
        Give dict-like access to Params instance by: 'params.dict['learning_rate']'
        """
        return self.__dict__


def add_contour_to_gt(gt):
   gt = gt.astype(np.uint8)
   gt2 = np.copy(gt)
   gt2[gt2 == 2] = 1
   gt2[gt2 == 3] = 1

   ret, thresh = cv2.threshold(gt2 * 255, 127, 255, 0)

   contours, _ = cv2.findContours(
       thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
   )
   img_contours = np.zeros(gt2.shape)
   for contour in contours:
       if cv2.contourArea(contour) > 8000:  # just a condition
           cv2.drawContours(img_contours, [contour], -1, (255, 255, 255), 1)

   # Morphological operations to connect contour lines
   kernel = np.ones((1, 20), np.uint8)
   img_contours = cv2.dilate(img_contours, kernel, iterations=1)
   img_contours = cv2.erode(img_contours, kernel, iterations=1)

   kernel_vert = np.ones((20, 1), np.uint8)
   img_contours = cv2.dilate(img_contours, kernel_vert, iterations=1)
   img_contours = cv2.erode(img_contours, kernel_vert, iterations=1)

   contours2, _ = cv2.findContours(
       img_contours.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
   )

   if not contours2:  # Check if contours2 is empty
       return None  # Return None to indicate skipping this patient

   img_contours2 = np.zeros(gt2.shape)
   largest_contour = max(contours2, key=cv2.contourArea)
   cv2.drawContours(img_contours2, [largest_contour], -1, (255, 255, 255), 1)

   # Fill contour
   img_contours_fill = np.copy(img_contours2)
   cv2.drawContours(img_contours_fill, [largest_contour], -1, 1, thickness=-1)

   img_contours2[img_contours2 > 0] = 4
   added_gt = np.copy(gt)
   added_gt += img_contours2.astype(np.uint8)
   added_gt[added_gt >= 4] = 4
   return added_gt, img_contours2, img_contours_fill


def normalize_min_max(img):
    """
    Normalize data between 0 and 1.

    Parameters
    ----------
    img : numpy.ndarray
        image.

    Returns
    -------
    img
        Normalized image between 0 and 1.

    """
    a = img - np.min(img)
    b = np.max(img) - np.min(img)
    return np.divide(a, b, np.zeros_like(a), where=b != 0)


def load_dataset(fname, params, norm=True):
    with h5py.File(fname, "r") as f:

        a_group_key2 = list(f.keys())
        # TODO: Changed images dtype
        images = np.zeros([512, 512, len(a_group_key2)], dtype=np.float32)
        labels = np.zeros([512, 512, len(a_group_key2)]).astype(np.int16)
        for i, name in enumerate(a_group_key2):
            image = f[str(name)]["images"][()]
            label = f[str(name)]["labels"][()]
            if norm:
                image = normalize(
                    image[:, :],
                    "True",
                    params.dict["min_bound"],
                    params.dict["max_bound"],
                )
            images[:, :, i] = image.astype(np.float32)

            labels[:, :, i] = label
    print(">>> Unique labels in {}: {}".format(fname, np.unique(labels)))
    return images, labels, a_group_key2


def predict_contour(contour_model, img_src, params, device='cuda'):
    # Convert the source image to a NumPy array, normalize it, then convert back to tensor
    ct = np.copy(img_src)
    ct = normalize(ct, "False", params.dict["min_bound_contour"], params.dict["max_bound_contour"])
    ct = torch.from_numpy(ct).float().to(device)

    # Add batch and channel dimensions
    ct = ct.unsqueeze(0).unsqueeze(0)

    # Predict using the model
    contour_model.eval()
    with torch.no_grad():
        pred = contour_model(ct)

    # Process prediction to get mask
    pred_max = pred.argmax(dim=1, keepdim=True)
    img_mask = pred_max.squeeze().byte()


    unique_mask_values = torch.unique(img_mask)
    #print(f"Unique mask values from prediction: {unique_mask_values.cpu().numpy()}")

    return img_mask.cpu().numpy()


def load_dataset_contour(fname, params, norm=True, device='cuda'):
    # Load the PyTorch model
    model = torch.load(params.dict["contour_model_path"])
    model = model.to(device)

    with h5py.File(fname, "r") as f:
        a_group_key2 = list(f.keys())
        images = np.zeros([512, 512, len(a_group_key2)], dtype=np.float32)
        labels = np.zeros([512, 512, len(a_group_key2)], dtype=np.int16)

        for i, name in enumerate(a_group_key2):
            if i % 200 == 0:
                print(f"Loading: {i} of {len(a_group_key2)}")
            image = f[str(name)]["images"][()]
            label = f[str(name)]["labels"][()]

            unique_labels_before = np.unique(label)
            #print(f"Unique labels before processing for {name}: {unique_labels_before}")


            mask = predict_contour(model, image, params, device)

            if norm:
                image = normalize(image, "True", params.dict["min_bound"], params.dict["max_bound"])

            masked_image = image * mask  # This operates on NumPy arrays
            images[:, :, i] = masked_image
            labels[:, :, i] = label


            unique_labels_after = np.unique(label)
            #print(f"Unique labels after processing for {name}: {unique_labels_after}")

    overall_unique_labels = np.unique(labels)
    print(f">>> Overall unique labels in dataset: {overall_unique_labels}")

    del model
    torch.cuda.empty_cache()
    return images, labels, a_group_key2

def load_dataset_contour_prediction(fname, params, norm=True, device='cuda'):
    # Load the PyTorch model
    model = torch.load(params.dict["contour_model_path"])
    model = model.to(device)

    with h5py.File(fname, "r") as f:
        a_group_key2 = list(f.keys())
        original_images = np.zeros([512, 512, len(a_group_key2)], dtype=np.float32)  # Store original images
        images = np.zeros([512, 512, len(a_group_key2)], dtype=np.float32)  # Store masked images
        labels = np.zeros([512, 512, len(a_group_key2)], dtype=np.int16)

        for i, name in enumerate(a_group_key2):
            if i % 200 == 0:
                print(f"Loading: {i} of {len(a_group_key2)}")
            image = f[str(name)]["images"][()]
            label = f[str(name)]["labels"][()]

            unique_labels_before = np.unique(label)
            #print(f"Unique labels before processing for {name}: {unique_labels_before}")

            mask = predict_contour(model, image, params, device)

            if norm:
                image = normalize(image, "True", params.dict["min_bound"], params.dict["max_bound"])

            masked_image = image * mask  
            original_images[:, :, i] = image  
            images[:, :, i] = masked_image 
            labels[:, :, i] = label

            unique_labels_after = np.unique(label)
            #print(f"Unique labels after processing for {name}: {unique_labels_after}")

    overall_unique_labels = np.unique(labels)
    print(f">>> Overall unique labels in dataset: {overall_unique_labels}")

    del model
    torch.cuda.empty_cache()
    return original_images, images, labels, a_group_key2

def load_dataset_contour_mapping(fname, params, norm=True, device='cuda'):

    model = torch.load(params.dict["contour_model_path"])
    model = model.to(device)

    with h5py.File(fname, "r") as f:
        a_group_key2 = list(f.keys())
        images = np.zeros([512, 512, len(a_group_key2)], dtype=np.float32)
        labels = np.zeros([512, 512, len(a_group_key2)], dtype=np.int16)

        # Define the label mapping
        label_mapping = {0: 0, 1: 1, 5: 2, 7: 3}
        required_labels = set(label_mapping.keys())
        valid_labels = {0, 1, 2, 3}

        for i, name in enumerate(a_group_key2):
            if i % 200 == 0:
                print(f"Loading: {i} of {len(a_group_key2)}")
            image = f[str(name)]["images"][()]
            label = f[str(name)]["labels"][()]

            unique_labels_before = np.unique(label)
            print(f"Unique labels before processing for {name}: {unique_labels_before}")

            mask = predict_contour(model, image, params, device)

            if norm:
                image = normalize(image, "True", params.dict["min_bound"], params.dict["max_bound"])

            masked_image = image * mask  
            images[:, :, i] = masked_image

            # Check and apply if mapping is necessary
            if required_labels.intersection(unique_labels_before):

                mapped_label = np.vectorize(lambda x: label_mapping.get(x, x))(label)
            elif valid_labels.issuperset(unique_labels_before):

                mapped_label = label
            else:

                raise ValueError(f"Unexpected labels {unique_labels_before} found in {name}")

            labels[:, :, i] = mapped_label


            unique_labels_after = np.unique(mapped_label)
            print(f"Unique labels after processing for {name}: {unique_labels_after}")

    overall_unique_labels = np.unique(labels)
    print(f">>> Overall unique labels in dataset: {overall_unique_labels}")

    del model
    torch.cuda.empty_cache()
    return images, labels, a_group_key2



def load_dataset_contour_creation(fname, params, norm=True):
    with h5py.File(fname, "r") as f:
        a_group_key2 = list(f.keys())
        pad_width = 0
        images = np.zeros([512 + pad_width, 512 + pad_width, len(a_group_key2)], dtype=np.float32)
        labels = np.zeros([512 + pad_width, 512 + pad_width, len(a_group_key2)], dtype=np.int16)
        labels_bcs = np.zeros([512 + pad_width, 512 + pad_width, len(a_group_key2)], dtype=np.int16)
        struct = ndimage.generate_binary_structure(2, 2)

        skipped_keys = []  # List to store keys of skipped patients

        for i, name in enumerate(a_group_key2):
            image = f[str(name)]['images'][()]
            label = f[str(name)]['labels'][()]
            if norm:
                image = normalize(image,'True', params.dict['min_bound_contour'], params.dict['max_bound_contour'])
            pad_label = np.pad(label, 20, 'constant', constant_values=0)
            images[:, :, i] = image
            labels_bcs[:, :, i] = label

            result = add_contour_to_gt(pad_label)
            if result is None:  # If no contours found Add this key to the list of skipped patients
                skipped_keys.append(name)  # 
                continue  

            added_gt, img_contours2, img_contours_fill = result
            label = ndimage.binary_dilation(img_contours_fill, structure=struct, iterations=3)
            label = label[20:-20, 20:-20]
            labels[:, :, i] = np.uint8(label)

    print('>>> Unique labels in {}: {}'.format(fname, np.unique(labels)))
    print('Skipped patient keys due to no contours:', skipped_keys)
    return images, labels, labels_bcs, a_group_key2


def load_samples(sample_txt_file, seed=42):
    """
    Load samples from a .txt file, extracts relevant information, splits
    between CT and GT, and shuffles samples according to a specified seed.

    Parameters
    ----------
    sample_txt_file : str
        Text file containing samples.
    seed : int - default = 42
        Seed for shuffling

    Returns
    -------
    samples_dict : dict
        Dictionary containing strings with locations to CT and GT patches.
    """

    with open(sample_txt_file, "r") as infile:
        data = infile.readlines()

        ct_patches = []
        gt_patches = []
        for i in data:
            line = i.strip(",")
            line = line.split(",")
            ct_patches.append(line[0])
            gt_patches.append(line[1])

    array = list(zip(ct_patches, gt_patches))
    random.seed(seed)
    random.shuffle(array)
    ct_patches, gt_patches = zip(*array)
    samples_dict = {"ct_patches": list(ct_patches),
                    "gt_patches": list(gt_patches)}

    return samples_dict


def shuffle_samples(samples, seed=42):
    """

    Parameters
    ----------
    samples : list
        List containing paths to patches
    seed : int
        Seed for shuffling

    Returns
    -------
    Shuffled Samples
    """
    random.seed(seed)
    return random.shuffle(list(samples))


def load_batch(samples_dict, patch_path, iteration, batch_size):
    def _load_batch(sample_list, patch_path_dir):
        batch = []

        for sample_path in sample_list:
            patch = nib.load(os.path.join(patch_path_dir,
                                          sample_path)).get_fdata()
            patch = np.expand_dims(patch, -1)
            batch.append(patch)
        return np.array(batch)

    min_index = (iteration * batch_size) - batch_size
    max_index = iteration * batch_size
    ct_samples = samples_dict["ct_patches"][min_index:max_index]
    gt_samples = samples_dict["gt_patches"][min_index:max_index]

    ct_batch = _load_batch(ct_samples, patch_path)
    gt_batch = _load_batch(gt_samples, patch_path)

    return ct_batch, gt_batch


def normalize(img, bound, min_bound, max_bound):
    """
    Normalize an image between "min_bound" and "max_bound", and scale between
    0 and 1. If "bound" = 'True', scale between 2.5th and 97.5th percentile.

    Parameters
    ----------
    img : np.ndarray
        Image to normalize.
    bound : str - True or False.
        Whether to scale between percentiles.
    min_bound : int
        Lower bound for normalization.
    max_bound : int
        Upper bound for normalization.

    Returns
    -------
    img : np.ndarray
        Normalized and scaled image.
    """
    img = (img - min_bound) / (max_bound - min_bound)
    img[img > 1] = 0
    img[img < 0] = 0
    # if bound == 'True':
    # norm = 2.5
    #   mn = np.percentile(img, norm)
    #  mx = np.percentile(img, 100 - norm)
    # a = (img - mn)
    # b = (mx - mn)
    # img = np.divide(a, b, np.zeros_like(a), where=b != 0)
    # print(np.min(img))
    # print(np.max(img))
    c = img - np.min(img)
    d = np.max(img) - np.min(img)
    img = np.divide(c, d, np.zeros_like(c), where=d != 0)

    return img

class ColorMap:
    def __init__(self, name: str) -> None:
        self._name = name
        self._values = []

    def name(self) -> str:
        return self._name
    
    def values(self) -> List[List[int]]:
        return self._values

class AlbertaColorMap(ColorMap):
    def __init__(self) -> None:
        super(AlbertaColorMap, self).__init__(name='AlbertaColorMap')
        for i in range(256):
            if i == 1:  # muscle
                self.values().append([255, 0, 0])
            elif i == 2:  # inter-muscular adipose tissue
                self.values().append([0, 255, 0])
            elif i == 5:  # visceral adipose tissue
                self.values().append([255, 255, 0])
            elif i == 7:  # subcutaneous adipose tissue
                self.values().append([0, 255, 255])
            elif i == 12:  # unknown
                self.values().append([0, 0, 255])
            else:
                self.values().append([0, 0, 0])


def applyColorMap(pixels: np.array, colorMap: ColorMap) -> np.array:
    pixelsNew = np.zeros((*pixels.shape, 3), dtype=np.uint8)
    np.take(colorMap.values(), pixels, axis=0, out=pixelsNew)
    return pixelsNew

def calculateArea(labels, label, pixelSpacing):
    mask = np.copy(labels)
    mask[mask != label] = 0
    mask[mask == label] = 1
    area = np.sum(mask) * (pixelSpacing[0] * pixelSpacing[1]) / 100.0
    return area
