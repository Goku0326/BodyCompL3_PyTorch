import numpy as np
import random
import cv2
import skimage
import skimage.transform
from skimage import exposure
from skimage.util import random_noise
from scipy import ndimage


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
    a = (img - np.min(img))
    b = (np.max(img) - np.min(img))
    return np.divide(a, b, np.zeros_like(a), where=b != 0)


def vertical_flip(img):
    """
    Flip an image along the x-axis.

    Parameters
    ----------
    img : numpy.ndarray
        Image.

    Returns
    -------
    img
        Flipped image.

    """
    return np.flipud(np.copy(img))


def horizontal_flip(img):
    """
    Flips an input image horizontally (mirror)

    Parameters
    ----------
    img : numpy.ndarray
        The image to flip.

    Returns
    -------
    TYPE
        Flipped version of image.

    """
    return np.fliplr(np.copy(img))


def rot90_flip(img, k_times):
    """
    Rotate and image 90 degrees 'k_times'.

    Parameters
    ----------
    img : numpy.ndarray
        Image.

    Returns
    -------
    img
        Flipped image.

    """
    return np.rot90(np.copy(img), k=k_times)


def upside_down_flip(img):
    """
    Flip a 3D array upside down [:, :, 0] --> [:, :, -1].

    Parameters
    ----------
    img : numpy.ndarray
        The image to flip.

    Returns
    -------
    img
        Flipped version of img.

    """
    return np.copy(img)[:, :, ::-1]


def rotate(img, rad):
    """
    Rotates the source image rad degrees.

    Parameters
    ----------
    img : numpy.ndarray
        The image to rotate.
    rad : number
        degrees to rate img.

    Returns
    -------
    img2 : numpy.ndarray
        Rotated version of img.

    """
    img2 = np.copy(img)
    if np.ndim(img2) == 3:
        for z in range(0, img2.shape[-1]):
            img2[:, :, z] = skimage.transform.rotate(img2[:, :, z],
                                                     rad,
                                                     preserve_range=True)
    elif np.ndim(img2) == 2:
        img2 = skimage.transform.rotate(img2, rad, preserve_range=True)
    return img2


def blur_gaussian(img, sigma, eps=1e-3):
    """
    Blur an image using gaussian blurring.

    Parameters
    ----------
    img : numpy.ndarray
        The image to blur. Expected to be of shape ``(H, W)`` or ``(H, W, C)``.
    sigma : number
        Standard deviation of the gaussian blur. Larger numbers result in
        more large-scale blurring.
    eps : number, optional
        A threshold used to decide whether `sigma` can be considered zero.
        The default is 1e-3.

    Returns
    -------
    img : numpy.ndarray
        The blurred image with identical shape and datatype as the input img.

    """
    if sigma < eps:
        return img

    if img.ndim == 2:
        img[:, :] = ndimage.gaussian_filter(img[:, :],
                                            sigma,
                                            mode="mirror")
    else:
        nb_channels = img.shape[2]
        for channel in range(nb_channels):
            img[:, :, channel] = ndimage.gaussian_filter(img[:, :, channel],
                                                         sigma,
                                                         mode="mirror")
    return img


def gamma_correction(img, gamma, gain=1):
    """
    Perform pixel-wise gamma correction on the input image according to the
    Power Law transformation: output = ct_src ** gamma.
    Alters the luminance levels, simulating different lighting conditions.
    
    Parameters
    ----------
    img : numpy.ndarray
        DESCRIPTION.
    gamma : float
        DESCRIPTION.
    gain : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    adjusted_gamma_image : numpy.ndarray
        DESCRIPTION.

    """
    if np.min(img) < 0:
        img = normalize_min_max(img)
    adjusted_gamma_image = exposure.adjust_gamma(img, gamma=gamma, gain=gain)
    return adjusted_gamma_image


def shear_transform(img, shearing_factor):
    """
    Shear angle in counter-clockwise direction as radians.
    Distorts the image, simulating a slanting effect seen in some perspectives.

    Parameters
    ----------
    img : numpy.ndarray
        Image.
    shearing_factor : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    shear_transformer = skimage.transform.AffineTransform(shear=shearing_factor)
    return skimage.transform.warp(img, inverse_map=shear_transformer.inverse, preserve_range=True)


def random_noise_augment(im):
    """


    Parameters
    ----------
    im : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    sigma = random.uniform(0, 0.0001)
    return random_noise(im, var=sigma)


def adaptive_hist_equalization(im):
    """


    Parameters
    ----------
    im : TYPE
        DESCRIPTION.

    Returns
    -------
    im : TYPE
        DESCRIPTION.

    """
    im = exposure.equalize_adapthist(im, clip_limit=0.005, nbins=512)
    return im


def augment_hand(img, gt):
    zero_locs = np.argwhere(img == 0)
    mask = np.copy(img)
    mask[mask != 0] = 1
    random_loc = random.choice(zero_locs)

    x0 = random_loc[0]
    y0 = random_loc[1]

    a = np.random.randint(20, 60)  # half width
    b = np.random.randint(20, 60)  # half height
    x = np.linspace(0, 512, 512)  # x values of interest
    y = np.linspace(0, 512, 512)[:,None]  # y values of interest, as a "column" array

    def create_ellipse(x, y, x0, y0, a, b):
        ellipse = ((x - x0) / a) ** 2 + ((y - y0) / b) ** 2 <= 1  # True for points inside the ellipse
        return np.uint8(ellipse)

    ellipse = create_ellipse(x, y, x0, y0, a, b)
    ellipse = np.float32(ellipse)
    locs_ellipse = np.argwhere(ellipse != 0)

    # TODO: In future tests switch these back
    for loc_c in locs_ellipse:
        # ellipse[loc[0], loc[1]] = 1
        ellipse[loc_c[0], loc_c[1]] = random.uniform(0.6, 0.65)
    # ellipse[ellipse == 1] = random.uniform(0.55, 0.7)
    skin_thickness = 1.2

    skin_art = create_ellipse(x, y, x0, y0, (a - skin_thickness), (b - skin_thickness))
    skin = np.uint8(1 - skin_art)
    # Skin
    ellipse *= skin

    # Fat
    fat_thickness = 1
    fat_art = create_ellipse(x, y, x0, y0, (a - fat_thickness- skin_thickness), (b - fat_thickness - skin_thickness))
    fat_radius = 1.1
    fat = create_ellipse(x, y, x0, y0,
                          np.round((a - fat_thickness - skin_thickness) / fat_radius),
                          np.round((b - fat_thickness - skin_thickness) / fat_radius))
    fat = np.uint8(1 - fat)
    fat_art *= fat

    muscle_thickness = 5
    muscle_art = create_ellipse(x, y, x0, y0, (a - fat_thickness - skin_thickness- muscle_thickness), (b - fat_thickness - skin_thickness - muscle_thickness))
    muscle_radius = 5
    muscle = create_ellipse(x, y, x0, y0,
                          np.round((a - fat_thickness - skin_thickness - muscle_thickness) / muscle_radius),
                          np.round((b - fat_thickness - skin_thickness - muscle_thickness) / muscle_radius))
    muscle = np.uint8(1 - muscle)
    muscle_art *= muscle


    muscle = np.copy(gt)
    fat = np.copy(gt)
    muscle[gt != 1] = 0
    fat[gt != 3] = 0
    muscle[muscle != 0] = 1
    fat[fat != 0] = 1
    muscle = img * np.float32(muscle)
    fat = img * np.float32(fat)

    muscle_flat = muscle[muscle != 0]
    fat_flat = fat[fat != 0]

    fat_art = fat_art.astype(np.float32)
    locs = np.argwhere(fat_art != 0)

    # TODO: In future tests switch these back
    for loc in locs:
        # ellipse[loc[0], loc[1]] = 1
        fat_art[loc[0], loc[1]] = random.choice(fat_flat)

    muscle_art = muscle_art.astype(np.float32)
    locs_musc = np.argwhere(muscle_art != 0)
    for loc in locs_musc:
        # muscle_art[loc[0], loc[1]] = 1

        muscle_art[loc[0], loc[1]] = random.choice(muscle_flat)

    art_arm = ellipse + fat_art + muscle_art
    art_arm = cv2.GaussianBlur(art_arm, (0, 0), 1)
    art_arm *= (1 - mask)
    artificial_img = img + art_arm

    return artificial_img


def invert_colours(ct_src):
    ct = 1 - ct_src
    return ct


def invert_colours_set(ct_src, gt_src):
    # ct = invert_colours(ct_src)
    ct = 1 - ct_src
    return ct, gt_src


def add_hand_set(ct_src, gt_src):
    ct = augment_hand(ct_src, gt_src)
    return ct, gt_src


def adaptive_hist_equalization_set(ct_src, gt_src):
    ct = adaptive_hist_equalization(ct_src)
    return ct, gt_src


def random_noise_set(ct_src, gt_src, gt_bcs_src):
    ct = random_noise_augment(ct_src)
    return ct, gt_src, gt_bcs_src


def shear_transform_set(ct_src, gt_src, gt_bcs_src):
    shearing_factor = random.uniform(-0.1, 0.1)
    ct = shear_transform(ct_src, shearing_factor)
    gt = shear_transform(gt_src, shearing_factor)
    gt_bcs = shear_transform(gt_bcs_src, shearing_factor)
    return ct, gt.astype(np.int16), gt_bcs.astype(np.int16)


def gamma_correction_set(ct_src, gt_src, gt_bcs_src):
    gamma = random.uniform(0.8, 1.2)
    ct = gamma_correction(ct_src, gamma)
    return ct, gt_src, gt_bcs_src


def blur_gaussian_set(ct_src, gt_src, gt_bcs_src):
    sigma = np.random.uniform(0, 1.5)
    ct = blur_gaussian(ct_src, sigma)
    # gt = blur_gaussian(gt_src, sigma)
    return ct, gt_src, gt_bcs_src


def rotate_set(ct_src, gt_src, gt_bcs_src):
    rad = np.random.randint(-30, 30)
    ct = rotate(ct_src, rad)
    gt = rotate(gt_src, rad)
    gt_bcs = rotate(gt_bcs_src, rad)
    return ct, gt.astype(np.int16), gt_bcs.astype(np.int16)


def horizontal_flip_set(ct_src, gt_src, gt_bcs_src):
    ct = horizontal_flip(ct_src)
    gt = horizontal_flip(gt_src)
    gt_bcs = horizontal_flip(gt_bcs_src)
    return ct, gt, gt_bcs


def rot90_flip_set(ct_src, gt_src, gt_bcs_src):
    x = np.random.randint(0, 4)
    ct = rot90_flip(ct_src, k_times=x)
    gt = rot90_flip(gt_src, k_times=x)
    gt_bcs = rot90_flip(gt_bcs_src, k_times=x)
    return ct, gt, gt_bcs


def vertical_flip_set(ct_src, gt_src, gt_bcs_src):
    ct = vertical_flip(ct_src)
    gt = vertical_flip(gt_src)
    gt_bcs = vertical_flip(gt_bcs_src)
    return ct, gt, gt_bcs


def upside_down_flip_set(ct_src, gt_src, gt_bcs_src):
    ct = upside_down_flip(ct_src)
    gt = upside_down_flip(gt_src)
    gt_bcs = upside_down_flip(gt_bcs_src)
    return ct, gt, gt_bcs


def get_augmentations():
    """
    Pool augmentations into a single function

    Returns
    -------
    augmentations : list
        list with all possible augmentation functions.

    """
    augmentations = [horizontal_flip_set,
                     vertical_flip_set,
                     rotate_set,
                     rot90_flip_set,
                     shear_transform_set,
                     ]
    return list(augmentations)


def apply_augmentations(ct_src, gt_src, gt_bcs_src, num_augmentations):
    augmenters = get_augmentations()

    if random.randint(0, 1) == 1:
        augmentations = random.sample(list(augmenters), int(num_augmentations))

        for augmenter in augmentations:
            ct_src, gt_src, gt_bcs_src = augmenter(ct_src,
                                                   gt_src,
                                                   gt_bcs_src)
            ct_src = normalize_min_max(ct_src)

        return ct_src, gt_src, gt_bcs_src
    else:
        return ct_src, gt_src, gt_bcs_src
