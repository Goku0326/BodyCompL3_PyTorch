import os
import h5py
import pydicom
import numpy as np
import argparse

from barbell2.converters.dcm2npy import Dicom2Numpy
from barbell2.converters.tag2npy import Tag2Numpy

def get_dicom_pixels(f, normalize):
    dcm2numpy = Dicom2Numpy(f)
    dcm2numpy.set_normalize_enabled(normalize)
    npy_array = dcm2numpy.execute()
    return npy_array

def get_tag_pixels(f, shape):
    base_path = f[:-4]  
    tag_file_path = base_path + '.tag'
    if not os.path.exists(tag_file_path):
        tag_file_path = base_path + '.dcm.tag'  
        if not os.path.exists(tag_file_path):
            return None, 'TAG file missing'
    tag2numpy = Tag2Numpy(tag_file_path, shape)
    try:
        npy_array = tag2numpy.execute()
        return npy_array, None
    except Exception as e:
        return None, str(e)

def has_dimension(f, rows, columns):
    try:
        p = pydicom.dcmread(f)
        return p.Rows == rows and p.Columns == columns, None
    except Exception as e:
        return False, str(e)

def update_labels(pixels):
    labels_to_keep = [0, 1, 5, 7]
    labels_to_remove = [2, 12, 14]
    for label in np.unique(pixels):
        if label in labels_to_remove:
            pixels[pixels == label] = 0
    for label in np.unique(pixels):
        if label not in labels_to_keep:
            return None, 'Incorrect labels'
    if len(np.unique(pixels)) != 4:
        return None, 'Incorrect number of labels'
    return pixels, None

def create_h5(target_file, root_dir, rows=512, columns=512, normalize=True):
    skipped_files = []
    added_files = 0
    with h5py.File(target_file, 'w') as h5f:
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.dcm'):
                    f_full = os.path.join(root, f)
                    dim_ok, err = has_dimension(f_full, rows, columns)
                    if dim_ok:
                        dicom_pixels = get_dicom_pixels(f_full, normalize)
                        tag_pixels, err = get_tag_pixels(f_full, dicom_pixels.shape)
                        if tag_pixels is not None:
                            tag_pixels, err = update_labels(tag_pixels)
                            if tag_pixels is not None:
                                patient_group = h5f.create_group(f.replace('.dcm', ''))
                                patient_group.create_dataset('images', data=dicom_pixels)
                                patient_group.create_dataset('labels', data=tag_pixels)
                                added_files += 1
                            else:
                                skipped_files.append((f, 'Label update error: ' + err))
                        else:
                            skipped_files.append((f, 'TAG error: ' + err))
                    else:
                        skipped_files.append((f, 'Dimension error: ' + err))

    print(f'Total files processed: {added_files + len(skipped_files)}')
    print(f'Files added to HDF5: {added_files}')
    print(f'Files skipped: {len(skipped_files)}')
    for file, reason in skipped_files:
        print(f'{file} - Reason: {reason}')

def main():
    parser = argparse.ArgumentParser(description='Convert DICOM and TAG files to HDF5 format.')
    parser.add_argument('--dicom_dir', type=str, required=True, help='Directory containing DICOM files')
    parser.add_argument('--output_file', type=str, required=True, help='Output HDF5 file path')
    args = parser.parse_args()
    create_h5(args.output_file, args.dicom_dir, 512, 512, True)

if __name__ == '__main__':
    main()
