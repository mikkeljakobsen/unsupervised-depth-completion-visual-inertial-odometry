'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Xiaohan Fei <feixh@cs.ucla.edu>
If you use this code, please cite the following paper:
A. Wong, X. Fei, S. Tsuei, and S. Soatto. Unsupervised Depth Completion from Visual Inertial Odometry.
https://arxiv.org/pdf/1905.08616.pdf
@article{wong2020unsupervised,
  title={Unsupervised Depth Completion From Visual Inertial Odometry},
  author={Wong, Alex and Fei, Xiaohan and Tsuei, Stephanie and Soatto, Stefano},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={1899--1906},
  year={2020},
  publisher={IEEE}
}
'''
import os, sys, glob, argparse
import multiprocessing as mp
import numpy as np
import cv2
sys.path.insert(0, 'src')
import data_utils
#VOID_ROOT_DIRPATH       = os.path.join('data', 'void_release')
VOID_ROOT_DIRPATH       = os.path.join('/home', 'mikkel', 'data', 'void_release')
VOID_DATA_150_DIRPATH   = os.path.join(VOID_ROOT_DIRPATH, 'void_150')

VOID_OUT_DIRPATH        = os.path.join('data', 'void_sparse')
VOID_TEST_SPARSE_DEPTH_FILENAME   = 'test_sparse_depth.txt'
VOID_TEST_VALIDITY_MAP_FILENAME   = 'test_validity_map.txt'

TEST_REFS_DIRPATH       = 'testing'

# VOID testing set 150 density
VOID_TEST_SPARSE_DEPTH_150_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'void_test_sparse_depth_150.txt')
VOID_TEST_INTERP_DEPTH_150_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'void_test_interp_depth_150.txt')
VOID_TEST_VALIDITY_MAP_150_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'void_test_validity_map_150.txt')


def process_frame(args):
  sparse_depth_path, validity_map_path = args

  # Create interpolated depth
  sz, vm = data_utils.load_depth_with_validity_map(sparse_depth_path)
  iz = data_utils.interpolate_depth(sz, vm)

  sparse_depth_ref_path = os.path.join(*sparse_depth_path.split(os.sep)[2:])
  # Set output paths
  sparse_depth_output_path = sparse_depth_path
  interp_depth_output_path = os.path.join(VOID_OUT_DIRPATH, sparse_depth_ref_path) \
      .replace('sparse_depth', 'interp_depth')
  validity_map_output_path = validity_map_path

  sparse_depth_filename = os.path.basename(sparse_depth_output_path)
  validity_map_filename = os.path.basename(validity_map_output_path)

  # Write to disk
  data_utils.save_depth(iz, interp_depth_output_path)

  return (sparse_depth_output_path, interp_depth_output_path, validity_map_output_path)


parser = argparse.ArgumentParser()

parser.add_argument('--n_thread',  type=int, default=8)
args = parser.parse_args()

for ref_dirpath in [TEST_REFS_DIRPATH]:
  if not os.path.exists(ref_dirpath):
    os.makedirs(ref_dirpath)

data_dirpaths = [VOID_DATA_150_DIRPATH]

test_output_filepaths = [
    [VOID_TEST_SPARSE_DEPTH_150_FILEPATH, VOID_TEST_INTERP_DEPTH_150_FILEPATH, VOID_TEST_VALIDITY_MAP_150_FILEPATH]]

data_filepaths = zip(data_dirpaths, test_output_filepaths)
for data_dirpath, test_filepaths in data_filepaths:
  # Testing set
  test_sparse_depth_filepath = os.path.join(data_dirpath, VOID_TEST_SPARSE_DEPTH_FILENAME)
  test_validity_map_filepath = os.path.join(data_dirpath, VOID_TEST_VALIDITY_MAP_FILENAME)
  # Read testing paths
  test_sparse_depth_paths = data_utils.read_paths(test_sparse_depth_filepath)
  test_validity_map_paths = data_utils.read_paths(test_validity_map_filepath)

  # Get test set directories
  test_seq_dirpaths = set(
      [test_image_paths[idx].split(os.sep)[-3] for idx in range(len(test_image_paths))])

  # Initialize placeholders for training output paths
  train_sparse_depth_output_paths = []
  train_interp_depth_output_paths = []
  train_validity_map_output_paths = []
  # Initialize placeholders for testing output paths
  test_sparse_depth_output_paths = []
  test_interp_depth_output_paths = []
  test_validity_map_output_paths = []

  # For each dataset density, grab the sequences
  seq_dirpaths = glob.glob(os.path.join(data_dirpath, 'data', '*'))
  n_sample = 0
  for seq_dirpath in seq_dirpaths:
    # For each sequence, grab the images, sparse depths and valid maps
    sparse_depth_paths = sorted(glob.glob(os.path.join(seq_dirpath, 'sparse_depth', '*.png')))
    validity_map_paths = sorted(glob.glob(os.path.join(seq_dirpath, 'validity_map', '*.png')))

    # Load intrinsics and process first
    interp_depth_output_dirpath = os.path.join(os.path.dirname(intrinsics_output_path), 'interp_depth')

    for output_dirpath in [interp_depth_output_dirpath]:
      if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    pool_inputs = []
    for idx in range(len(image_paths)):
      # Find images with enough parallax, pose are from camera to world
      pool_inputs.append(
          (sparse_depth_paths[idx],
           validity_map_paths[idx]))

    sys.stdout.write('Processing {} examples for sequence={}\r'.format(
        len(pool_inputs), seq_dirpath))
    sys.stdout.flush()

    with mp.Pool(args.n_thread) as pool:
      pool_results = pool.map(process_frame, pool_inputs)

      for result in pool_results:
        sparse_depth_output_path, interp_depth_output_path, \
            validity_map_output_path = result

        # Split into training, testing and unused testing sets
        if image_ref_path in test_image_paths:
          test_sparse_depth_output_paths.append(sparse_depth_output_path)
          test_interp_depth_output_paths.append(interp_depth_output_path)
          test_validity_map_output_paths.append(validity_map_output_path)

    print('Completed processing {} examples for sequence={}'.format(
        len(pool_inputs), seq_dirpath))

  print('Completed processing {} examples for density={}'.format(n_sample, data_dirpath))

  void_test_sparse_depth_filepath, void_test_interp_depth_filepath, void_test_validity_map_filepath = test_filepaths

  print('Storing testing sparse depth file paths into: %s' % void_test_sparse_depth_filepath)
  with open(void_test_sparse_depth_filepath, "w") as o:
    for idx in range(len(test_sparse_depth_output_paths)):
      o.write(test_sparse_depth_output_paths[idx]+'\n')
  print('Storing testing interpolated depth file paths into: %s' % void_test_interp_depth_filepath)
  with open(void_test_interp_depth_filepath, "w") as o:
    for idx in range(len(test_interp_depth_output_paths)):
      o.write(test_interp_depth_output_paths[idx]+'\n')
  print('Storing testing validity map file paths into: %s' % void_test_validity_map_filepath)
  with open(void_test_validity_map_filepath, "w") as o:
    for idx in range(len(test_validity_map_output_paths)):
      o.write(test_validity_map_output_paths[idx]+'\n')
