import os
import os.path as osp
from glob import glob
import numpy as np
import pickle
import torch
from util.preprocessing import load_img, augmentation_keep_size
from util.formatting import DefaultFormatBundle
from detrsmpl.data.datasets.pipelines.transforms import Normalize
from detrsmpl.utils.ffmpeg_utils import video_to_images, vid_info_reader
from mmcv.runner import get_dist_info
import torch.distributed as dist
from tqdm import tqdm

from fractions import Fraction

def _parse_fps(value: str, default: float = 30.0) -> float:
    if value and value not in ("0/0", "0", "N/A"):
        try:
            return float(Fraction(value))
        except (ValueError, ZeroDivisionError):
            pass
    return float(default)


class INFERENCE(torch.utils.data.Dataset):
    def __init__(self, img_dir=None, out_path=None, id_file=None, skip_file=None):

        self.output_path = out_path
        self.img_dir = img_dir
        self.id_file = id_file
        self.skip_file = skip_file
        self.skip_existing = True  # Enable skip logic for existing files

        self.is_vid = False

        rank, _ = get_dist_info()
        self.img_name = osp.basename(self.img_dir.rstrip('/')) if isinstance(self.img_dir, str) else 'inference'
        if self.img_dir.endswith('.mp4'):
            self.is_vid = True
            self.img_name = self.img_name[:-4]

        self.img_name = self.img_name + '_out'

        if self.is_vid:
            info = vid_info_reader(self.img_dir)
            fps_str = info.video_stream.get('avg_frame_rate') or info.video_stream.get('r_frame_rate')
            self.source_fps = _parse_fps(fps_str)
        else:
            self.source_fps = 30.0

        self.output_path = os.path.join(self.output_path, self.img_name)
        os.makedirs(self.output_path, exist_ok=True)

        # Directory dedicated to SMPL-X parameter dumps
        self.smplx_output_root = os.path.join(self.output_path, 'smplx_params')
        os.makedirs(self.smplx_output_root, exist_ok=True)

        # Load filter IDs if id_file is provided
        self.filter_ids = None
        if self.id_file and osp.exists(self.id_file):
            with open(self.id_file, 'r') as f:
                # Read IDs from file, strip whitespace and ignore empty lines
                # Support both newline-separated and space-separated IDs
                all_ids = []
                for line in f:
                    line = line.strip()
                    if line:
                        # Split by whitespace to handle space-separated IDs
                        all_ids.extend(line.split())
                self.filter_ids = set(all_ids)
            if rank == 0:
                print(f'[INFERENCE] Loaded {len(self.filter_ids)} IDs from {self.id_file}')

        # Load skip IDs if skip_file is provided
        self.skip_ids = None
        if self.skip_file and osp.exists(self.skip_file):
            with open(self.skip_file, 'r') as f:
                skip_list = []
                for line in f:
                    line = line.strip()
                    if line:
                        skip_list.extend(line.split())
                self.skip_ids = set(skip_list)
            if rank == 0:
                print(f'[INFERENCE] Loaded {len(self.skip_ids)} skip IDs from {self.skip_file}')

        self.samples = []  # keep metadata for each frame
        self.tmp_dir = None

        if self.is_vid:
            self.tmp_dir = os.path.join(self.output_path, 'temp_img')
            os.makedirs(self.tmp_dir, exist_ok=True)
            if rank == 0:
                video_to_images(self.img_dir, self.tmp_dir)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            frame_paths = sorted(glob(osp.join(self.tmp_dir, '*')))
            for frame_path in frame_paths:
                frame_name = osp.splitext(osp.basename(frame_path))[0]
                # Filter by ID if filter_ids is provided; skip if in skip_ids
                if self.skip_ids and frame_name in self.skip_ids:
                    continue
                if self.filter_ids is None or frame_name in self.filter_ids:
                    self.samples.append({
                        'image_path': frame_path,
                        'relative_dir': '',
                        'frame_name': frame_name
                    })
        else:
            self._gather_image_samples()

        self.score_threshold = 0.2
        self.resolution = [720 ,1280] # AGORA test
        self.format = DefaultFormatBundle()
        self.normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

        if rank == 0:
            print(f'[INFERENCE] collected {len(self.samples)} frames from {self.img_dir}')
            if self.filter_ids:
                print(f'[INFERENCE] filtered to {len(self.samples)} frames based on id_file')
            if self.skip_ids:
                print(f'[INFERENCE] skipped IDs from skip_file, {len(self.samples)} frames remaining')

    def _gather_image_samples(self):
        rank, _ = get_dist_info()

        # If we have filter_ids, directly build paths from IDs (much faster)
        if self.filter_ids:
            if rank == 0:
                print(f'[INFERENCE] Building paths for {len(self.filter_ids)} IDs...')

            image_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
            image_patterns = ['*.jpg', '*.png', '*.jpeg']
            ids_to_process = self.filter_ids - self.skip_ids if self.skip_ids else self.filter_ids
            iterator = tqdm(ids_to_process, desc='Building image paths', disable=(rank != 0))

            for img_id in iterator:
                # First, check if img_id is a directory containing images
                dir_path = osp.join(self.img_dir, img_id)
                if osp.isdir(dir_path):
                    # It's a directory, gather all images from it
                    for pattern in image_patterns:
                        images_in_dir = glob(osp.join(dir_path, pattern))
                        for img_path in images_in_dir:
                            frame_name = osp.splitext(osp.basename(img_path))[0]
                            self.samples.append({
                                'image_path': img_path,
                                'relative_dir': img_id,
                                'frame_name': frame_name
                            })
                else:
                    # Try to find it as a file with different extensions
                    found = False
                    for ext in image_extensions:
                        img_path = osp.join(self.img_dir, img_id + ext)
                        if osp.exists(img_path):
                            # Parse relative directory from the ID
                            rel_dir = osp.dirname(img_id) if '/' in img_id or '\\' in img_id else ''
                            frame_name = osp.basename(img_id)

                            self.samples.append({
                                'image_path': img_path,
                                'relative_dir': rel_dir,
                                'frame_name': frame_name
                            })
                            found = True
                            break

                    if not found and rank == 0:
                        # Only warn occasionally to avoid flooding logs
                        if len(self.samples) % 10000 == 0:
                            print(f'[WARNING] Could not find image or directory for ID: {img_id}')

        else:
            # Original behavior: scan all images in directory
            image_patterns = ['*.jpg', '*.png', '*.jpeg']
            all_images = []

            if rank == 0:
                print('[INFERENCE] Searching for images in directory...')

            for pattern in image_patterns:
                all_images.extend(glob(osp.join(self.img_dir, '**', pattern), recursive=True))

            all_images = sorted(set(all_images))

            if not all_images:
                raise FileNotFoundError(f'No image files found under {self.img_dir}')

            if rank == 0:
                print(f'[INFERENCE] Found {len(all_images)} total images')

            # Use tqdm only on rank 0 to avoid duplicate progress bars
            iterator = tqdm(all_images, desc='Processing images', disable=(rank != 0))

            for img_path in iterator:
                rel_path = osp.relpath(img_path, self.img_dir)
                rel_dir = osp.dirname(rel_path)
                if rel_dir == '.':
                    rel_dir = ''
                frame_name = osp.splitext(osp.basename(img_path))[0]

                # Skip if the relative dir or frame name is in skip_ids
                if self.skip_ids and (rel_dir in self.skip_ids or frame_name in self.skip_ids):
                    continue

                self.samples.append({
                    'image_path': img_path,
                    'relative_dir': rel_dir,
                    'frame_name': frame_name
                })

    def _resolve_save_directory(self, sample):
        if sample['relative_dir']:
            save_dir = osp.join(self.smplx_output_root, sample['relative_dir'])
        else:
            save_dir = self.smplx_output_root
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

       
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = load_img(sample['image_path'],'BGR')
        img_whole_bbox = np.array([0, 0, img.shape[1],img.shape[0]])
        img, img2bb_trans, bb2img_trans, _, _ = \
            augmentation_keep_size(img, img_whole_bbox, 'test')

        cropped_img_shape=img.shape[:2]
        img = (img.astype(np.float32)) 
        
        inputs = {'img': img}
        targets = {
            'body_bbox_center': np.array(img_whole_bbox[None]),
            'body_bbox_size': np.array(img_whole_bbox[None])}
        meta_info = {
            'ori_shape':np.array(self.resolution),
            'img_shape': np.array(img.shape[:2]),
            'img2bb_trans': img2bb_trans,
            'bb2img_trans': bb2img_trans,
            'ann_idx': idx}
        result = {**inputs, **targets, **meta_info}
        
        result = self.normalize(result)
        result = self.format(result)
            
        return result
        
    def inference(self, outs):
        output = {}

        for out in outs:
            ann_idx = out['image_idx']
            sample = self.samples[ann_idx]

            # Skip if this frame already has output files
            save_dir = self._resolve_save_directory(sample)
            if self.skip_existing:
                existing_files = glob(osp.join(save_dir, f"{sample['frame_name']}_person_*.pkl"))
                if existing_files:
                    # Frame already processed, skip it
                    continue

            scores = out['scores'].clone().cpu().numpy()
            img_shape = out['img_shape'].cpu().numpy()[::-1]  # width, height
            width, height = img_shape
            width += width % 2
            height += height % 2
            img_shape = np.array([width, height])

            joint_proj = out['smplx_joint_proj'].clone().cpu().numpy()

            if self.resolution == [720, 1280]:
                joint_proj[:, :, 0] = joint_proj[:, :, 0] / img_shape[0] * 3840
                joint_proj[:, :, 1] = joint_proj[:, :, 1] / img_shape[1] * 2160
            elif self.resolution == [1200, 1600]:
                joint_proj[:, :, 0] = joint_proj[:, :, 0] * (1200 / 800)
                joint_proj[:, :, 1] = joint_proj[:, :, 1] * (1600 / 1066)

            rel_output_key = osp.join(sample['relative_dir'], sample['frame_name']) if sample['relative_dir'] else sample['frame_name']
            per_frame_results = []

            for i, score in enumerate(scores):
                if score < self.score_threshold:
                    break

                save_dict = {
                        'cam_trans': out['cam_trans'][i].reshape(1, -1).cpu().numpy(),
                        'smplx_root_pose': out['smplx_root_pose'][i].reshape(1, -1).cpu().numpy(),
                        'smplx_body_pose': out['smplx_body_pose'][i].reshape(1, -1).cpu().numpy(),
                        'smplx_lhand_pose': out['smplx_lhand_pose'][i].reshape(1, -1).cpu().numpy(),
                        'smplx_rhand_pose': out['smplx_rhand_pose'][i].reshape(1, -1).cpu().numpy(),
                        # 'reye_pose': np.zeros((1, 3)),
                        # 'leye_pose': np.zeros((1, 3)),
                        'smplx_jaw_pose': out['smplx_jaw_pose'][i].reshape(1, -1).cpu().numpy(),
                        'smplx_expr': out['smplx_expr'][i].reshape(1, -1).cpu().numpy(),
                        'smplx_shape': out['smplx_shape'][i].reshape(1, -1).cpu().numpy()
                }




                save_path = osp.join(save_dir, f"{sample['frame_name']}_person_{i}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(save_dict, f)

                per_frame_results.append({'score': float(score), 'path': save_path})

            if per_frame_results:
                output[rel_output_key] = per_frame_results

        return output
