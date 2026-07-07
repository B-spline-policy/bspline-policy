from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
from diffusion_policy.dataset.preprocessed_sample_cache import (
    build_preprocessed_sample_cache,
    get_preprocessed_cached_item,
    make_empty_preprocessed_sample_cache,
    normalize_rgb_cache_dtype,
)
register_codecs()


def _cache_base_path(dataset_path: str, cache_suffix: str = None) -> str:
    if not cache_suffix:
        return dataset_path
    if not cache_suffix.startswith("."):
        cache_suffix = "." + cache_suffix
    return dataset_path + cache_suffix


def _replay_cache_path(dataset_path: str, cache_suffix: str = None) -> str:
    return _cache_base_path(dataset_path, cache_suffix) + ".zarr.zip"


class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            cache_decoded_replay=False,
            cache_preprocessed_samples=False,
            cache_preprocessed_device="cpu",
            cache_preprocessed_share_memory=False,
            cache_preprocessed_rgb_dtype="float32",
            cache_suffix=None,
            seed=42,
            val_ratio=0.0
        ):
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)
        cache_preprocessed_rgb_dtype = normalize_rgb_cache_dtype(
            cache_preprocessed_rgb_dtype)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = _replay_cache_path(dataset_path, cache_suffix)
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                        if cache_decoded_replay:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=replay_buffer.root.store,
                                store=None)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        if cache_decoded_replay:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=zip_store, store=None)
                        else:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer)
            if cache_decoded_replay:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=replay_buffer.root.store,
                    store=None)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self.cache_decoded_replay = cache_decoded_replay
        self.cache_preprocessed_samples = cache_preprocessed_samples
        self.cache_preprocessed_device = cache_preprocessed_device
        self.cache_preprocessed_share_memory = cache_preprocessed_share_memory
        self.cache_preprocessed_rgb_dtype = cache_preprocessed_rgb_dtype
        self.cache_suffix = cache_suffix
        self._preprocessed_cache = None
        self._length = len(sampler)

        if cache_preprocessed_samples:
            self._build_preprocessed_cache(
                device=cache_preprocessed_device,
                share_memory=cache_preprocessed_share_memory)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        val_set._length = len(val_set.sampler)
        if self._preprocessed_cache is not None:
            val_set._preprocessed_cache = None
            val_set._build_preprocessed_cache(
                device=self.cache_preprocessed_device,
                share_memory=self.cache_preprocessed_share_memory)
        return val_set

    def __copy__(self):
        result = self.__class__.__new__(self.__class__)
        result.__dict__.update(self.__dict__)
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("_preprocessed_cache") is not None:
            state["replay_buffer"] = None
            state["sampler"] = None
        return state

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            # if key.endswith('pos'):
            if 'pos' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            # elif key.endswith('quat'):
            elif 'quat' in key:
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            # elif key.endswith('qpos'):
            elif 'qpos' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == 'base_pose':
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return self._length

    def _sample_uncached_item(
            self,
            idx: int,
            rgb_dtype: str = "float32") -> Dict[str, torch.Tensor]:
        if self.sampler is None:
            raise RuntimeError(
                "RobomimicReplayImageDataset was serialized without replay/sampler "
                "because cache_preprocessed_samples is enabled, but no "
                "preprocessed cache is available for this item."
            )
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            image = np.moveaxis(data[key][T_slice], -1, 1)
            if rgb_dtype == "uint8":
                obs_dict[key] = np.ascontiguousarray(image, dtype=np.uint8)
            else:
                obs_dict[key] = np.ascontiguousarray(image, dtype=np.float32)
                obs_dict[key] *= 1.0 / 255.0
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data

    def _build_preprocessed_cache(self, device: str = "cpu", share_memory: bool = False):
        print(
            f"Building preprocessed sample cache: samples={len(self)}, "
            f"device={device}, share_memory={share_memory}, "
            f"rgb_dtype={self.cache_preprocessed_rgb_dtype}"
        )
        obs_steps = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        empty_cache = make_empty_preprocessed_sample_cache(
            shape_meta=self.shape_meta,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.lowdim_keys,
            action_shape=(self.horizon,) + tuple(self.shape_meta["action"]["shape"]),
            obs_steps=obs_steps,
            device=torch.device(device),
            share_memory=share_memory,
            rgb_dtype=self.cache_preprocessed_rgb_dtype,
        )
        self._preprocessed_cache = build_preprocessed_sample_cache(
            length=len(self),
            sample_fn=lambda idx: self._sample_uncached_item(
                idx,
                rgb_dtype=self.cache_preprocessed_rgb_dtype),
            device=device,
            share_memory=share_memory,
            empty_cache=empty_cache,
            desc="Precomputing samples",
        )
        print("Preprocessed sample cache ready.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._preprocessed_cache is not None:
            return get_preprocessed_cached_item(
                self._preprocessed_cache,
                idx,
                rgb_keys=self.rgb_keys,
            )
        return self._sample_uncached_item(idx)

# mujoco
# def _convert_actions(raw_actions, abs_action, rotation_transformer):
#     actions = raw_actions
#     if abs_action:
#         pos = raw_actions[...,:6]
#         rot = raw_actions[...,6:9]
#         gripper = raw_actions[...,9:]
#         rot = rotation_transformer.forward(rot)
#         raw_actions = np.concatenate([
#             pos, rot, gripper
#         ], axis=-1).astype(np.float32)
#         actions = raw_actions
#     return actions

# real
def _convert_actions(raw_actions, abs_action, rotation_transformer, target_action_dim=None):
    raw_actions = raw_actions.astype(np.float32)
    if not abs_action:
        return raw_actions

    raw_dim = raw_actions.shape[-1]
    if target_action_dim is None:
        target_action_dim = 23 if raw_dim == 17 else raw_dim

    if raw_dim == target_action_dim:
        return raw_actions

    def convert_arm(pos, rot, gripper):
        rot = rotation_transformer.forward(rot)
        return np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

    if raw_dim == 7:
        converted_by_dim = {
            10: convert_arm(
                raw_actions[..., :3],
                raw_actions[..., 3:6],
                raw_actions[..., 6:7],
            ),
        }
    elif raw_dim == 14:
        left_pos = raw_actions[..., :3]
        left_rot = raw_actions[..., 3:6]
        right_pos = raw_actions[..., 6:9]
        right_rot = raw_actions[..., 9:12]
        left_gripper = raw_actions[..., 12:13]
        right_gripper = raw_actions[..., 13:14]
        left = convert_arm(
            left_pos,
            left_rot,
            left_gripper,
        )
        right = convert_arm(
            right_pos,
            right_rot,
            right_gripper,
        )
        dual_arm_ee = np.concatenate(
            [
                left_pos,
                rotation_transformer.forward(left_rot),
                right_pos,
                rotation_transformer.forward(right_rot),
                left_gripper,
                right_gripper,
            ],
            axis=-1,
        ).astype(np.float32)
        converted_by_dim = {
            10: left,
            20: dual_arm_ee,
        }
    elif raw_dim == 17:
        base_vel = raw_actions[..., :3]
        left = convert_arm(
            raw_actions[..., 3:6],
            raw_actions[..., 6:9],
            raw_actions[..., 9:10],
        )
        right = convert_arm(
            raw_actions[..., 10:13],
            raw_actions[..., 13:16],
            raw_actions[..., 16:17],
        )
        converted_by_dim = {
            10: left,
            20: np.concatenate([left, right], axis=-1).astype(np.float32),
            23: np.concatenate([base_vel, left, right], axis=-1).astype(np.float32),
        }
    else:
        converted_by_dim = {}

    if target_action_dim not in converted_by_dim:
        raise ValueError(
            f"Unsupported action conversion from raw dim {raw_dim} "
            f"to target dim {target_action_dim}."
        )
    return converted_by_dim[target_action_dim]

# real (bimanual): base_vel(3) + left[pos(3) rot(3) grip(1)] + right[pos(3) rot(3) grip(1)] = 17 -> 23
# def _convert_actions(raw_actions, abs_action, rotation_transformer):
#     actions = raw_actions
#     if abs_action:
#         base_vel = raw_actions[...,:3]
#
#         left_pos = raw_actions[...,3:6]
#         left_rot = raw_actions[...,6:9]
#         left_gripper = raw_actions[...,9:10]
#         left_rot = rotation_transformer.forward(left_rot)
#
#         right_pos = raw_actions[...,10:13]
#         right_rot = raw_actions[...,13:16]
#         right_gripper = raw_actions[...,16:17]
#         right_rot = rotation_transformer.forward(right_rot)
#
#
#         raw_actions = np.concatenate([
#             base_vel, left_pos, left_rot, left_gripper, right_pos, right_rot, right_gripper
#         ], axis=-1).astype(np.float32)
#         actions = raw_actions
#     return actions

def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    target_action_dim=shape_meta['action']['shape'][0],
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(demos)):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
