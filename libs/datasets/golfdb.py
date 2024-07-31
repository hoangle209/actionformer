import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import os
import json
import pickle

from .datasets import register_dataset
from .data_utils import truncate_feats

@register_dataset("golfdb")
class GolfDB(Dataset):
    event_names = {
        0: 'Address',
        1: 'Toe-up',
        2: 'Mid-backswing (arm parallel)',
        3: 'Top',
        4: 'Mid-downswing (arm parallel)',
        5: 'Impact',
        6: 'Mid-follow-through (shaft parallel)',
        7: 'Finish'
    }

    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        json_file,        # json or pkl file for annotations 
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        force_upsampling  # force to upsample to max_seq_len
    ):
        # assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert crop_ratio == None or len(crop_ratio) == 2

        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes # num classes always is 8
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db = self._load_annotation_file(self.json_file)
        self.data_list = dict_db
        self.label_dict = GolfDB.event_names


    def _load_annotation_file(self, annotation_file_path):
        postfix = annotation_file_path.split('.')[-1]
        with open(annotation_file_path, 'r') as f:
            if postfix == 'json':
                annotation_data = json.load(f)
            elif postfix == 'pkl':
                annotation_data = pickle.load(f)

        dict_db = tuple()
        for key, value in annotation_data.items():
            # feat_file = os.path.join(self.feat_folder,
            #                          self.file_prefix + key + self.file_ext)
            # if not os.path.exists(feat_file):
            #     continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."
        
            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            segments, labels = [], []
            events = value["events"]
            for event_id, (ts, te) in enumerate(zip(events, events[1:])):
                ts = ts - events[0]
                te = te - events[0]
                segments.append((ts, te))
                labels.append(event_id)

            dict_db += ({
                "id": key,
                "fps": fps,
                "duration": duration,
                "segments": segments,
                "labels": labels
            },)
        
        return dict_db
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        video_item = self.data_list[index]

        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset # each feat stride frames will have same feature 
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None
        
        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict


            
    