# Created by Xu Yan at 2021/10/17

import copy
import random

from torch.utils.data import Dataset
from datasets.data_classes import PointCloud, Box
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import os
import warnings
import pickle
from functools import reduce
from tqdm import tqdm
from datasets.generate_waymo_sot import generate_waymo_data
from collections import defaultdict
from datasets import points_utils, base_dataset
import json

class TrackletAnnotation: 
    def __init__(self, anno_str, anno_length):
        self.anno_str = anno_str
        self.anno_length = anno_length

    def __len__(self):
        return self.anno_length

class WaymoEvalDataset(base_dataset.BaseDataset):
    def __init__(self, path, split, category_name="VEHICLE", **kwargs):
        super().__init__(path, split, category_name, **kwargs)
        self.Waymo_Folder = path
        self.category_name = category_name
        bench_dir = os.path.join(self.Waymo_Folder, 'benchmark',self.category_name.lower())
        self.bench = json.load(open(os.path.join(bench_dir, 'bench_list.json'))) #list，里面是1121个dict

        def extract_ids_from_bench(bench_name):
            b = json.load(open(os.path.join(bench_dir, bench_name)))
            ids = set()
            for tracklet_info in b:
                ids.add(tracklet_info['id'])
            return ids
        
        self.easy_ids = extract_ids_from_bench('easy.json') #set格式，存储字符串格式的id，这个id对应了1121个序列里的标识符
        self.medium_ids = extract_ids_from_bench('medium.json') #set格式，存储字符串格式的id，这个id对应了1121个序列里的标识符
        self.hard_ids = extract_ids_from_bench('hard.json') #set格式，存储字符串格式的id，这个id对应了1121个序列里的标识符

        self.tracklet_anno_list, self.tracklet_len_list = self._build_tracklet_anno() #此时self.tracklet_anno_list只存储tracklet的id

        self.pcds = None
        self.gt_infos = None
        self.cache_tracklet_id = None
        self.mode = None


    def get_num_scenes(self):
        return len(self.tracklet_anno_list)

    def get_num_tracklets(self):
        return len(self.tracklet_anno_list)

    def get_num_frames_total(self):
        return sum(self.tracklet_len_list)

    def get_num_frames_tracklet(self, tracklet_id):
        return self.tracklet_len_list[tracklet_id]

    def _build_tracklet_anno(self):
        list_of_tracklet_anno = []
        list_of_tracklet_len = []

        for tracklet in self.bench: #bench是list
            tracklet_id = tracklet["id"] #取出字符串id
            frame_range = tracklet['frame_range']
            tracklet_len = frame_range[1]-frame_range[0]+1
            anno = TrackletAnnotation(tracklet_id,tracklet_len)
            list_of_tracklet_anno.append(anno) #此时 list_of_tracklet_anno 仅仅存储每个tracklet的字符串id
            list_of_tracklet_len.append(tracklet_len)

        return list_of_tracklet_anno, list_of_tracklet_len
    
    def get_frame(self, tracklet_id, frame_id):
        tracklet_info = self.bench[tracklet_id]
        t_id = tracklet_info['id']
        if t_id in self.easy_ids:
            self.mode = 'easy'
        elif t_id in self.medium_ids:
            self.mode = 'medium'
        elif t_id in self.hard_ids:
            self.mode = 'hard'
        segment_name = tracklet_info['segment_name']
        frame_range = tracklet_info['frame_range']
        if tracklet_id != self.cache_tracklet_id: #每次取数据的时候会把整个tracklet缓存，这样在取每一帧的时候就不用每次都读取整个tracklet，很巧妙
            self.cache_tracklet_id = tracklet_id
            if self.gt_infos:
                del self.gt_infos
            if self.pcds:
                del self.pcds
            self.gt_infos = np.load(os.path.join(self.Waymo_Folder, 'gt_info', '{:}.npz'.format(
                segment_name)), allow_pickle=True)
            self.pcds = np.load(os.path.join(self.Waymo_Folder, 'pc', 'raw_pc', '{:}.npz'.format(
                segment_name)), allow_pickle=True)
            self.egos = np.load(os.path.join(self.Waymo_Folder, 'ego_info', '{:}.npz'.format(
                segment_name)), allow_pickle=True)
        return self._build_frame(frame_range, t_id, frame_id) #根据字符串id和帧的id获取注释，帧id是从0开始，需要在frame_range范围之内取值

    def get_frames(self, seq_id, frame_ids): #根据tracklet的字符串id以及帧的ids，获取一串anno

        frames = []
        for i in frame_ids:
            frames.append(self.get_frame(seq_id, i)) #根据序列的id和这个序列之内的每一帧的id，就能获取到一个具体的注释

        return frames

    def _build_frame(self, frame_range, t_id, frame_id):
        idx = frame_range[0]+frame_id

        pointcloud = self.pcds[str(idx)].transpose((1, 0))

        ref_pose = np.reshape(self.egos[str(idx)], [4, 4])
        global_from_car, _ = self.veh_pos_to_transform(ref_pose)
        nbr_points = pointcloud.shape[1]
        pointcloud[:3, :] = global_from_car.dot(
            np.vstack((pointcloud[:3, :], np.ones(nbr_points)))
        )[:3, :]
        pc = PointCloud(pointcloud)


        frame_bboxes = self.gt_infos['bboxes'][idx]
        frame_ids = self.gt_infos['ids'][idx]
        index = frame_ids.index(t_id)
        bbox = frame_bboxes[index]

        center = [bbox[0], bbox[1], bbox[2]]
        size = [bbox[5], bbox[4], bbox[6]]
        orientation = Quaternion(axis=[0, 0, 1], angle=-bbox[3])

        bbox = Box(center, size, orientation)
        bbox.rotate(Quaternion(matrix=global_from_car))
        bbox.translate(global_from_car[:3, -1])

        return {"pc": pc, "3d_bbox": bbox, 'mode': self.mode}   

    @staticmethod
    def veh_pos_to_transform(veh_pos):
        def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                             rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                             inverse: bool = False) -> np.ndarray:
            """
            Convert pose to transformation matrix.
            :param translation: <np.float32: 3>. Translation in x, y, z.
            :param rotation: Rotation in quaternions (w ri rj rk).
            :param inverse: Whether to compute inverse transform matrix.
            :return: <np.float32: 4, 4>. Transformation matrix.
            """
            tm = np.eye(4)

            if inverse:
                rot_inv = rotation.rotation_matrix.T
                trans = np.transpose(-np.array(translation))
                tm[:3, :3] = rot_inv
                tm[:3, 3] = rot_inv.dot(trans)
            else:
                tm[:3, :3] = rotation.rotation_matrix
                tm[:3, 3] = np.transpose(np.array(translation))

            return tm

        "convert vehicle pose to two transformation matrix"
        rotation = veh_pos[:3, :3]
        tran = veh_pos[:3, 3]

        global_from_car = transform_matrix(
            tran, Quaternion(matrix=rotation), inverse=False
        )

        car_from_global = transform_matrix(
            tran, Quaternion(matrix=rotation), inverse=True
        )

        return global_from_car, car_from_global


