# This file is used to create a stratified splits based on videos.

"""
This is a boilerplate pipeline 'coco_datasplit'
generated using Kedro 0.18.4
"""
import json
import logging
import random
import collections
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from typing import Dict, List
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class TrainValTest_VideoNames(BaseModel):
    train: List[str]
    val: List[str]
    test: List[str]


class CocoVideoStratSplitter():

    def __init__(self, 
                 random_seed=42, 
                 shuffle=True,
                 coco_json: Dict = {}):
        
        assert len(coco_json) > 0, "coco_json should not be empty!"

        self.random_seed = random_seed
        self.shuffle = shuffle
        self.coco_json = coco_json

        self.X, self.y = self.__prepare_X_y(coco_json)

    def __prepare_X_y(self, coco_json: Dict):
        
        class_count = len(coco_json['categories'])
        logger.info(f'Classes num: {class_count}')
        
        video2classes = {}
        for ann in coco_json['annotations']:
            video_id = ann['image_id'].split('__')[0]
            category_id = ann['category_id']
            if video_id not in video2classes:
                video2classes[video_id] = set()
            video2classes[video_id].add(category_id)
            
        # Next line is important to strictly ensure repeatability
        video2classes = collections.OrderedDict(sorted(video2classes.items()))
        
        X = []
        y = []
        for video_name, labels in video2classes.items():
            X.append(video_name)
            y.append(list(labels))
            
        # recode y
        y_recoded = []
        for video_classes in y:
            recoded_classes = [0] * class_count
            for class_no in video_classes:
                recoded_classes[class_no] = 1
            y_recoded.append(recoded_classes)

        X = np.array(X)
        y = np.array(y_recoded)

        return X, y

    def xfold_over_xfold(self, inner_folds_num=5, outer_folds_num=5):
        raise NotImplementedError

    def xfold_over_test(self, inner_folds_num=10, outer_test_fraction: float = 0.15):
        raise NotImplementedError

    def repeating_xfold_over_repeating_xfold(self, inner_folds_num=3, inner_folds_test_fraction=0.15, outer_folds_num=3, outer_folds_fraction=0.15):
        raise NotImplementedError

    def repeating_xfold_over_test(self, inner_folds_num=3, inner_folds_test_fraction=0.15, outer_test_fraction=0.15):
        raise NotImplementedError

    def repeating_train_val_test(self, 
                                 number_of_repeats=5, 
                                 val_fraction=0.15, 
                                 test_fraction=0.15) -> List[TrainValTest_VideoNames]:
        
        msss_outer = MultilabelStratifiedShuffleSplit(n_splits=number_of_repeats, 
                                                random_state=self.random_seed, 
                                                test_size=test_fraction,
                                                train_size=None)
        
        splits_list: TrainValTest_VideoNames = []

        for train_index_outer, test_index_outer in msss_outer.split(self.X, self.y):
            
            train_val_X = self.X[train_index_outer]
            train_val_y = self.y[train_index_outer]
            adjusted_val_fraction = val_fraction / (1-test_fraction)

            test_X = self.X[test_index_outer].tolist()

            msss_inner = MultilabelStratifiedShuffleSplit(n_splits=1, 
                                                random_state=self.random_seed, 
                                                test_size=adjusted_val_fraction,
                                                train_size=None)
            
            for train_index_inner, val_index_inner in msss_inner.split(train_val_X, train_val_y):
                train_X = train_val_X[train_index_inner].tolist()
                val_X = train_val_X[val_index_inner].tolist()
                train_test_split = TrainValTest_VideoNames(train=train_X,
                                                           val=val_X,
                                                           test=test_X,
                                                        )
                
                splits_list.append(train_test_split)
        
        return splits_list
    
    def repeating_train_val_test_no_strat(self, number_of_repeats=5, val_fraction=0.15, test_fraction=0.15) -> List[TrainValTest_VideoNames]:
        all_videos = list(set([ann['image_id'].split('__')[0] for ann in self.coco_json['annotations']]))
        splits_list: List[TrainValTest_VideoNames] = []

        for repeat in range(number_of_repeats):
            random.seed(self.random_seed + repeat)
            random.shuffle(all_videos)
            test_size = int(len(all_videos) * test_fraction)
            val_size = int(len(all_videos) * val_fraction)

            test_videos = all_videos[:test_size]
            val_videos = all_videos[test_size:test_size + val_size]
            train_videos = all_videos[test_size + val_size:]

            train_test_split = TrainValTest_VideoNames(train=train_videos, val=val_videos, test=test_videos)
            splits_list.append(train_test_split)

        return splits_list

    def repeating_train_val_test_no_strat2(self, number_of_repeats=5, val_fraction=0.15, test_fraction=0.15) -> List[TrainValTest_VideoNames]:
        # Step 1: Perform stratified split for trainval vs test
        all_videos = list(set([ann['image_id'].split('__')[0] for ann in self.coco_json['annotations']]))
        # test_size = int(len(all_videos) * test_fraction)
        val_size = int(len(all_videos) * val_fraction)

        msss_outer = MultilabelStratifiedShuffleSplit(n_splits=number_of_repeats, 
                                                random_state=self.random_seed, 
                                                test_size=test_fraction,
                                                train_size=None)
        
        splits_list: List[TrainValTest_VideoNames] = []
        repeat = 0
        for train_index_outer, test_index_outer in tqdm(msss_outer.split(self.X, self.y), desc='Performing nonstrat2 splits'):
            
            train_val_X = self.X[train_index_outer]
            test_X = self.X[test_index_outer].tolist()

            random.seed(self.random_seed + repeat)
            random.shuffle(train_val_X)

            val_X = train_val_X[:val_size].tolist()
            train_X = train_val_X[val_size:].tolist()

            train_test_split = TrainValTest_VideoNames(train=train_X,
                                                    val=val_X,
                                                    test=test_X)
            
            splits_list.append(train_test_split)

            repeat += 1

        return splits_list
    
    def __train_val_test_split_statistic(self,
                                         train_val_test: TrainValTest_VideoNames):
        
        splits_stats = {'train': {}, 'test': {}, 'val': {}}
        for a in self.coco_json['annotations']:
            
            category_id = a['category_id']
            
            annotation_video = a['image_id'].split('__')[0]
            if annotation_video in train_val_test.train:
                set_name = 'train'
            elif annotation_video in train_val_test.val:
                set_name = 'val'
            elif annotation_video in train_val_test.test:
                set_name = 'test'
            else:
                raise Exception('Unknown video! ', annotation_video)
                
            if category_id not in splits_stats[set_name]:
                splits_stats[set_name][category_id] = 0
                
            splits_stats[set_name][category_id] = splits_stats[set_name][category_id] + 1
            
        return splits_stats
    

    def gather_train_val_test_split_list_statistics(self,
                                                    splits_list: List[TrainValTest_VideoNames]):
        
        return [self.__train_val_test_split_statistic(s) for s in splits_list]



    def coco_datasplit(self, 
                       coco_json: Dict,
                       chart_file_path: str,
                       num_splits: int):
        
        X, y = self.__prepare_X_y(coco_json)
        
        mskf = MultilabelStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

        test_set = None
        val_set = None
        train_set = []

        for _, test_index in mskf.split(X, y):
            if test_set is None:
                test_set = test_index
            elif val_set is None:
                val_set = test_index
            else:
                train_set.extend(test_index)
                
        train_video_names = X[train_set].tolist()
        val_video_names = X[val_set].tolist()
        test_video_names = X[test_set].tolist()
        glenda_coco_datasplits = {
            'train': train_video_names,
            'val': val_video_names,
            'test': test_video_names,
        }
        
        # Gather statistics about splits
        glenda_coco_datasplits_statistics = self.gather_splits_statistics(
            coco_json,
            train_video_names,
            val_video_names,
            test_video_names
        )
        
        # Generate data splits statistics chart
        self.create_save_data_split_chart(coco_json,
                                    glenda_coco_datasplits_statistics,
                                    chart_file_path)
        
        return glenda_coco_datasplits, glenda_coco_datasplits_statistics
        

    def gather_splits_statistics(self,
                                 data: Dict, 
                                 train_video_names: List[str],
                                 val_video_names: List[str],
                                 test_video_names: List[str]):
        
        splits_stats = {'train': {}, 'test': {}, 'val': {}}
        for a in data['annotations']:
            
            category_id = a['category_id']
            
            annotation_video = a['image_id'].split('__')[0]
            if annotation_video in train_video_names:
                set_name = 'train'
            elif annotation_video in val_video_names:
                set_name = 'val'
            elif annotation_video in test_video_names:
                set_name = 'test'
            else:
                raise Exception('Unknown video! ', annotation_video)
                
            if category_id not in splits_stats[set_name]:
                splits_stats[set_name][category_id] = 0
                
            splits_stats[set_name][category_id] = splits_stats[set_name][category_id] + 1
            
        return splits_stats


    def create_save_data_split_chart(self, 
                                     data: Dict, 
                                     splits_stats: Dict,
                                     chart_file_path: str):
        df = pd.DataFrame(splits_stats)
        df['class'] = df.index
        df_melted = pd.melt(df, id_vars=["class"], value_vars=["train", "test", "val"], var_name="set_type", value_name="count")
        cat_id2name = {c['id']:c['name'] for c in data['categories']}
        df_melted['class_name'] = df_melted['class'].map(cat_id2name)


        a4_dims = (18, 12)
        fig, ax = plt.subplots(figsize=a4_dims)
        g = sns.barplot(ax=ax, data=df_melted, x="set_type", y="count", hue="class_name")
        g.set_yscale("log")
        plt.savefig(chart_file_path)

    def create_save_train_test_val_split_charts(self, 
                                 data: Dict, 
                                 splits_stats_list: List[Dict],
                                 chart_file_path: str):
        cat_id2name = {c['id']:c['name'] for c in data['categories']}
        num_charts = len(splits_stats_list)
        
        # Create subplots with a variable number of rows and columns depending on the number of charts
        num_rows = (num_charts + 1) // 2  # Adjust the number of rows based on the number of charts
        fig, axs = plt.subplots(num_rows, 2, figsize=(18, 6*num_rows))
        
        for i, splits_stats in enumerate(splits_stats_list):
            df = pd.DataFrame(splits_stats)
            df['class'] = df.index
            df_melted = pd.melt(df, id_vars=["class"], value_vars=["train", "test", "val"], var_name="set_type", value_name="count")
            df_melted['class_name'] = df_melted['class'].map(cat_id2name)

            # Determine the subplot to use
            if num_rows > 1:
                ax = axs[i // 2, i % 2]
            else:
                ax = axs[i % 2]

            g = sns.barplot(ax=ax, data=df_melted, x="set_type", y="count", hue="class_name")
            g.set_yscale("log")
            ax.set_title(f"Chart {i+1}")

        plt.tight_layout()
        plt.savefig(chart_file_path)


if __name__ == "__main__":
    print('Hello, world!')

    coco_json_path = "/mnt/data_vilen/02_intermediate/glenda_full_v5_bbox_and_segm.json"
    with open(coco_json_path, "r") as file:
        coco_json = json.load(file)

    splitter = CocoVideoStratSplitter(coco_json=coco_json, 
                                      random_seed=42,
                                      shuffle=True)
    ####################
    # splits_tvt = splitter.repeating_train_val_test(number_of_repeats=5,
    #                                                val_fraction=0.15,
    #                                                test_fraction=0.15)
    
    # tvt_stats = splitter.gather_train_val_test_split_list_statistics(splits_tvt)

    # # Convert Pydantic objects to dictionaries
    # splits_tvt_dicts = [obj.model_dump() for obj in splits_tvt]

    # # Save the list of dictionaries to a JSON file
    # file_path = "/mnt/data_vilen/02_intermediate/repeated_stratified_train_val_test_split/5_splits_train_val_test.json"
    # with open(file_path, "w") as file:
    #     json.dump(splits_tvt_dicts, file, indent=4)
    

    # splitter.create_save_train_test_val_split_charts(data=coco_json,
    #                                                 splits_stats_list=tvt_stats,
    #                                                 chart_file_path='/mnt/data_vilen/02_intermediate/repeated_stratified_train_val_test_split/5_splits_train_val_test_stats.jpg')
    ####################

    # splits_tvt_nonstr = splitter.repeating_train_val_test_no_strat(number_of_repeats=5,
    #                                                         val_fraction=0.15,
    #                                                         test_fraction=0.15)
    
    # tvt_stats_nonstr = splitter.gather_train_val_test_split_list_statistics(splits_tvt_nonstr)

    # # Convert Pydantic objects to dictionaries
    # splits_tvt_nonstr_dicts = [obj.model_dump() for obj in splits_tvt_nonstr]

    # # Save the list of dictionaries to a JSON file
    # file_path = "/mnt/data_vilen/02_intermediate/repeated_nonstratified_train_val_test_split/5_splits_train_val_test.json"
    # with open(file_path, "w") as file:
    #     json.dump(splits_tvt_nonstr_dicts, file, indent=4)
    

    # splitter.create_save_train_test_val_split_charts(data=coco_json,
    #                                                 splits_stats_list=tvt_stats_nonstr,
    #                                                 chart_file_path='/mnt/data_vilen/02_intermediate/repeated_nonstratified_train_val_test_split/5_splits_train_val_test_stats.jpg')
    #####################

    splits_tvt_nonstr2 = splitter.repeating_train_val_test_no_strat2(number_of_repeats=5,
                                                            val_fraction=0.15,
                                                            test_fraction=0.15)
    
    tvt_stats_nonstr2 = splitter.gather_train_val_test_split_list_statistics(splits_tvt_nonstr2)

    # Convert Pydantic objects to dictionaries
    splits_tvt_nonstr_dicts2 = [obj.model_dump() for obj in splits_tvt_nonstr2]

    # Save the list of dictionaries to a JSON file
    file_path = "/mnt/data_vilen/02_intermediate/repeated_nonstratified2_train_val_test_split/5_splits_train_val_test.json"
    with open(file_path, "w") as file:
        json.dump(splits_tvt_nonstr_dicts2, file, indent=4)
    

    splitter.create_save_train_test_val_split_charts(data=coco_json,
                                                    splits_stats_list=tvt_stats_nonstr2,
                                                    chart_file_path='/mnt/data_vilen/02_intermediate/repeated_nonstratified2_train_val_test_split/5_splits_train_val_test_stats.jpg')
    
    