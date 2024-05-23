import math
import pandas as pd
import geopandas as gpd
import numpy as np
from copy import deepcopy
from utilsforminds.containers import merge_dictionaries
import random
import os
from collections import OrderedDict
from tqdm import tqdm
import utilsforminds
import group_finder
import skimage
import cv2
from lungmask import mask
import SimpleITK as sitk
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm
import imageio
import utils
from datetime import datetime, timedelta

pd.set_option('display.max_columns', 100)

def reorder_snp(basic_data, snp_data):
    """Reorder the SNP Data to be correctly associated with the other .xlsx data frames"""
    tmp = snp_data['ID'].apply(lambda x: x[-4:])
    tmp.rename('sortID', inplace=True)

    snp_data = pd.concat([tmp, snp_data], axis=1)
    snp_data.sort_values(by='sortID', inplace=True)
    snp_data.drop(columns=['sortID'], inplace=True)

    all_patients = basic_data["SubjID"]
    snp_final = pd.merge(all_patients.to_frame(), snp_data, left_on='SubjID', right_on='ID', how='outer')

    # Remove 018_S_0055 (because it doesn't exist in longitudinal data
    snp_final = snp_final[snp_final.ID != '018_S_0055']

    # Remove unwanted ID labels
    snp_final.drop(columns=['SubjID', 'ID'], inplace=True)
    return snp_final

def rand_gen_with_range(range_ = None):
    if range_ is None : range_ = [0., 1.]
    assert(range_[0] <= range_[1])
    return range_[0] + (range_[1] - range_[0]) * random.random()

class Dataset():
    def __init__(self, path_to_dataset, obj_path, dataset_kind = "covid", init_number_for_unobserved_entry = 0.0, static_data_input_mapping = None, kwargs_dataset_specific = None, if_separate_img = False):
        """Preprocess the input raw tabular dataset.

        The most important attributes are self.dicts and self.input_to_features_map.
        
        Attributes
        ----------
        dicts : list of dict
            List of dictionaries of patients records where each dictionary for each patient.
        static_data_input_mapping : dict
            Static feature -> input, let the user to decide which static feature feed which input. For example, static_data_input_mapping = dict(RNN_vectors = [], static_encoder = ["SNP"], raw = ["BL_age", "Gender"]).
        prediction_labels_bag : list
            The set of possible target labels, for example, for COVID-19 dataset: [0, 1], for Alz dataset: [1, 2, 3].
        input_to_features_map : dict
            The feature names of each input. For example, looks like a dict(RNN_vectors= [], static_encoder= [], raw= [], dynamic_vectors_decoder= []). The stack order of numpy array data follows the order of self.input_to_features_map, except the static feature precedes the dynamic features in RNN input.
        groups_of_features_info : 
            Used to calculate the feature importance of each plotting group. For example, self.groups_of_features_info[feature_group = "FS"][feature_name] = {"input": self.feature_to_input_map[name], "idx": self.input_to_features_map[self.feature_to_input_map[name]].index(name), "observed_numbers": [], "mean": None, "std": None}.
        """

        self.this_dataset_is_splitted = False ## Not yet k-fold splitted.
        self.shape_2D_records = None ## What is the dimension of image input if it has?, None means raw image shape which can be different across the participants.
        self.if_contains_imgs = False
        self.if_contains_static_img = False
        self.init_number_for_unobserved_entry = init_number_for_unobserved_entry
        self.ID = utilsforminds.helpers.get_current_utc_timestamp()
        self.dataset_kind = dataset_kind
        self.path_to_dataset = path_to_dataset
        self.if_separate_img = if_separate_img
        self.obj_path = obj_path
        self.input_to_features_map = dict(RNN_vectors= [], static_encoder= [], raw= [], dynamic_vectors_decoder= []) ## Feature names for each input.
        self.classification_axis = None
        self.if_partial_label = False
        self.dummy_name_to_original_name = {}
        self.if_create_data_frame_each_bag = False

        if self.if_separate_img:
            self.obj_dir_path = self.obj_path[:-4]
            if not os.path.isdir(self.obj_dir_path):
                os.mkdir(self.obj_dir_path)

        ## Set class names for ROC-curve plot, sequence is decided by one-hot encoded form, such that  self.class_names = [name of [1, 0, 0], name of [0, 1, 0], name of [0, 0, 1]]
        if self.dataset_kind == "challenge":
            self.class_names = ["survival", "death"]
        elif self.dataset_kind == "alz":
            self.class_names = ["AD", "MCI", "HC"]
        elif self.dataset_kind == "traffic":
            self.class_names = ["Congested", "Mild", "Free"]
        elif self.dataset_kind == "colorado_traffic":
            self.class_names = ["near_crash"]
        else:
            self.class_names = None

        if dataset_kind == "covid":
            ## Set default option.
            self.static_data_input_mapping = merge_dictionaries([dict(RNN_vectors = [], static_encoder = [], raw = ["age", "gender", 'Admission time', 'Discharge time']), static_data_input_mapping])
            self.static_column_names = ['PATIENT_ID', "age", "gender", 'Admission time', 'Discharge time', "outcome"] ## All the information regardless of whether it is used by models or not.
            self.input_to_features_map = merge_dictionaries([self.input_to_features_map, self.static_data_input_mapping]) ## Information actually used by the models. \in self.static_column_names.
            self.dynamic_feature_names = []

            ## Set dataframe.
            self.path_to_dataset = path_to_dataset
            self.dataframe_raw = pd.read_excel(path_to_dataset)
            self.dataframe = self.dataframe_raw.copy()

            ## Set dynamic features names.
            for name in self.dataframe_raw.columns: ## Should not include target label, such as 'outcome'.
                if name not in self.static_column_names + kwargs_dataset_specific["excluded_features"]: 
                    self.dynamic_feature_names.append(name)
                    self.input_to_features_map["RNN_vectors"].append(name)

            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(self.input_to_features_map["RNN_vectors"])
            self.input_to_features_map["dynamic_vectors_decoder"].remove("RE_DATE") ## decoder labels = LSTM labels - RE_DATE.
            self.dynamic_feature_names.remove("RE_DATE")
            # for name in self.decoder_labels_feature_names:
            #     if name in self.static_column_names:
            #         self.decoder_labels_feature_names_is_static.append(1.)
            #     else:
            #         self.decoder_labels_feature_names_is_static.append(0.)
            # self.decoder_labels_feature_names_is_static = np.array(self.decoder_labels_feature_names_is_static)

            self.scale_times()
            self.min_max_scale()
            self.set_observed_numbers_pool_for_each_group_of_features()
        
        elif dataset_kind == "chestxray":
            self.if_contains_imgs = True
            self.image_info = {"RNN_2D": {"num_rgb": 1}}
            self.shape_2D_records = kwargs_dataset_specific["image_shape"]
            self.apply_segment = kwargs_dataset_specific["apply_segment"]
            if self.if_separate_img:
                assert(not self.apply_segment)
            ## Set dataframe.
            self.path_to_dataset = path_to_dataset
            self.dataframe_raw = pd.read_csv(os.path.join(path_to_dataset, "metadata.csv"))
            self.dataframe = self.dataframe_raw.copy()
            self.dataframe = self.dataframe.drop(columns = kwargs_dataset_specific["excluded_features"])
            categorical_columns = [name for name in ["sex", "finding", "RT_PCR_positive", "survival", "intubated", "intubation_present", "went_icu", "in_icu", "needed_supplemental_O2", "extubated", "view", "modality"] if name not in kwargs_dataset_specific["excluded_features"]]

            ## One-hot encoding
            self.dataframe = pd.get_dummies(self.dataframe, columns = categorical_columns)
            df_columns = self.dataframe.columns.to_list()

            ## Set default option.
            self.static_data_input_mapping = merge_dictionaries([dict(RNN_vectors = [], static_encoder = [], raw = ["sex", "age", "RT_PCR_positive"]), static_data_input_mapping])
            self.static_column_names = ['patientid', "sex", "age", "RT_PCR_positive", "finding", "survival", "intubated", "went_icu", "needed_supplemental_O2", "extubated", "filename"] ## All the information regardless of whether it is used by models or not.
            self.input_to_features_map = merge_dictionaries([self.input_to_features_map, self.static_data_input_mapping]) ## Information actually used by the models. \in self.static_column_names.
            self.dynamic_feature_names = []

            ## One-hot encoded colmun names.
            for dict_ in [self.static_data_input_mapping, self.input_to_features_map]:
                for key in dict_.keys():
                    dict_[key] = change_names_from_starting_names(dict_[key], df_columns)
            self.static_column_names = change_names_from_starting_names(self.static_column_names, df_columns)

            ## Set dynamic features names.
            for name in df_columns: ## Should not include target label, such as 'outcome'.
                if name not in self.static_column_names: 
                    self.dynamic_feature_names.append(name)
                    self.input_to_features_map["RNN_vectors"].append(name)

            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(self.input_to_features_map["RNN_vectors"])
            self.input_to_features_map["dynamic_vectors_decoder"].remove("offset") ## decoder labels = LSTM labels - RE_DATE.
            self.dynamic_feature_names.remove("offset")

            ## One-hot encode for categorical columns
            self.min_max_scale()
            self.set_observed_numbers_pool_for_each_group_of_features()

        elif dataset_kind == "traffic":
            self.if_contains_imgs = True
            self.image_info = {"RNN_2D": {"num_rgb": 1}}
            self.if_contains_static_img = True
            self.shape_2D_records = kwargs_dataset_specific["image_shape"]
            self.apply_segment = kwargs_dataset_specific["apply_segment"]
            self.prediction_interval = kwargs_dataset_specific["prediction_interval"]
            self.num_bags_max = kwargs_dataset_specific["num_bags_max"]
            self.num_records = kwargs_dataset_specific["num_records"]

            if self.if_separate_img:
                assert (not self.apply_segment)
            ## Set dataframe.
            self.path_to_dataset = path_to_dataset
            self.dataframe_raw = pd.read_csv(os.path.join(path_to_dataset, "austin_data_with_speeds.csv"))
            dynamic_filenames = os.listdir(os.path.join(path_to_dataset, "Austin Images"))
            self.dataframe = self.dataframe_raw.copy()
            self.dataframe = self.dataframe.drop(columns=kwargs_dataset_specific["excluded_features"])
            categorical_columns = [name for name in
                                   ["Road_Class_WE", "Road_Class_NS", "One_Way_WE", "One_Way_NS", "Zoning_Code_NW",
                                    "Zoning_Type_NW", "Zoning_Code_NE", "Zoning_Type_NE", "Zoning_Code_SW",
                                    "Zoning_Type_SW", "Zoning_Code_SE", "Zoning_Type_SE", "FLUM", "Urban",
                                    "Proximity_to_Wildland"] if
                                   name not in kwargs_dataset_specific["excluded_features"]]
            self.dataframe = self.dataframe.drop(self.dataframe[(self.dataframe['image_0'] == "no image") | (
                        self.dataframe['image_5'] == "no image") | (self.dataframe['image_10'] == "no image") | (
                                                                    ~self.dataframe['image_0'].isin(
                                                                        dynamic_filenames)) | (
                                                                    ~self.dataframe['image_5'].isin(
                                                                        dynamic_filenames)) | (
                                                                    ~self.dataframe['image_10'].isin(
                                                                        dynamic_filenames))].index)
            self.dataframe = self.dataframe.fillna(0)

            ## One-hot encoding
            self.static_column_names = []  ## All the information regardless of whether it is used by models or not.
            # dummy_to_categ_map = {}
            for categ_feature in categorical_columns:
                df_dummy = self.dataframe.loc[:, categ_feature].str.get_dummies()
                dummy_col_names = [categ_feature + '_is_' + col for col in df_dummy.columns]
                # for dummy in dummy_col_names:
                #     dummy_to_categ_map[dummy] = categ_feature
                self.static_column_names += dummy_col_names
                df_dummy.columns = deepcopy(dummy_col_names)
                self.dataframe = pd.concat([self.dataframe, df_dummy], axis=1)

            ## interpolates the missing labels.
            self.target_columns = []
            for i in range(len(kwargs_dataset_specific["target_columns"])):
                colname = kwargs_dataset_specific["target_columns"][i]
                self.target_columns.append(f"target_{colname}")
                self.dataframe[f"target_{colname}"] = self.dataframe[colname]
            self.dataframe["datetime"] = pd.to_datetime(self.dataframe['time_id'],
                                                        format="%H:%M %Y-%m-%d")  ## not '2022/11/21  8:15:00 PM', yes '09:00 2022-11-22'
            self.dataframe = self.dataframe.set_index("datetime")
            austin_image_info_df_labels = self.dataframe[self.target_columns]
            # # set_index(austin_image_info_df['minutes_scaled'])
            austin_image_info_df_labels = austin_image_info_df_labels.replace(to_replace=-1, value=np.nan)
            austin_image_info_df_labels = austin_image_info_df_labels.interpolate(method="time")
            austin_image_info_df_labels = austin_image_info_df_labels.fillna(method='bfill')
            self.labels_info = {"speeds_means": [], "speeds_threshold": [], "num_label_divisions": 3}
            for i in range(austin_image_info_df_labels.shape[0]):
                self.labels_info["speeds_means"].append(np.mean(austin_image_info_df_labels.iloc[i].to_numpy()))
            self.labels_info["speeds_means"] = sorted(self.labels_info["speeds_means"])
            for i in range(1, self.labels_info["num_label_divisions"]):
                self.labels_info["speeds_threshold"].append(self.labels_info["speeds_means"][
                                                                int(len(self.labels_info["speeds_means"]) * (
                                                                            i / self.labels_info[
                                                                        "num_label_divisions"]))])
            self.dataframe[self.target_columns] = austin_image_info_df_labels[self.target_columns]

            ## Set default option.
            self.static_data_input_mapping = merge_dictionaries(
                [dict(RNN_vectors=[], static_encoder=[], raw=self.static_column_names), static_data_input_mapping])
            self.input_to_features_map = merge_dictionaries([self.input_to_features_map,
                                                             self.static_data_input_mapping])  ## Information actually used by the models. \in self.static_column_names.
            self.dynamic_feature_names = []

            df_columns = self.dataframe.columns.to_list()
            ## One-hot encoded colmun names.
            for dict_ in [self.static_data_input_mapping, self.input_to_features_map]:
                for key in dict_.keys():
                    dict_[key] = change_names_from_starting_names(dict_[key], df_columns)
            self.static_column_names = change_names_from_starting_names(self.static_column_names, df_columns)

            ## Set dynamic features names.
            self.dataframe['minutes'] = self.dataframe.apply(lambda row: utils.convert_time_to_minutes(row.image_0),
                                                             axis=1)
            self.dataframe['minutes_scaled'] = self.dataframe['minutes']
            dynamic_features = kwargs_dataset_specific[
                "dynamic_features"] if "dynamic_features" in kwargs_dataset_specific.keys() else self.target_columns + [
                "minutes_scaled"]
            for name in dynamic_features:  ## Should not include target label, such as 'outcome'.
                if name not in self.static_column_names:
                    self.dynamic_feature_names.append(name)
                    self.input_to_features_map["RNN_vectors"].append(name)

            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(self.input_to_features_map["RNN_vectors"])
            self.input_to_features_map["dynamic_vectors_decoder"].remove(
                "minutes_scaled")  ## decoder labels = LSTM labels - RE_DATE.
            self.dynamic_feature_names.remove("minutes_scaled")

            ## One-hot encode for categorical columns
            self.min_max_scale()
            self.set_observed_numbers_pool_for_each_group_of_features()

        elif dataset_kind == "colorado_traffic":
            self.if_contains_static_img = True
            self.if_contains_imgs = False
            self.if_partial_label = True
            self.if_create_data_frame_each_bag = True

            self.num_bags_max = kwargs_dataset_specific['num_bags_max']  # Maximum number of bags to build.
            self.num_records = kwargs_dataset_specific["num_records"] if "num_records" in kwargs_dataset_specific.keys(
            ) else [5, 10, 15]
            if not isinstance(self.num_records, (list, tuple)): self.num_records = [self.num_records]
            self.target_columns = kwargs_dataset_specific['target_columns']
            self.classification_axis = None if 'future_congestion' not in self.target_columns else self.target_columns.index('near_crash')

            # Maximum time gap for observations within the same bag.
            self.max_time_gap_hrs = kwargs_dataset_specific["max_time_gap_hrs"]
            # How far ahead to look when computing future congestion.
            self.future_congestion_lookahead_hrs = kwargs_dataset_specific["future_congestion_lookahead_hrs"]

            # Parameters for computing near crash.
            self.crash_spatial_thresh = kwargs_dataset_specific["crash_spatial_thresh"]
            self.crash_temporal_thresh = kwargs_dataset_specific["crash_temporal_thresh"]
            self.compute_near_crash = kwargs_dataset_specific["compute_near_crash"]

            self.path_to_dataset = path_to_dataset
            filename = os.path.join(path_to_dataset, "CDOT_Traffic_Counts_Full_Combined_Weather_Hourly.csv")
            if True:
                num_rows_total = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header), max: 11244216
                num_rows_sample = 50000 #desired sample size
                # skiprows = sorted(random.sample(range(1,num_rows_total+1), num_rows_total-num_rows_sample)) #the 0-indexed header will not be included in the skip list
                
                self.dataframe = pd.read_csv(filename, usecols= ["datetime", "count_station_id"])
                chunk_size = 100
                skiprows = []
                datetime_to_station_ids = {}
                i = 1
                while i < num_rows_total - chunk_size and len(skiprows) < num_rows_total - num_rows_sample:
                    row = self.dataframe.iloc[i]
                    if row["datetime"] < '2023-09-21 00:00:00': ## original range: '2010-01-01 00:00:00' ~ '2023-09-30 23:00:00'
                        for j in range(i, i + chunk_size):
                            skiprows.append(j)
                    elif row["datetime"] not in datetime_to_station_ids.keys():
                        datetime_to_station_ids[row["datetime"]] = {row["count_station_id"]}
                    elif row["count_station_id"] not in datetime_to_station_ids[row["datetime"]]:
                        datetime_to_station_ids[row["datetime"]].add(row["count_station_id"])
                    elif random.random() < (num_rows_total-num_rows_sample) / (num_rows_total):
                        for j in range(i, i + chunk_size):
                            skiprows.append(j)
                    i += chunk_size

                self.dataframe = pd.read_csv(filename, skiprows= skiprows) ## pd.read_csv(os.path.join(path_to_dataset, "CDOT_Traffic_Counts_Full_Combined_Weather_Hourly.csv"), nrows=1000)
            else:
                self.dataframe = pd.read_csv(filename)

            # Load file containing geometry of colorado highways. # TODO: Make use of this data
            # self.static_spatial_data = gpd.GeoDataFrame.from_file(
            #     os.path.join(kwargs_dataset_specific['path_to_dataset'], 'Highways.geojson'))

            # self.dataframe = self.dataframe_raw.copy()
            self.dataframe = self.dataframe.drop(columns=kwargs_dataset_specific['excluded_features'], axis=1)

            self.dataframe = self.dataframe.fillna(init_number_for_unobserved_entry)  # Fill all missing values
            self.dataframe['datetime'] = pd.to_datetime(self.dataframe['datetime'])  # Convert from string to timestamp
            self.dataframe['minutes'] = self.dataframe['datetime'].astype('int64') / (10**9) / 60 # Convert to unix timestamp minutes
            for column in ["lat", "lon"]:
                self.dataframe[f"{column}_raw"] = deepcopy(self.dataframe[column])
            self.dataframe['minutes_scaled'] = deepcopy(self.dataframe['minutes'])
            df_columns = self.dataframe.columns.to_list()

            # Label which are categorical and which are numerical for encoding and scaling respectively.
            self.categorical_columns = [
                x for x in
                ['county', 'location', 'direction', 'day_of_week', 'route', 'population',
                 'func_class', 'access', 'prime_out_shoulder', 'median', 'precip_type', 'conditions']
                if x not in kwargs_dataset_specific['excluded_features']
            ]
            # Only need to scale minutes. Other date information will be dropped.
            self.numerical_columns = [x for x in df_columns if x not in self.categorical_columns and
                                      x not in [*self.target_columns, 'hour', 'day', 'month', 'year', 'datetime', 'count_station_id']]

            # Columns containing static information which remains constant for a bag.
            self.static_column_names = np.setdiff1d(['county', 'location', 'start_mile', 'end_mile',
                                                     'route', 'population', 'func_class', 'access', 'thru_lane_qty',
                                                     'thru_lane_width', 'speed_lim', 'prime_out_shoulder', 'median',
                                                     'prime_out_shoulder_width',
                                                     'lat', 'lon'],
                                                    kwargs_dataset_specific['excluded_features']).tolist()
            self.dynamic_feature_names = [x for x in df_columns if x not in [*self.static_column_names,
                                                                             *self.target_columns,
                                                                             'hour', 'day', 'month',
                                                                             'year', 'datetime', 'future_count', 'count_station_id'] + kwargs_dataset_specific['excluded_features'] and not x.endswith('_raw')]

            # if print_feature_info:
            #     print('Categorical Columns: ', self.categorical_columns)
            #     print('Numerical Columns: ', self.numerical_columns)
            #     print('Static Columns: ', self.static_column_names)
            #     print('Dynamic Columns: ', self.dynamic_feature_names, end='\n\n\n')

            # Compute the future congestion using self.future_congestion_lookahead_hrs
            self.__compute_future_congestion()
            if self.compute_near_crash:
                # Compute if each sample is near a crash using provided spatial and temporal thresholds.
                self.__compute_near_crash()
                # self.dynamic_feature_names.append('near_crash')

            # One-hot encode categorical columns.
            one_hot = pd.get_dummies(self.dataframe[self.categorical_columns])  # Encode all categorical columns.
            for dummy_name in one_hot.columns:
                for original_name in self.categorical_columns:
                    if dummy_name.startswith(original_name):
                        original_name_selected = original_name
                assert(dummy_name not in self.dummy_name_to_original_name.keys())
                self.dummy_name_to_original_name[dummy_name] = original_name_selected

            self.dataframe.drop(self.categorical_columns, axis=1, inplace=True)  # Drop old columns.
            self.dataframe = self.dataframe.join(one_hot)  # Join encodings with the dataframe.
            df_columns = self.dataframe.columns.to_list()  # Update column names

            # Set default option.
            self.static_data_input_mapping = merge_dictionaries([dict(RNN_vectors = [], static_encoder = [], raw = self.static_column_names), static_data_input_mapping])
            self.input_to_features_map = merge_dictionaries([self.input_to_features_map, self.static_data_input_mapping]) ## Information actually used by the models. \in self.static_column_names.

            # One-hot encoded column names.
            for dict_ in [self.static_data_input_mapping, self.input_to_features_map]:
                for key in dict_.keys():
                    dict_[key] = change_names_from_starting_names(dict_[key], df_columns)

            # Ensure column names are updated after the one-hot encoding.
            self.static_column_names = change_names_from_starting_names(self.static_column_names, df_columns)
            self.dynamic_feature_names = change_names_from_starting_names(self.dynamic_feature_names, df_columns)

            for name in self.dynamic_feature_names:  # Should not include target label.
                if name not in self.static_column_names + ["minutes"]:  # Be sure it is not in the static column.
                    self.input_to_features_map["RNN_vectors"].append(name)
                else:  # If needed, remove from list.
                    self.dynamic_feature_names = list(filter(lambda x: x != name, self.dynamic_feature_names))

            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(self.input_to_features_map["RNN_vectors"])

            self.dataframe['datetime_timestamp'] = self.dataframe.apply(lambda row: row['datetime'].timestamp(), axis = 1)
            # Drop unnecessary date-time information
            self.dataframe.drop(['hour', 'day', 'month', 'year', 'datetime'], axis=1, inplace=True)

            # if print_feature_info:
            #     print('Static Columns after one-hot:', self.static_column_names)
            #     print('Dynamic Columns after one-hot:', self.dynamic_feature_names, end='\n\n')

            self.min_max_scale()  # Scale the numerical features (Also scale).
            self.set_observed_numbers_pool_for_each_group_of_features()

        
        elif dataset_kind == "challenge":
            ## Set default option.
            self.static_data_input_mapping = merge_dictionaries([dict(RNN_vectors = [], static_encoder = [], raw = ["Age", "Gender", 'Height', "Weight", "CCU", "CSRU", "SICU"]), static_data_input_mapping])
            self.static_column_names = ["SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death", "Age", "Gender", 'Height', "Weight", "CCU", "CSRU", "SICU", ] ## All the information regardless of whether it is used by models or not.
            self.input_to_features_map = merge_dictionaries([self.input_to_features_map, self.static_data_input_mapping]) ## Information actually used by the models. \in self.static_column_names.
            self.dynamic_feature_names = []

            ## Set dataframe.
            self.path_to_dataset = path_to_dataset
            self.dataframe_static = pd.read_csv(os.path.join(path_to_dataset, "PhysionetChallenge2012-static-set-a.csv"))
            self.dataframe_dynamic = pd.read_csv(os.path.join(path_to_dataset, "PhysionetChallenge2012-temporal-set-a.csv"))

            ## Set dynamic features names.
            for name in self.dataframe_dynamic.columns: ## Should not include target label, such as 'outcome'.
                if name != "recordid": 
                    self.dynamic_feature_names.append(name)
                    self.input_to_features_map["RNN_vectors"].append(name)

            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(self.input_to_features_map["RNN_vectors"])
            self.input_to_features_map["dynamic_vectors_decoder"].remove("time") ## decoder labels = LSTM labels - RE_DATE.
            self.dynamic_feature_names.remove("time")

            # self.scale_times()
            self.set_observed_numbers_pool_for_each_group_of_features(excluded_features = ["time"])
            # self.dataframe_static = self.dataframe_static.fillna(init_number_for_unobserved_entry)
            # self.dataframe_dynamic = self.dataframe_dynamic.fillna(init_number_for_unobserved_entry)
            self.min_max_scale()

        elif dataset_kind == "toy":
            # raise Exception("Deprecated.")
            self.set_observed_numbers_pool_for_each_group_of_features() ## Just to prevent exception, actually do nothing.
            pass

        elif dataset_kind == "alz":
            ## Set default keyword arguments.
            kwargs_dataset_specific = merge_dictionaries([dict(TIMES = ["BL", "M6", "M12", "M18", "M24", "M36"], target_label = "CurrentDX_032911", dynamic_modalities_to_use = ["FS", "VBM"]), kwargs_dataset_specific]) ## You can add cognitive tests to dynamic_modalities_to_use.
            self.static_data_input_mapping = merge_dictionaries([dict(RNN_vectors = [], static_encoder = ["SNP"], raw = []), static_data_input_mapping]) ## dict(RNN_vectors = [], static_encoder = ["SNP"], raw = ["BL_Age", "Gender"]) ## You can add age/gender.
            
            ## Features extracted from the participant's basic info.
            info_feature_names = []
            for list_ in self.static_data_input_mapping.values():
                for feature in list_:
                    if feature != "SNP": info_feature_names.append(feature)

            ## --- Set SNPs info dataframe for SNPs plots.
            ## Get the indices of SNPs group.
            SNP_identified_group_df = pd.read_excel(path_to_dataset + 'CanSNPs_Top40Genes_org.xlsx', 'CanSNPs_Top40Genes_LD', usecols=['L1', 'L2', 'r^2'])
            SNP_label_dict = utilsforminds.biomarkers.get_SNP_name_sequence_dict(path_to_dataset + 'snp_labels.csv')
            adjacency_matrix = utilsforminds.biomarkers.get_adjacency_matrix_from_pairwise_value(SNP_label_dict, SNP_identified_group_df, ['L1', 'L2'], 'r^2', 0.2)
            group_finder_inst = group_finder.Group_finder(adjacency_matrix)
            snp_group_idc_lst = group_finder_inst.connected_components() ## snp_group_idc_lst = [[141, 423, 58, ...], [indices of second group], ...]

            #%% Reorder SNP data following groups
            snps_popular_group_df = pd.read_excel(path_to_dataset + 'CanSNPs_Top40Genes_org.xlsx', sheet_name = "CanSNPs_Top40Genes_Annotation")
            reordered_SNPs_info_list = []
            col_names = ["index_original", "identified_group", "SNP", "chr", "AlzGene", "location"]
            for idc_lst, group_idx in zip(snp_group_idc_lst, range(len(snp_group_idc_lst))): ## for indices(in original SNPs array) of each SNPs group
                for idx in idc_lst:
                    reordered_SNPs_info_list.append([idx, group_idx] + list(snps_popular_group_df.loc[idx, ["SNP", "chr", "AlzGene", "location"]]))
            self.reordered_SNPs_info_df = pd.DataFrame(reordered_SNPs_info_list, columns = col_names) ## For SNP group-wise plots.
            # self.static_feature_names = self.reordered_SNPs_info_df['SNP'].tolist()
            series_obj = self.reordered_SNPs_info_df.apply(lambda x: True if x['chr'] != 'X' else False, axis = 1) ## True if 'chr' column is not 'X'.
            self.idc_chr_not_X_list = list(series_obj[series_obj == True].index) ## list of indices where 'chr' column is not 'X'.
            self.reordered_SNPs_info_df = self.reordered_SNPs_info_df.loc[self.idc_chr_not_X_list, :]
            self.colors_list = ["red", "navy", "lightgreen", "teal", "violet", "green", "orange", "blue", "coral", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"]
            for group_column in ['chr', 'AlzGene', 'location', 'identified_group']:
                self.reordered_SNPs_info_df = utilsforminds.helpers.add_column_conditional(self.reordered_SNPs_info_df, group_column, self.colors_list, new_column_name= group_column + "_colors") ## Adds color column for each group.

            ### Actual dataframe preprocess.
            ## Set time stamp.
            self.TIMES = deepcopy(kwargs_dataset_specific["TIMES"])
            # self.TIMES_TO_RE_DATE = {self.TIMES[i]: (i + 1) / len(self.TIMES) for i in range(len(self.TIMES))} ## Normalize the time stamp.
            self.TIMES_TO_RE_DATE = dict(BL= 0.0, M6= 0.1, M12= 0.2, M18= 0.3, M24= 0.4, M36= 0.6,) ## Normalize the time stamp.
            self.dynamic_modalities_sequence = deepcopy(kwargs_dataset_specific["dynamic_modalities_to_use"]) ## ["FS", "VBM", "ADAS", "FLU", "MMSE", "RAVLT", "TRAILS"]

            ## --- --- Set dataframes of dataset.
            self.df_dict = {}
            ## Set dynamic dataframes.
            file_and_sheet_name_base_dict = dict(FS = dict(file= "longitudinal imaging measures_FS_final.xlsx", sheet_name= "FS_"), 
            VBM = dict(file= "longitudinal imaging measures_VBM_mod_final.xlsx",sheet_name= "VBM_mod_"), 
            RAVLT = dict(file= "longitudinal cognitive scores_RAVLT.xlsx", sheet_name= "RAVLT_"), 
            ADAS = dict(file= "longitudinal cognitive scores_ADAS.xlsx", sheet_name= "ADAS_"), 
            FLU = dict(sheet_name= "FLU_", file= "longitudinal cognitive scores_FLU.xlsx"),
            MMSE = dict(sheet_name= "MMSE_", file= "longitudinal cognitive scores_MMSE.xlsx"),
            TRAILS = dict(sheet_name= "TRAILS_", file= "longitudinal cognitive scores_TRAILS.xlsx"),)
            df_dict_of_dynamic_temp = {time: {modality: pd.read_excel(path_to_dataset + file_and_sheet_name_base_dict[modality]["file"], sheet_name=file_and_sheet_name_base_dict[modality]["sheet_name"] + time, header=0) for modality in self.dynamic_modalities_sequence} for time in self.TIMES}

            ## Set static dataframes.
            self.df_dict["snp"] = pd.read_excel(path_to_dataset + "CanSNPs_Top40Genes_org.xlsx", sheet_name="CanSNPs_Top40Genes_org", header=0) ## Reorder the order of SNPs features, [["ID"] + self.static_feature_names]
            self.df_dict["info"] = pd.read_excel(path_to_dataset + "longitudinal basic infos.xlsx", sheet_name="Sheet1", header=0)
            self.df_dict["snp"] = reorder_snp(self.df_dict["info"], self.df_dict["snp"]) ## Reorder the order of SNPs features, [["ID"] + self.static_feature_names], the indices of SNPs become consistent with the patient's sequence of other biomarkers.
            self.df_dict["static"] = self.df_dict["info"][info_feature_names]
            self.df_dict["outcome"] = pd.read_excel(path_to_dataset + "longitudinal diagnosis.xlsx", sheet_name="Sheet1", header=0)[kwargs_dataset_specific["target_label"]]

            # For naive prediction (the same diagnosis at the right before time point).
            # diagnosis_df = pd.read_excel(path_to_dataset + "longitudinal diagnosis.xlsx", sheet_name="Sheet1", header=0)
            # matches_dict = {"M12_DX": 0, "M18_DX": 0, "M24_DX": 0, "remaining": 0, "total": 0}
            # for index, row in diagnosis_df.iterrows():
            #     if not pd.isnull(row["M36_DX"]):
            #         matches_dict["total"] += 1
            #         if not pd.isnull(row["M24_DX"]):
            #             if row["M24_DX"] == row["M36_DX"]: matches_dict["M24_DX"] += 1
            #         elif not pd.isnull(row["M18_DX"]):
            #             if row["M18_DX"] == row["M36_DX"]: matches_dict["M18_DX"] += 1
            #         elif not pd.isnull(row["M12_DX"]):
            #             if row["M12_DX"] == row["M36_DX"]: matches_dict["M12_DX"] += 1
            #         else:
            #             matches_dict["remaining"] += 1

            ## Set static feature names, for RNN, static precedes the dynamic. 
            self.static_modality_to_features_map = {"SNP": list(self.df_dict["snp"].columns)[1:]} ## To handle grouped SNP features and individual info static feature.
            for input_ in self.static_data_input_mapping.keys(): ## input for static.
                for modality in self.static_data_input_mapping[input_]:
                    if modality == "SNP":
                        self.input_to_features_map[input_] = deepcopy(self.input_to_features_map[input_] + list(self.df_dict["snp"].columns)[1:]) ## Get rid of the ID label
                    else:
                        self.input_to_features_map[input_].append(modality)
                        self.static_modality_to_features_map[modality] = [modality] ## Non-group, just single modality.

            ### Set feature names of dynamic modality.


            self.dynamic_feature_names_for_each_modality = {modality: [] for modality in self.dynamic_modalities_sequence}
            ## --- Set 'feature name' of FS and VBM using baseline timepoint.
            for modality in self.dynamic_modalities_sequence: ## FS, VBM, or other cognitive scores.
                for column_name in df_dict_of_dynamic_temp["BL"][modality].columns: ## Use column names of BL timepoint.
                    ## Set feature name.
                    def converter_column_name(column_name): ## Rename dynamic column names.
                        column_name_splitted = column_name.split('_')
                        if len(column_name_splitted) >= 3:
                            return f"{modality}_{column_name_splitted[-2]}_{column_name_splitted[-1]}" ## Use last TWO blocks for dynamic feature name.
                        else:
                            return f"{modality}_" + "_".join(column_name_splitted[1:]) ## [1:] to remove time label.
                    converted_column_name = converter_column_name(column_name)
                    
                    self.input_to_features_map["RNN_vectors"].append(converted_column_name) ## [(static features), dynamic data, RE_DATE] or [(static features), dynamic data, RE_DATE, (static masks), dynamic mask, RE_DATE]
                    self.dynamic_feature_names_for_each_modality[modality].append(converted_column_name)
                ## Simplify the column names of dataframe of dynamic.
                for time in self.TIMES:
                    df_dict_of_dynamic_temp[time][modality] = df_dict_of_dynamic_temp[time][modality].rename(mapper = converter_column_name, axis = 1) ## Be careful for the scope of modality, this is hard to debug if bug.
            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(self.input_to_features_map["RNN_vectors"])
            self.input_to_features_map["RNN_vectors"].append("RE_DATE") ## Decoder label = RNN label - "RE_DATE".

            ## --- Set the dummy nan rows for consistent number of rows = number of patients.
            self.num_participants_in_dirty_dataset = len(self.df_dict["snp"])
            ## --- For dynamic.
            ## Makes concatenated dataframe with consistent length for all dynamic modalities.
            self.df_dict["dynamic"] = {}
            for time in self.TIMES:
                concatenated_dynamic_dfs = []
                for modality in self.dynamic_modalities_sequence:
                    df = df_dict_of_dynamic_temp[time][modality]
                    df_dict_of_dynamic_temp[time][modality] = pd.concat([df, pd.DataFrame([[np.nan for i in range(df.shape[1])] for j in range(self.num_participants_in_dirty_dataset - len(df))], columns=df.columns)], ignore_index=True) ## add i column and j row of nans.
                    concatenated_dynamic_dfs.append(df_dict_of_dynamic_temp[time][modality])
                self.df_dict["dynamic"][time] = pd.concat(concatenated_dynamic_dfs, axis = 1) ## Combine all dynamic modalities, the sequence follows self.dynamic_modalities_sequence.
            ## --- For static, because actual static data is sliced with column names, the order of columns in static dataframe does not affect the order of actual input of model.
            self.df_dict["static"] = pd.concat([self.df_dict["static"], pd.DataFrame([[np.nan for i in range(self.df_dict["static"].shape[1])] for j in range(self.num_participants_in_dirty_dataset - len(self.df_dict["static"]))], columns=self.df_dict["static"].columns)])
            self.df_dict["static"] = pd.concat([self.df_dict["snp"], self.df_dict["static"]], axis = 1) ## static feature sequence : [SNP, other features], combine all static modalities.

            ## Get patient's valid (intersection of two list of indices) indices from SNP/static features and Target Label.
            ## Example code : df_dict_of_dynamic_temp["M12"]["FS"][~df_dict_of_dynamic_temp["M12"]["FS"].isna().any(axis = 1)].index.tolist()
            self.valid_indices = {}
            self.valid_indices["static"]= self.df_dict["static"][~self.df_dict["static"].isna().any(axis = 1)].index.tolist() ## ~ means 'not'.
            self.valid_indices["outcome"]= self.df_dict["outcome"][~self.df_dict["outcome"].isna()].index.tolist()
            self.valid_indices["intersect"] = []
            for static_valid_index in self.valid_indices["static"]:
                if static_valid_index in self.valid_indices["outcome"]: self.valid_indices["intersect"].append(static_valid_index) ## static_valid_index is included in snp and outcome both.
            self.min_max_scale()
            self.set_observed_numbers_pool_for_each_group_of_features()
        else:
            raise Exception(NotImplementedError)

    def __compute_future_congestion(self):
        """Computes future traffic congestion for all samples and stores in a new column"""
        if self.dataset_kind != 'colorado_traffic':
            raise NotImplementedError('Only implemented for colorado_traffic dataset.')

        datetimes = pd.to_datetime(self.dataframe[['year', 'month', 'day', 'hour']])
        station_id = self.dataframe['count_station_id']

        future_counts = []
        future_congestion = []
        for date, id_ in zip(datetimes, station_id):
            future_datetime = date + timedelta(hours=self.future_congestion_lookahead_hrs)

            # Find row of dataframe containing sample with the future date and time.
            sample = self.dataframe[(self.dataframe['datetime'] == future_datetime)
                                    & (self.dataframe['count_station_id'] == id_)]

            if len(sample) == 0:
                # Then no sample has a time stamp matching future_datetime
                future_counts.append(None)
                future_congestion.append(None)
                continue

            # Compute mean and std of traffic counts for this counting station on the same hour and weekday.
            weekday_hour_counts = self.dataframe[
                           (self.dataframe['count_station_id'] == id_)
                           & (self.dataframe['hour'] == future_datetime.hour)
                           & (self.dataframe['day_of_week'] == sample['day_of_week'].iloc[0])
            ]['traffic_count']

            # Compute the congestion score as z-score.
            mean_count = weekday_hour_counts.mean()
            std_count = weekday_hour_counts.std()
            if std_count != 0:
                future_counts.append(sample['traffic_count'].iloc[0])
                future_congestion.append((future_counts[-1]-mean_count) / std_count)
            else:
                future_counts.append(None)
                future_congestion.append(None)
                continue

        self.dataframe['future_count'] = future_counts
        self.dataframe['future_congestion'] = future_congestion

    def __compute_near_crash(self):
        """Labels each data sample as being near a crash or not using self."""
        if self.dataset_kind != 'colorado_traffic':
            raise NotImplementedError('Only implemented for colorado_traffic dataset.')

        def is_close(lat1, lon1, lat2, lon2, threshold=1):  # threshold in miles
            """Checks if two pairs of lat/lon coordinates are within threshold miles.
            (assumes spherical earth)"""

            # Convert latitude and longitude from degrees to radians
            lat1_rad,  lat2_rad, lon1_rad, lon2_rad = (math.radians(lat1), math.radians(lat2),
                                                       math.radians(lon1), math.radians(lon2))

            dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad  # Difference in coordinates

            # Compute distance using Haversine formula (3956.0 = radius of earth in miles)
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
            distance = 3956.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            return distance <= threshold

        def is_temporally_close(datetime1, datetime2, range_hours=1):
            """Checks if two datetimes are within range_hours hours"""
            # datetime1 = datetime.strptime(datetime1, '%Y-%m-%d %H:%M:%S')
            datetime2 = datetime.strptime(datetime2, '%Y-%m-%d %H:%M:%S')  # Convert string to timestamp
            if range_hours >= 0:
                return datetime1 <= datetime2 and datetime1 + timedelta(hours=range_hours) >= datetime2
            else:
                return datetime1 >= datetime2 and datetime1 + timedelta(hours=range_hours) <= datetime2

        # Load accident dataset.
        accidents_df = pd.read_csv(os.path.join(self.path_to_dataset, 'CDOTRM_CD_Crash_Listing_Full.csv'))

        # Iterate over the traffic DataFrame and check for each record
        for index, traffic_row in self.dataframe.iterrows():
            accidents_on_day = accidents_df[
                ((accidents_df['year'] == traffic_row['year'])
                 & (accidents_df['month'] == traffic_row['month'])
                 & (accidents_df['day'] == traffic_row['day']))
            ]

            is_close_ = accidents_on_day[['lat', 'lon']].apply(
                lambda x: is_close(traffic_row['lat'], traffic_row['lon'], *x, threshold=self.crash_spatial_thresh),
                axis=1)

            is_temporally_close_ = accidents_on_day[['datetime']].apply(
                lambda x: is_temporally_close(traffic_row['datetime'], x[0], range_hours=self.crash_temporal_thresh),
                axis=1)

            if any(is_close_ & is_temporally_close_):
                # Then crash is close spatially and temporally.
                self.dataframe.at[index, 'near_crash'] = 1.0

        # Fill NaN values with False (no nearby accident)
        self.dataframe['near_crash'].fillna(0.0, inplace=True)

    def scale_times(self):
        """Convert datatime into seconds of floating numbers. The absolute RE_DATE time is changed to the relative time elapsed from the admission time"""

        for column_name in ["RE_DATE", "Admission time", 'Discharge time']:
            self.dataframe[column_name] = self.dataframe[column_name].apply(lambda x : x.timestamp() if not pd.isnull(x) else None)
        self.dataframe["RE_DATE"] = self.dataframe["RE_DATE"] - self.dataframe["Admission time"] ## Use RE_DATE as relative time elapsed after admission.
    
    def min_max_scale(self):
        """Scale the data of dataframes range of each feature into [0, 1]"""
        
        self.feature_min_max_dict = {}
        if self.dataset_kind == "covid":
            for column in self.dataframe.columns:
                if column not in ['PATIENT_ID', "gender", "outcome"]:
                    min_ = self.dataframe[column].min()
                    max_ = self.dataframe[column].max()
                    assert(max_ >= min_)
                    range_ = max_ - min_ if max_ > min_ else 1e-16
                    self.dataframe[column] = (self.dataframe[column] - min_) / range_
                    self.feature_min_max_dict[column] = dict(max = max_, min = min_)
        
        elif self.dataset_kind == "chestxray":
            for column in self.dataframe.columns:
                if column not in ["filename", "patientid"]:
                    min_ = self.dataframe[column].min()
                    max_ = self.dataframe[column].max()
                    assert(max_ >= min_)
                    range_ = max_ - min_ if max_ > min_ else 1e-16
                    self.dataframe[column] = (self.dataframe[column] - min_) / range_
                    self.feature_min_max_dict[column] = dict(max = max_, min = min_)

        elif self.dataset_kind == "traffic":
            path_to_satellite_images = os.path.join(self.path_to_dataset, "austin_satellite_images", "content",
                                                    "images")
            features_range = {}
            for numer_feature in ["Latitude", "Longitude", "Speed_Limit_WE", "Speed_Limit_NS",
                                  "minutes_scaled"] + self.target_columns:
                self.dataframe[numer_feature + "_raw"] = self.dataframe[numer_feature]
                numer_feature_column = self.dataframe[numer_feature]
                features_range[numer_feature] = {}
                features_range[numer_feature]["max-min"], features_range[numer_feature]["min"] = max(
                    numer_feature_column.max() - numer_feature_column.min(), 1e-8), numer_feature_column.min()
                self.dataframe[numer_feature] = (numer_feature_column - features_range[numer_feature]["min"]) / \
                                                features_range[numer_feature]["max-min"]

        elif self.dataset_kind == "colorado_traffic":
            # Scale all numerical features. Also scale congestion score.
            for feat in [*self.numerical_columns, 'future_congestion']:
                if feat.endswith("_raw"): continue
                min_ = pd.to_numeric(self.dataframe[feat].dropna().min())
                max_ = pd.to_numeric(self.dataframe[feat].dropna().max())
                assert (max_ >= min_)

                # Apply min-max scaling
                self.dataframe[feat] = (pd.to_numeric(self.dataframe[feat]) - min_) / max((max_ - min_), 1e-8)
                self.feature_min_max_dict[feat] = dict(max=max_, min=min_)  # Add computed min-max to dict.

        elif self.dataset_kind == "challenge":
            for dataframe in [self.dataframe_static, self.dataframe_dynamic]:
                for column in dataframe.columns:
                    if column not in ["In-hospital_death", "Gender", "CCU", "CSRU", "SICU", "MechVent"]:
                        min_ = dataframe[column].dropna().min()
                        max_ = dataframe[column].dropna().max()
                        assert(max_ >= min_)
                        range_ = max_ - min_
                        dataframe[column] = (dataframe[column] - min_) / range_
                        self.feature_min_max_dict[column] = dict(max = max_, min = min_)

        elif self.dataset_kind == "alz":
            for static_feature_name in self.df_dict["static"].columns: ## For static,
                self.feature_min_max_dict[static_feature_name] = {"max": self.df_dict["static"][static_feature_name].max(), "min": self.df_dict["static"][static_feature_name].min()}
                if self.feature_min_max_dict[static_feature_name]["max"] != self.feature_min_max_dict[static_feature_name]["min"]:
                    self.df_dict["static"][static_feature_name] = (self.df_dict["static"][static_feature_name] - self.feature_min_max_dict[static_feature_name]["min"]) / (self.feature_min_max_dict[static_feature_name]["max"] - self.feature_min_max_dict[static_feature_name]["min"])
            for modality in self.dynamic_modalities_sequence: ## For dynamic,
                for feature_name in self.dynamic_feature_names_for_each_modality[modality]: ## min, max for each dynamic feature.
                    assert(feature_name not in self.feature_min_max_dict.keys())
                    max_, min_ = -1e+30, +1e+30
                    for time in self.TIMES: ## min, max across all the time points.
                        if self.df_dict["dynamic"][time][feature_name].max() > max_:
                            max_ = self.df_dict["dynamic"][time][feature_name].max()
                        if self.df_dict["dynamic"][time][feature_name].min() < min_:
                            min_ = self.df_dict["dynamic"][time][feature_name].min()
                    self.feature_min_max_dict[feature_name] = {"max": max_, "min": min_}
                    for time in self.TIMES:
                        if max_ > min_: self.df_dict["dynamic"][time][feature_name] = (self.df_dict["dynamic"][time][feature_name] - min_) / (max_ - min_)
        else:
            raise Exception(NotImplementedError)
    
    def set_observed_numbers_pool_for_each_group_of_features(self, excluded_features = None):
        """Prepare the group of features used to plot feature importance, "input" means which input where that feature exists. Each group is plotted separately.

        Set the groups_of_features_info.
        
        Attributes
        ----------
        groups_of_features_info : dict
            For example, {group_1: {feature_1: {idx: 3, observed_numbers: [1.3, 5.3, ...], mean: 3.5, std: 1.4}, feature_2: {...}, ...}, ...}
        """

        if excluded_features is None: excluded_features = []
        self.groups_of_features_info = {}
        self.feature_to_input_map = {}
        for input_ in ["RNN_vectors", "static_encoder", "raw"]: ## "dynamic_vectors_decoder" has no feature importance.
            for feature in self.input_to_features_map[input_]:
                # assert(feature not in self.feature_to_input_map.keys()) ## Avoid duplication, but dynamic decoder/RNN can be duplicated, so comment out.
                self.feature_to_input_map[feature] = input_
        
        if self.dataset_kind in ["covid", "chestxray", "colorado_traffic"]: ## dynamic and all groups.
            if self.dataset_kind in ["covid", "chestxray"]: self.groups_of_features_info["all"] = {}
            elif self.dataset_kind in ["colorado_traffic"]: self.groups_of_features_info["static"] = {}
            self.groups_of_features_info["dynamic"] = {}
            for input_ in ["raw", "RNN_vectors", "static_encoder"]:
                for feature in self.input_to_features_map[input_]:
                    if self.dataset_kind in ["covid", "chestxray"]: self.groups_of_features_info["all"][feature] = dict(idx= self.input_to_features_map[input_].index(feature), input= input_)
                    elif self.dataset_kind in ["colorado_traffic"] and feature in self.static_column_names: self.groups_of_features_info["static"][feature] = dict(idx= self.input_to_features_map[input_].index(feature), input= input_)
                    if feature not in self.static_column_names: self.groups_of_features_info["dynamic"][feature] = dict(idx= self.input_to_features_map[input_].index(feature), input= input_)

            for feature_group in self.groups_of_features_info.keys():
                for feature_name in self.groups_of_features_info[feature_group].keys():
                    self.groups_of_features_info[feature_group][feature_name]["observed_numbers"] = list(self.dataframe[feature_name].dropna())
                    self.groups_of_features_info[feature_group][feature_name]["mean"] = np.mean(self.groups_of_features_info[feature_group][feature_name]["observed_numbers"])
                    self.groups_of_features_info[feature_group][feature_name]["std"] = np.std(self.groups_of_features_info[feature_group][feature_name]["observed_numbers"])            

        elif self.dataset_kind in ["challenge"]:
            self.groups_of_features_info["static"] = {}
            self.groups_of_features_info["dynamic"] = {}

            for input_ in ["raw", "static_encoder"]:
                for feature in self.input_to_features_map[input_]:
                    if feature not in excluded_features: self.groups_of_features_info["static"][feature] = dict(idx= self.input_to_features_map[input_].index(feature), input= input_)
            for feature in self.input_to_features_map["RNN_vectors"]:
                if feature not in excluded_features: self.groups_of_features_info["dynamic"][feature] = dict(idx= self.input_to_features_map["RNN_vectors"].index(feature), input= "RNN_vectors")

            for feature_group, dataframe in zip(["static", "dynamic"], [self.dataframe_static, self.dataframe_dynamic]):
                for feature_name in self.groups_of_features_info[feature_group].keys():
                    if feature_name not in excluded_features:
                        self.groups_of_features_info[feature_group][feature_name]["observed_numbers"] = list(dataframe[feature_name].dropna())
                        self.groups_of_features_info[feature_group][feature_name]["mean"] = np.mean(self.groups_of_features_info[feature_group][feature_name]["observed_numbers"])
                        self.groups_of_features_info[feature_group][feature_name]["std"] = np.std(self.groups_of_features_info[feature_group][feature_name]["observed_numbers"])
        
        elif self.dataset_kind == "alz":
            ## Prepare result of the group of features used to plot feature importance, "input" (means where that feature exists in the input) can be different with self.separate_dynamic_static_features. Each group is plotted separately.
            self.groups_of_features_info = {
                "FS": {name: {"input": self.feature_to_input_map[name], "idx": self.input_to_features_map[self.feature_to_input_map[name]].index(name), "observed_numbers": [], "mean": None, "std": None} for name in self.dynamic_feature_names_for_each_modality["FS"]}, 
                "VBM": {name: {"input": self.feature_to_input_map[name], "idx": self.input_to_features_map[self.feature_to_input_map[name]].index(name), "observed_numbers": [], "mean": None, "std": None} for name in self.dynamic_feature_names_for_each_modality["VBM"]}}
            ## Static
            self.groups_of_features_info["SNP"] = {name: {"input": self.feature_to_input_map[name], "idx": self.input_to_features_map[self.feature_to_input_map[name]].index(name), "observed_numbers": self.df_dict["static"][name].dropna().tolist(), "mean": None, "std": None} for name in list(self.df_dict["snp"].columns)[1:]}
            ## Dynamic
            for time in self.TIMES: ## Collect observations across the time points.
                for modality in ["FS", "VBM"]:
                    for feature_name in self.dynamic_feature_names_for_each_modality[modality]:
                        self.groups_of_features_info[modality][feature_name]["observed_numbers"] += deepcopy(self.df_dict["dynamic"][time][feature_name].dropna().tolist())
            ## Calculate the mean, std for each group.
            for group in self.groups_of_features_info.keys():
                for feature_name in self.groups_of_features_info[group].keys():
                    self.groups_of_features_info[group][feature_name]["mean"] = np.mean(self.groups_of_features_info[group][feature_name]["observed_numbers"])
                    self.groups_of_features_info[group][feature_name]["std"] = np.std(self.groups_of_features_info[group][feature_name]["observed_numbers"])
        
        ## Delete RE_DATE from the feature, you may/may not want to include RE_DATE.
        for group in self.groups_of_features_info.keys():
            if "RE_DATE" in self.groups_of_features_info[group].keys():
                del self.groups_of_features_info[group]["RE_DATE"]

    def set_dicts(self, kwargs_toy_dataset = None, prediction_label_mapping = None, observation_density_threshold = 0.1):
        """Load the pandas dataframe datset into the dictionaries of Numpy arrays for Encoder or Decoder
        
        Attributes
        ----------
        observation_density_threshold : 0. < float < 1.
            The minimum threshold of observed entries divided by the total entries. The larger observation_density_threshold, result the more strict selection, thus the smaller number of records.
        """
        
        ### outcome -> prediction label, mapping.
        if prediction_label_mapping is None:
            ## Set prediction label length.
            if self.dataset_kind == "covid" or self.dataset_kind == "toy" or self.dataset_kind == "challenge" or self.dataset_kind == "chestxray": self.prediction_labels_bag = [0, 1]
            elif self.dataset_kind == "alz" : self.prediction_labels_bag = [1, 2, 3]
            elif self.dataset_kind == "traffic":
                self.prediction_labels_bag = list(range(self.labels_info["num_label_divisions"]))
                print(self.labels_info)
            elif self.dataset_kind == 'colorado_traffic':
                self.prediction_labels_bag = [0, 1]
            else: raise Exception(NotImplementedError)

            self.prediction_label_mapping = self.prediction_label_mapping_default_funct

        self.dicts = [] ## list of dictionaries where each dictionary for each patient.
        if self.dataset_kind == "covid": ## --- --- COVID-19 DATASET.
            for index, row in self.dataframe.iterrows():
                if pd.isnull(row["PATIENT_ID"]): ## Middle row of patient.
                    for column_name in self.static_column_names:
                        if column_name != 'PATIENT_ID':
                            assert(patient_dict[column_name] == row[column_name]) ## Static data should not change.
                    
                    for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                        patient_dict[input_]["data"].append(row[self.input_to_features_map[input_]].fillna(self.init_number_for_unobserved_entry).tolist())
                        patient_dict[input_]["mask"].append((1. - row[self.input_to_features_map[input_]].isna()).tolist())
                        patient_dict[input_]["concat"].append(deepcopy(patient_dict[input_]["data"][-1] + patient_dict[input_]["mask"][-1]))
                    patient_dict["RE_DATE"].append(row["RE_DATE"])
                    if (index == len(self.dataframe) - 1 or not pd.isnull(self.dataframe.iloc[index + 1]["PATIENT_ID"])) and not pd.isnull(patient_dict["RE_DATE"][0]): ## Final row of patient.
                        for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                            for data_type in ["data", "mask", "concat"]:
                                patient_dict[input_][data_type] = self.convert_to_tf_input([patient_dict[input_][data_type]]) ## Dummy dimension for time series.
                        for input_ in ["static_encoder", "raw"]:
                            for data_type in ["data", "mask", "concat"]:
                                patient_dict[input_][data_type] = self.convert_to_tf_input(patient_dict[input_][data_type]) ## There is no dummy dimension for static features.
                        for date_idx in range(len(patient_dict["RE_DATE"]) - 1): ## time stamp should be increasing.
                            assert(patient_dict["RE_DATE"][date_idx] <= patient_dict["RE_DATE"][date_idx + 1])
                        patient_dict["RE_DATE"] = self.convert_to_tf_input([[[RE_DATE] for RE_DATE in patient_dict["RE_DATE"]]])
                        patient_dict["predictor label"] = self.prediction_label_mapping(patient_dict["outcome"], inverse = False)
                        self.dicts.append(patient_dict)
                else: ## First row of patient.
                    patient_dict = {feature : row[feature] for feature in self.static_column_names}
                    for input_ in ["RNN_vectors", "dynamic_vectors_decoder", "static_encoder", "raw"]:
                        patient_dict[input_] = {}
                        patient_dict[input_]["data"] = [row[self.input_to_features_map[input_]].fillna(self.init_number_for_unobserved_entry).tolist()]
                        patient_dict[input_]["mask"] = [(1. - row[self.input_to_features_map[input_]].isna()).tolist()]
                        patient_dict[input_]["concat"] = [deepcopy(patient_dict[input_]["data"][0] + patient_dict[input_]["mask"][0])]
                    patient_dict["RE_DATE"] = [row["RE_DATE"]]
        
        elif self.dataset_kind == "challenge": ## --- --- physionet-challenge DATASET.
            self.dicts = {} ## Later, we will convert dict to list.
            for index, row in self.dataframe_static.iterrows(): ## static
                patient_dict = {feature : row[feature] for feature in self.static_column_names}
                patient_dict["outcome"] = patient_dict["In-hospital_death"]
                for input_ in ["static_encoder", "raw"]: ## Note that "RNN_vectors", "dynamic_vectors_decoder", are deleted, so feeding static data to LSTM will not be supported.
                    patient_dict[input_] = {}
                    patient_dict[input_]["data"] = [row[self.input_to_features_map[input_]].fillna(self.init_number_for_unobserved_entry).tolist()]
                    patient_dict[input_]["mask"] = [(1. - row[self.input_to_features_map[input_]].isna()).tolist()]
                    patient_dict[input_]["concat"] = [deepcopy(patient_dict[input_]["data"][0] + patient_dict[input_]["mask"][0])]
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    patient_dict[input_] = {}
                    patient_dict[input_]["data"] = []
                    patient_dict[input_]["mask"] = []
                    patient_dict[input_]["concat"] = []
                patient_dict["RE_DATE"] = []
                self.dicts[row["recordid"]] = patient_dict
            
            for index, row in self.dataframe_dynamic.iterrows(): ## dynamic
                recordid = row["recordid"]
                assert(row["recordid"] in self.dicts.keys())
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    self.dicts[recordid][input_]["data"].append(row[self.input_to_features_map[input_]].fillna(self.init_number_for_unobserved_entry).tolist())
                    self.dicts[recordid][input_]["mask"].append((1. - row[self.input_to_features_map[input_]].isna()).tolist())
                    self.dicts[recordid][input_]["concat"].append(deepcopy(self.dicts[recordid][input_]["data"][-1] + self.dicts[recordid][input_]["mask"][-1]))
                self.dicts[recordid]["RE_DATE"].append(row["time"])

            self.dicts = {key: val for key, val in self.dicts.items() if len(val["RE_DATE"]) > 0} ## remove participant with zero dynamic record.

            for recordid in self.dicts.keys(): ## finalize/clean up records.
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    for data_type in ["data", "mask", "concat"]:
                        self.dicts[recordid][input_][data_type] = self.convert_to_tf_input([self.dicts[recordid][input_][data_type]]) ## Dummy dimension for time series.
                    for input_ in ["static_encoder", "raw"]:
                        for data_type in ["data", "mask", "concat"]:
                            self.dicts[recordid][input_][data_type] = self.convert_to_tf_input(self.dicts[recordid][input_][data_type]) ## There is no dummy dimension for static features.
                for date_idx in range(len(self.dicts[recordid]["RE_DATE"]) - 1): ## time stamp should be increasing.
                    assert(self.dicts[recordid]["RE_DATE"][date_idx] <= self.dicts[recordid]["RE_DATE"][date_idx + 1])
                self.dicts[recordid]["RE_DATE"] = self.convert_to_tf_input([[[RE_DATE] for RE_DATE in self.dicts[recordid]["RE_DATE"]]])
                self.dicts[recordid]["predictor label"] = self.prediction_label_mapping(self.dicts[recordid]["In-hospital_death"], inverse = False)
            
            self.dicts = list(self.dicts.values())
        
        elif self.dataset_kind == "chestxray": ## --- --- chestxray dataset.
            self.dicts = {} ## Later, we will convert dict to list.
            self.img_shapes = []
            def if_good_sample(patient_dict):
                if not (patient_dict is not None and str(patient_dict["patientid"]) not in ["334", "323"] and len(patient_dict["RE_DATE"]) > 0 and len(patient_dict["RNN_2D"]["data"]) > 0):
                    return False
                else:
                    if self.shape_2D_records is not None:
                        return True
                    size = 1
                    for img_arr in patient_dict["RNN_2D"]["data"]:
                        # for axis in (0, 1):
                        #     size = size * img_arr.shape[axis]
                        num_entries = img_arr.shape[0] * img_arr.shape[1]
                        if 500 * 500 > num_entries or 1000 * 1000 < num_entries:
                            return False
                    # size /= len(patient_dict["RNN_2D"]["data"])
                    # return size < 2000 * 2000 * 80 ## 2000 * 2000 is maximum size of image in this dataset.
                    return True
            print("Loading static data of chestxray dataset.")

            # group_df = self.dataframe.iloc[:100].groupby(["patientid"])
            group_df = self.dataframe.groupby(["patientid"])
            for name, group in tqdm(group_df): ## name == row["patientid"]
                patientid = name
                # for index, row in tqdm(self.dataframe.iterrows()): ## static
                for index, row in group.iterrows(): ## static, max: 950 samples.
                    # self.dicts.append(None)
                    if not row["filename"].endswith("nii.gz"):
                        # patientid = index
                        img_arr, mask_arr = get_arr_of_image_from_path(path_to_img = os.path.join(self.path_to_dataset, "images", row["filename"]), shape = self.shape_2D_records, nopostprocess = False, apply_segment = self.apply_segment)
                        self.img_shapes.append(img_arr.shape)
                        if patientid not in self.dicts.keys(): ## first time.
                            self.dicts[patientid] = {feature : row[feature] for feature in self.static_column_names}
                            patient_dict = self.dicts[patientid] ## shallow copy.
                            patient_dict["outcome"] = patient_dict["survival_Y"]
                            for input_ in ["static_encoder", "raw", "RNN_vectors", "dynamic_vectors_decoder"]:
                                patient_dict[input_] = {}
                                patient_dict[input_]["data"] = [row[self.input_to_features_map[input_]].fillna(self.init_number_for_unobserved_entry).tolist()]
                                patient_dict[input_]["mask"] = [(1. - row[self.input_to_features_map[input_]].isna()).tolist()]
                                patient_dict[input_]["concat"] = [deepcopy(patient_dict[input_]["data"][0] + patient_dict[input_]["mask"][0])]
                            patient_dict["RE_DATE"] = [self.init_number_for_unobserved_entry if np.isnan(row["offset"]) else row["offset"]]
                            ## 2D image input
                            patient_dict["RNN_2D"] = {}
                            patient_dict["RNN_2D"]["data"] = [img_arr]
                            if not self.if_separate_img:
                                patient_dict["RNN_2D"]["mask"] = [mask_arr]
                                patient_dict["RNN_2D"]["concat"] = [np.concatenate([img_arr, mask_arr])]
                        else:
                            # for feature in self.static_column_names:
                            #     if feature not in ["filename", "age"]: assert(row[feature] == self.dicts[patientid][feature]) ## static data should not change.
                            patient_dict = self.dicts[patientid] ## shallow copy.
                            for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]: ## dynamic data
                                patient_dict[input_]["data"].append(row[self.input_to_features_map[input_]].fillna(self.init_number_for_unobserved_entry).tolist())
                                patient_dict[input_]["mask"].append((1. - row[self.input_to_features_map[input_]].isna()).tolist())
                                patient_dict[input_]["concat"].append(deepcopy(patient_dict[input_]["data"][-1] + patient_dict[input_]["mask"][-1]))
                            patient_dict["RE_DATE"].append(self.init_number_for_unobserved_entry if np.isnan(row["offset"]) else row["offset"])
                            ## 2D image input
                            patient_dict["RNN_2D"]["data"].append(img_arr)
                            if not self.if_separate_img:
                                patient_dict["RNN_2D"]["mask"].append(mask_arr)
                                patient_dict["RNN_2D"]["concat"].append(np.concatenate([img_arr, mask_arr]))
                if name in self.dicts.keys() and if_good_sample(self.dicts[name]):
                    for input_ in ["RNN_vectors", "dynamic_vectors_decoder", "RNN_2D"]:
                        for data_type in ["data", "mask", "concat"] if (not self.if_separate_img or input_ in ["RNN_vectors", "dynamic_vectors_decoder"]) else ["data"]:
                            arr = self.dicts[patientid][input_][data_type]
                            if input_ == "RNN_2D" and self.shape_2D_records is None:
                                arr, slice_idx = self.convert_to_tf_input([arr], if_var_shape= True) ## Dummy dimension for time series.
                                if "slice_idx" not in self.dicts[patientid][input_].keys():
                                    self.dicts[patientid][input_]["slice_idx"] = {data_type: slice_idx}
                                else:
                                    self.dicts[patientid][input_]["slice_idx"][data_type] = slice_idx
                            else:
                                arr = self.convert_to_tf_input([arr], if_var_shape= False) ## Dummy dimension for time series.
                            if input_ == "RNN_2D" and self.if_separate_img:
                                with open(f"{self.obj_dir_path}/{patientid}_{input_}_{data_type}.npy", 'wb') as file_name:
                                    np.save(file_name, arr)
                                del self.dicts[patientid][input_][data_type]
                            else:
                                self.dicts[patientid][input_][data_type] = arr
                    for input_ in ["static_encoder", "raw"]:
                        for data_type in ["data", "mask", "concat"]:
                            self.dicts[patientid][input_][data_type] = self.convert_to_tf_input(self.dicts[patientid][input_][data_type]) ## There is no dummy dimension for static features.
                    # for date_idx in range(len(self.dicts[patientid]["RE_DATE"]) - 1): ## time stamp should be increasing.
                    #     assert(self.dicts[patientid]["RE_DATE"][date_idx] <= self.dicts[patientid]["RE_DATE"][date_idx + 1])
                    self.dicts[patientid]["RE_DATE"] = self.convert_to_tf_input([[[RE_DATE] for RE_DATE in self.dicts[patientid]["RE_DATE"]]])
                    self.dicts[patientid]["predictor label"] = self.prediction_label_mapping(self.dicts[patientid]["outcome"], inverse = False)
                elif name in self.dicts.keys(): ## means not if_good_sample
                    del self.dicts[name]

            # self.dicts = {key: val for key, val in self.dicts.items() if len(val["RE_DATE"]) > 0 and len(val["RNN_2D"]["data"]) > 0} ## remove participant with zero dynamic record.
            # self.dicts = {key: val for key, val in self.dicts.items() if if_good_sample(key)} ## remove participant with zero dynamic record.
            self.dicts = [self.dicts[idx] for idx in self.dicts.keys()] ## remove participant with zero dynamic record and too large size.
            
            # self.dicts = list(self.dicts.values())

        elif self.dataset_kind == "traffic":  ## --- --- chestxray dataset.
            self.dicts = {}  ## Later, we will convert dict to list.
            self.img_shapes = []
            print("Loading static data of traffic dataset.")

            # group_df = self.dataframe.iloc[:100].groupby(["patientid"])
            group_df = self.dataframe.sort_values(['location_id', 'minutes'], ascending=True).groupby('location_id')
            locationid_to_size = dict(self.dataframe.groupby('location_id').size())
            locationid_to_sample_idx = {int(key): 0 for key in
                                        locationid_to_size.keys()}  ## accumulated index inside df group.
            locationid_to_satellite = {}
            path_to_satellite_images = os.path.join(self.path_to_dataset, 'austin_satellite_images', 'content',
                                                    'images')
            for filename in os.listdir(path_to_satellite_images):
                if filename.endswith(".jpg"):
                    locationid = int(filename.split(".")[0])
                    locationid_to_satellite[locationid] = cv2.resize(
                        imageio.imread(f'{path_to_satellite_images}/{filename}'), dsize=(101, 101))
                    self.shape_2D_record_static = locationid_to_satellite[locationid].shape

            num_records = self.num_records
            current_num_bags = 0
            while (current_num_bags < self.num_bags_max):
                if self.num_bags_max > 20 and current_num_bags % int(self.num_bags_max / 20) == 0:
                    print(f"{current_num_bags} bags are processed.")
                if all([value == -1 for value in locationid_to_sample_idx.values()]):
                    print(
                        f"Requested number of bags is {self.num_bags_max} but there are no more available bags to generate, so we end here.")
                    break
                for locationid, df_group in group_df:  ## bag level
                    locationid = int(locationid)
                    if locationid_to_sample_idx[locationid] != -1:
                        if locationid_to_sample_idx[locationid] + num_records > locationid_to_size[locationid]:
                            locationid_to_sample_idx[locationid] = -1
                            break
                        else:
                            prediction_target_idx = None
                            for idx in range(locationid_to_sample_idx[locationid] + num_records,
                                             locationid_to_size[locationid]):
                                if df_group.iloc[idx]['minutes'] > \
                                        df_group.iloc[locationid_to_sample_idx[locationid] + num_records][
                                            'minutes'] + self.prediction_interval:
                                    prediction_target_idx = idx
                            if prediction_target_idx is None:
                                locationid_to_sample_idx[locationid] = -1
                                break
                            # target_labels["data_arr"].append([[df_group.iloc[prediction_target_idx]['minutes']]])            
                        # static_data.append(np.concatenate((df_group.iloc[0][static_features_encoded].to_numpy(), locationid_to_satellite[str(int(locationid))]), axis= 0))
                        patientid = current_num_bags
                        self.dicts[patientid] = {feature: df_group.iloc[0][feature] for feature in
                                                 self.static_column_names}
                        patient_dict = self.dicts[patientid]  ## shallow copy.
                        patient_dict["patientid"] = patientid

                        img_arr = locationid_to_satellite[locationid]
                        mask_arr = np.ones(img_arr.shape)
                        patient_dict["static_2D"] = {}
                        patient_dict["static_2D"]["data"] = np.array([img_arr])
                        if not self.if_separate_img:
                            patient_dict["static_2D"]["mask"] = np.array([mask_arr])
                            patient_dict["static_2D"]["concat"] = np.array([np.concatenate([img_arr, mask_arr])])
                        patient_dict["predictor label"] = self.prediction_label_mapping(
                            np.mean(df_group.iloc[0][self.target_columns].to_numpy()), inverse=False)
                        patient_dict["outcome"] = self.prediction_label_mapping(patient_dict["predictor label"],
                                                                                inverse=True)

                        records_indices = list(range(locationid_to_sample_idx[locationid],
                                                     locationid_to_sample_idx[locationid] + num_records))
                        for input_ in ["static_encoder", "raw"]:
                            patient_dict[input_] = {}
                            row = df_group.iloc[records_indices[0]]
                            patient_dict[input_]["data"] = row[self.input_to_features_map[input_]].fillna(
                                self.init_number_for_unobserved_entry).tolist()
                            patient_dict[input_]["mask"] = (
                                        1. - row[self.input_to_features_map[input_]].isna()).tolist()
                            if isinstance(patient_dict[input_]["data"], list):
                                patient_dict[input_]["concat"] = deepcopy(
                                    patient_dict[input_]["data"] + patient_dict[input_]["mask"])
                            elif isinstance(patient_dict[input_]["data"], np.ndarray):
                                patient_dict[input_]["concat"] = np.concatenate(
                                    [patient_dict[input_]["data"], patient_dict[input_]["mask"]], axis=-1)
                            else:
                                raise Exception(NotImplementedError)
                        for input_ in ["RNN_vectors", "dynamic_vectors_decoder", "RNN_2D"]:
                            patient_dict[input_] = {}
                            for dtype in ["data", "mask", "concat"]:
                                patient_dict[input_][dtype] = []
                        patient_dict["RE_DATE"] = []
                        patient_dict["RNN_2D"]["file_name"] = []

                        for record_idx in records_indices:  ## instance level
                            row = df_group.iloc[record_idx]
                            for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                                patient_dict[input_]["data"].append(row[self.input_to_features_map[input_]].fillna(
                                    self.init_number_for_unobserved_entry).tolist())
                                patient_dict[input_]["mask"].append(
                                    (1. - row[self.input_to_features_map[input_]].isna()).tolist())
                                if isinstance(patient_dict[input_]["data"][-1], list):
                                    patient_dict[input_]["concat"].append(
                                        deepcopy(patient_dict[input_]["data"][-1] + patient_dict[input_]["mask"][-1]))
                                elif isinstance(patient_dict[input_]["data"][-1], np.ndarray):
                                    patient_dict[input_]["concat"].append(np.concatenate(
                                        [patient_dict[input_]["data"][-1], patient_dict[input_]["mask"][-1]], axis=-1))
                                else:
                                    raise Exception(NotImplementedError)
                            patient_dict["RE_DATE"].append(
                                self.init_number_for_unobserved_entry if np.isnan(row["minutes_scaled"]) else row[
                                    "minutes_scaled"])
                            ## 2D image input

                            file_name = f"{os.path.join(self.path_to_dataset, 'Austin Images')}/{row['image_0']}"
                            patient_dict["RNN_2D"]["file_name"].append(file_name)
                            img_arr, mask_arr = get_arr_of_image_from_path(path_to_img=file_name,
                                                                           shape=self.shape_2D_records,
                                                                           nopostprocess=False, apply_segment=False)
                            # img_arr = imageio.imread(f"{os.path.join(self.path_to_dataset, 'Austin Images')}/{row['image_0']}")
                            # mask_arr = np.ones(img_arr.shape)
                            self.img_shapes.append(img_arr.shape)
                            patient_dict["RNN_2D"]["data"].append(img_arr)
                            if not self.if_separate_img:
                                patient_dict["RNN_2D"]["mask"].append(mask_arr)
                                patient_dict["RNN_2D"]["concat"].append(np.concatenate([img_arr, mask_arr]))

                        # ## Set prediction target label
                        # row = df_group.iloc[prediction_target_idx]
                        # patient_dict["outcome"] = np.mean(row[self.target_columns].to_numpy())

                        ## Save image and set data type.
                        for input_ in ["RNN_vectors", "dynamic_vectors_decoder", "RNN_2D"]:
                            for data_type in ["data", "mask", "concat"] if (
                                    not self.if_separate_img or input_ in ["RNN_vectors",
                                                                           "dynamic_vectors_decoder"]) else ["data"]:
                                arr = self.dicts[patientid][input_][data_type]
                                if input_ == "RNN_2D" and self.shape_2D_records is None:
                                    arr, slice_idx = self.convert_to_tf_input([arr],
                                                                              if_var_shape=True)  ## Dummy dimension for time series.
                                    if "slice_idx" not in self.dicts[patientid][input_].keys():
                                        self.dicts[patientid][input_]["slice_idx"] = {data_type: slice_idx}
                                    else:
                                        self.dicts[patientid][input_]["slice_idx"][data_type] = slice_idx
                                else:
                                    arr = self.convert_to_tf_input([arr],
                                                                   if_var_shape=False)  ## Dummy dimension for time series.
                                if input_ == "RNN_2D" and self.if_separate_img:
                                    with open(f"{self.obj_dir_path}/{patientid}_{input_}_{data_type}.npy",
                                              'wb') as file_name:
                                        np.save(file_name, arr)
                                    del self.dicts[patientid][input_][data_type]
                                else:
                                    self.dicts[patientid][input_][data_type] = arr
                        for input_ in ["static_encoder", "raw"]:
                            for data_type in ["data", "mask", "concat"]:
                                self.dicts[patientid][input_][data_type] = self.convert_to_tf_input(
                                    [self.dicts[patientid][input_][data_type]])
                        # for date_idx in range(len(self.dicts[patientid]["RE_DATE"]) - 1): ## time stamp should be increasing.
                        #     assert(self.dicts[patientid]["RE_DATE"][date_idx] <= self.dicts[patientid]["RE_DATE"][date_idx + 1])
                        self.dicts[patientid]["RE_DATE"] = self.convert_to_tf_input(
                            [[[RE_DATE] for RE_DATE in self.dicts[patientid]["RE_DATE"]]])
                        # self.dicts[patientid]["predictor label"] = self.prediction_label_mapping(self.dicts[patientid]["outcome"], inverse = False)
                        locationid_to_sample_idx[locationid] += 1
                        current_num_bags += 1
                        if current_num_bags >= self.num_bags_max: break

            self.dicts = [self.dicts[idx] for idx in
                          self.dicts.keys()]  ## remove participant with zero dynamic record and too large size.

            # self.dicts = list(self.dicts.values())

        elif self.dataset_kind == "colorado_traffic":
            self.dicts = {}  # Later, we will convert dict to list.
            print("Loading static data of", self.dataset_kind, "dataset.")
            self.station_id_to_datetimes = {}

            group_df = self.dataframe.groupby('count_station_id')
            count_stationId_to_size = dict(group_df.size())
            count_stationId_to_sample_idx = {int(key): 0 for key in count_stationId_to_size.keys()}  # accumulated index inside df group.

            stationId_to_satellite = {}
            self.shape_2D_record_static = None

            current_num_bags = 0
            while current_num_bags < self.num_bags_max:
                if self.num_bags_max > 20 and current_num_bags % int(self.num_bags_max / 20) == 0:
                    print(f"{current_num_bags} bags are processed.")
                if all([value == -1 for value in count_stationId_to_sample_idx.values()]):
                    print(
                        f"Requested number of bags is {self.num_bags_max} but there are no more available bags to generate, so we end here.")
                    break
                for count_stationId, df_group in group_df:  # bag level
                    num_records = random.sample(self.num_records, k= 1)[0]
                    if count_stationId_to_sample_idx[count_stationId] != -1 and len(df_group) > max(self.num_records): # and not np.isnan(df_group.iloc[count_stationId_to_sample_idx[count_stationId] + num_records - 1][self.target_columns]).any()
                        if count_stationId not in stationId_to_satellite.keys():
                            image_OSM = cv2.cvtColor(utils.get_OSM_arr(latitude= df_group.iloc[0]["lat_raw"], longitude= df_group.iloc[0]["lon_raw"]), cv2.COLOR_BGR2GRAY)
                            image_OSM = np.transpose(np.array([image_OSM]), (1, 2, 0))
                            stationId_to_satellite[count_stationId] = image_OSM
                            if self.shape_2D_record_static is None:
                                self.shape_2D_record_static = image_OSM.shape
                            else:
                                assert(self.shape_2D_record_static == image_OSM.shape)
                        patientid = current_num_bags
                        self.dicts[patientid] = {feature: df_group.iloc[0][feature] for feature in self.static_column_names}
                        station_dict = self.dicts[patientid]  # shallow copy.
                        station_dict["patientid"] = patientid
                        station_dict["count_station_id"] = count_stationId
                        station_dict["lat_raw"] = df_group.iloc[0]["lat_raw"]
                        station_dict["lon_raw"] = df_group.iloc[0]["lon_raw"]

                        img_arr = stationId_to_satellite[count_stationId]
                        mask_arr = np.ones(img_arr.shape)
                        station_dict["static_2D"] = {}
                        station_dict["static_2D"]["data"] = np.array([img_arr])
                        if not self.if_separate_img:
                            station_dict["static_2D"]["mask"] = np.array([mask_arr])
                            station_dict["static_2D"]["concat"] = np.array([np.concatenate([img_arr, mask_arr])])
                        station_dict["predictor label"] = np.nan_to_num(np.array([df_group.iloc[count_stationId_to_sample_idx[count_stationId] + num_records - 1][self.target_columns]]), nan= self.init_number_for_unobserved_entry)
                        if self.classification_axis is None:
                            station_dict["outcome"] = int(station_dict["predictor label"][0, 0])
                        else:
                            station_dict["outcome"] = int(station_dict["predictor label"][0, self.classification_axis])
                        station_dict["label_tasks_mask"] = np.array([(1. - np.isnan(df_group.iloc[count_stationId_to_sample_idx[count_stationId] + num_records - 1][self.target_columns]))])
                        # station_dict["outcome"] = self.prediction_label_mapping(station_dict["predictor label"], inverse=True)
                        records_indices = list(range(count_stationId_to_sample_idx[count_stationId], count_stationId_to_sample_idx[count_stationId] + num_records))

                        for input_ in ["static_encoder", "raw"]:
                            station_dict[input_] = {"data": [], "mask": [], "concat": []}
                            static_row = df_group.iloc[0]

                            self.populate_data_mask_concat(dict_populated= station_dict[input_], row= static_row[self.input_to_features_map[input_]], if_append= False)

                        for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                            station_dict[input_] = {}
                            for dtype in ["data", "mask", "concat"]:
                                station_dict[input_][dtype] = []
                        station_dict["RE_DATE"] = []
                        station_dict["datetime"] = []

                        for idx in records_indices:  # Instance level.
                            row = df_group.iloc[idx]
                            station_dict = self.dicts[patientid] ## shallow copy.
                            for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]: ## dynamic data
                                self.populate_data_mask_concat(dict_populated= station_dict[input_], row= row[self.input_to_features_map[input_]], if_append= True)
                            station_dict["RE_DATE"].append(self.init_number_for_unobserved_entry if np.isnan(row["minutes_scaled"]) else row["minutes_scaled"])
                            station_dict["datetime"].append(datetime.utcfromtimestamp(row["datetime_timestamp"]))
                        station_dict["dataframe"] = {
                            "RNN_vectors": pd.DataFrame(data= station_dict["RNN_vectors"], columns= self.input_to_features_map["RNN_vectors"]),
                            "raw": pd.DataFrame(data= station_dict["raw"], columns= self.input_to_features_map["raw"]),
                        }

                        
                        ## Save image and set data type.
                        for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                            for data_type in ["data", "mask", "concat"] if (
                                    not self.if_separate_img or input_ in ["RNN_vectors",
                                                                           "dynamic_vectors_decoder"]) else ["data"]:
                                arr = self.dicts[patientid][input_][data_type]
                                if input_ == "RNN_2D" and self.shape_2D_records is None:
                                    arr, slice_idx = self.convert_to_tf_input([arr],
                                                                              if_var_shape=True)  ## Dummy dimension for time series.
                                    if "slice_idx" not in self.dicts[patientid][input_].keys():
                                        self.dicts[patientid][input_]["slice_idx"] = {data_type: slice_idx}
                                    else:
                                        self.dicts[patientid][input_]["slice_idx"][data_type] = slice_idx
                                else:
                                    arr = self.convert_to_tf_input([arr],
                                                                   if_var_shape=False)  ## Dummy dimension for time series.
                                if input_ == "RNN_2D" and self.if_separate_img:
                                    with open(f"{self.obj_dir_path}/{patientid}_{input_}_{data_type}.npy",
                                              'wb') as file_name:
                                        np.save(file_name, arr)
                                    del self.dicts[patientid][input_][data_type]
                                else:
                                    self.dicts[patientid][input_][data_type] = arr
                        for input_ in ["static_encoder", "raw"]:
                            for data_type in ["data", "mask", "concat"]:
                                self.dicts[patientid][input_][data_type] = self.convert_to_tf_input(
                                    [self.dicts[patientid][input_][data_type]])
                        self.dicts[patientid]["RE_DATE"] = self.convert_to_tf_input(
                            [[[RE_DATE] for RE_DATE in self.dicts[patientid]["RE_DATE"]]])
                        
                        current_num_bags += 1
                    if count_stationId_to_sample_idx[count_stationId] != -1:
                        if count_stationId_to_sample_idx[count_stationId] + max(self.num_records) >= count_stationId_to_size[count_stationId]:
                            count_stationId_to_sample_idx[count_stationId] = -1  # Mark bag as processed
                        else:
                            count_stationId_to_sample_idx[count_stationId] += 1


        elif self.dataset_kind == "toy": ## --- --- TOY DATASET.
            ## Set default arguments.
            kwargs_copy = merge_dictionaries([{"num_patients": 385, "max_time_steps": 80, "num_features_static_dict": dict(raw = 1, RNN_vectors = 1, static_encoder = 1), "num_features_dynamic": 2, "time_interval_range": [0., 0.05], "static_data_range": [0., 1.], "sin function info ranges dict": {"amplitude": [0., 1.], "displacement_along_x_axis": [0., 1.], "frequency": [0.5, 1.], "displacement_along_y_axis": [1., 2.]}, "observation noise range": [0., 0.], "missing probability": 0.0, "class_0_proportion": [0.5]}, kwargs_toy_dataset])
            num_features_static_dict= kwargs_copy["num_features_static_dict"]

            ## Change column names from the names of real dataset to the names of toy dataset.
            for input_ in num_features_static_dict.keys():
                for i in range(num_features_static_dict[input_]):
                    self.input_to_features_map[input_].append(f"static_{input_}_{i}")
            for input_ in ["RNN_vectors"]:
                for i in range(kwargs_copy["num_features_dynamic"]):
                    self.input_to_features_map[input_].append(f"dynamic_{i}")
            self.input_to_features_map["dynamic_vectors_decoder"] = deepcopy(self.input_to_features_map["RNN_vectors"])
            self.input_to_features_map["RNN_vectors"].append(f"RE_DATE") ## Last feature of LSTM input records is RE_DATE.

            ## Set the classification threshold for each class.
            # outcome_class_decision_threshold = [kwargs_copy[num_features_static] * (kwargs_copy["static_data_range"][1] -  kwargs_copy["static_data_range"][0])]
            # for range_ in kwargs_copy["sin function info ranges dict"].items():
            #     outcome_class_decision_threshold[0] += (range_[1] - range_[0]) * kwargs_copy["num_features_dynamic"]
            # outcome_class_decision_threshold[0] = outcome_class_decision_threshold[0] / 2.

            ## Stack 2D time series table.
            self.num_patients = kwargs_copy["num_patients"]
            for index in range(self.num_patients):
                time_steps = random.randint(1, kwargs_copy["max_time_steps"])

                ## Initialize the records. shape = (time_steps, num_features)
                num_features_RNN = len(self.input_to_features_map["RNN_vectors"])

                patient_dict = {}
                patient_dict["time_steps"] = time_steps
                patient_dict["RE_DATE"] = []
                patient_dict["predictor label"] = [[]]
                patient_dict["outcome_decision_values"] = [0.]
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    patient_dict[input_] = {}
                    patient_dict[input_]["data"] = [[None for i in range(len(self.input_to_features_map[input_]))] for j in range(time_steps)]
                    patient_dict[input_]["mask"] = [[None for i in range(len(self.input_to_features_map[input_]))] for j in range(time_steps)]
                    patient_dict[input_]["concat"] = [[] for j in range(time_steps)]
                    for i in range(num_features_static_dict["RNN_vectors"]):
                        static_feature = rand_gen_with_range(kwargs_copy["static_data_range"])
                        patient_dict["outcome_decision_values"][0] += static_feature
                        for j in range(time_steps):
                            patient_dict[input_]["data"][j][i] = static_feature
                            patient_dict[input_]["mask"][j][i] = 1.

                ## Set RE_DATE
                patient_dict["RE_DATE"] = [rand_gen_with_range(kwargs_copy["time_interval_range"])]
                for i in range(1, time_steps):
                    patient_dict["RE_DATE"].append(patient_dict["RE_DATE"][i - 1] + rand_gen_with_range(kwargs_copy["time_interval_range"]))
                
                ## Set static features.
                for input_ in ["raw", "static_encoder"]:
                    patient_dict[input_] = {}
                    patient_dict[input_]["data"] = [rand_gen_with_range(kwargs_copy["static_data_range"]) for i in range(num_features_static_dict[input_])]
                    patient_dict[input_]["mask"] = [1. for i in range(num_features_static_dict[input_])]
                    patient_dict[input_]["concat"] = deepcopy(patient_dict[input_]["data"] + patient_dict[input_]["mask"])
                    patient_dict["outcome_decision_values"][0] += sum(patient_dict[input_]["data"])


                ## Create records matrix
                for feature_idx in range(num_features_static_dict["RNN_vectors"], num_features_RNN - 1): ## Starting from num_features_static_dict["RNN_vectors"] to discard static features, -1 for discarded RE_DATE
                    amplitude = rand_gen_with_range(kwargs_copy["sin function info ranges dict"]["amplitude"])
                    frequency = rand_gen_with_range(kwargs_copy["sin function info ranges dict"]["frequency"])
                    displacement_along_x_axis = rand_gen_with_range(kwargs_copy["sin function info ranges dict"]["displacement_along_x_axis"])
                    displacement_along_y_axis = rand_gen_with_range(kwargs_copy["sin function info ranges dict"]["displacement_along_y_axis"])
                    patient_dict["outcome_decision_values"][0] += amplitude + frequency + displacement_along_x_axis + displacement_along_y_axis ## Static feature contribution to outcome_decision_values is already applied.

                    ## Create record vector
                    for t in range(time_steps):
                        if random.random() > kwargs_copy["missing probability"]: ## Observed case
                            data_gen = amplitude * np.sin((patient_dict["RE_DATE"][t] - displacement_along_x_axis) / frequency) + displacement_along_y_axis + rand_gen_with_range(kwargs_copy["observation noise range"])
                            mask_gen = 1.
                        else: ## Unobserved case
                            data_gen = self.init_number_for_unobserved_entry
                            mask_gen = 0.
                        for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                            patient_dict[input_]["data"][t][feature_idx] = data_gen
                            patient_dict[input_]["mask"][t][feature_idx] = mask_gen
                
                for t in range(time_steps): ## Last feature of each time step is RE_DATE.
                    assert(patient_dict["RNN_vectors"]["data"][t][-1] is None and patient_dict["RNN_vectors"]["mask"][t][-1] is None)
                    patient_dict["RNN_vectors"]["data"][t][-1] = patient_dict["RE_DATE"][t]
                    patient_dict["RNN_vectors"]["mask"][t][-1] = 1.

                ## Format data to fit with Keras Input.
                for date_idx in range(len(patient_dict["RE_DATE"]) - 1):
                    assert(patient_dict["RE_DATE"][date_idx] <= patient_dict["RE_DATE"][date_idx + 1])
                # patient_dict["RE_DATE"] = np.array([patient_dict["RE_DATE"]])
                patient_dict["RE_DATE"] = self.convert_to_tf_input([[[RE_DATE] for RE_DATE in patient_dict["RE_DATE"]]])

                ## list -> array
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    for t in range(time_steps):
                        patient_dict[input_]["concat"][t] = deepcopy(patient_dict[input_]["data"][t] + patient_dict[input_]["mask"][t])
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder", "raw", "static_encoder"]:
                    for type_ in ["data", "mask", "concat"]:
                        patient_dict[input_][type_] = self.convert_to_tf_input([patient_dict[input_][type_]]) ## Adds dummy batch dimension.
                # patient_dict['LSTM inputs data and mask concatenated'] = np.array([np.concatenate([patient_dict['LSTM inputs data'][0], patient_dict['LSTM inputs mask'][0]], axis = 1)])
                self.dicts.append(patient_dict)

            ## Set target labels.
            for class_0_proportion_idx in range(len(kwargs_copy["class_0_proportion"])): ## Supports multi-class label.
                rank = round(self.num_patients * kwargs_copy["class_0_proportion"][class_0_proportion_idx])
                self.dicts.sort(key = lambda x: x["outcome_decision_values"][class_0_proportion_idx])
                for patient_idx in range(0, rank):
                    self.dicts[patient_idx]["outcome"] = 0.
                    # self.dicts[patient_idx]["predictor label"][0].append(0.)
                for patient_idx in range(rank, self.num_patients):
                    self.dicts[patient_idx]["outcome"] = 1.
                    # self.dicts[patient_idx]["predictor label"][0].append(1.)
            for patient_idx in range(self.num_patients):
                self.dicts[patient_idx]["predictor label"] = self.prediction_label_mapping(self.dicts[patient_idx]["outcome"], inverse = False) ## Set outcome to be label of first class, change here if you want to adds multi classes.
                # self.dicts[patient_idx]["predictor label"] = np.array(self.dicts[patient_idx]["predictor label"])
            np.random.shuffle(self.dicts)
        
        elif self.dataset_kind == "alz": ## --- --- ADNI dataset.
            self.dicts = []
            ## Stack the actual data which will be fed into Autoencoder.
            for valid_idx in tqdm(self.valid_indices["intersect"]):
                patient_dict = {"valid_idx": valid_idx, "RE_DATE": []} ## only for this patient.
                ## --- Set static info.
                for input_ in ["static_encoder", "raw"]:
                    patient_dict[input_] = dict(
                        data = [self.df_dict["static"][self.input_to_features_map[input_]].iloc[valid_idx].fillna(self.init_number_for_unobserved_entry).tolist()],
                        mask = [(1. - self.df_dict["static"][self.input_to_features_map[input_]].iloc[valid_idx].isna()).tolist()],
                        concat = [self.df_dict["static"][self.input_to_features_map[input_]].iloc[valid_idx].fillna(self.init_number_for_unobserved_entry).tolist() + (1. - self.df_dict["static"][self.input_to_features_map[input_]].iloc[valid_idx].isna()).tolist()]
                        )

                ## --- Set target label.
                patient_dict["outcome"] = self.df_dict["outcome"].iloc[valid_idx]
                patient_dict["ID"] = self.df_dict["info"]["SubjID"].iloc[valid_idx]
                patient_dict["predictor label"] = self.prediction_label_mapping(patient_dict["outcome"], inverse = False)

                ## --- Set dynamic info
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder"]:
                    patient_dict[input_] = dict(data= [[]], mask= [[]], concat= [[]])
                patient_dict["RE_DATE"] = [[]] ## [[[RE_DATE], [RE_DATE], ...]]

                ## Set static features for RNN input.
                static_for_RNN = {"data": [], "mask": []}
                for static_modality in self.static_data_input_mapping["RNN_vectors"]: ## appends this list before dynamic feature list, and follows the sequence in self.static_data_input_mapping["RNN_vectors"].
                    static_for_RNN["data"]= deepcopy(static_for_RNN["data"] + self.df_dict["static"][self.static_modality_to_features_map[static_modality]].iloc[valid_idx].fillna(self.init_number_for_unobserved_entry).tolist())
                    static_for_RNN["mask"] = deepcopy(static_for_RNN["mask"] + (1. - self.df_dict["static"][self.static_modality_to_features_map[static_modality]].iloc[valid_idx].isna()).tolist())
                
                for time in self.TIMES:
                    RE_DATE = [self.TIMES_TO_RE_DATE[time]]
                    record_mask = (1. - self.df_dict["dynamic"][time].iloc[valid_idx].isna()).tolist()
                    if sum(record_mask) / max(len(record_mask), 1) >= observation_density_threshold: ## Filter out sparse record.
                        record_data = self.df_dict["dynamic"][time].iloc[valid_idx].fillna(self.init_number_for_unobserved_entry).tolist()
                        patient_dict["dynamic_vectors_decoder"]["data"][0].append(deepcopy(static_for_RNN["data"] + record_data))
                        patient_dict["dynamic_vectors_decoder"]["mask"][0].append(deepcopy(static_for_RNN["mask"] + record_mask))
                        patient_dict["dynamic_vectors_decoder"]["concat"][0].append(deepcopy(static_for_RNN["data"] + record_data + static_for_RNN["mask"] + record_mask))
                        patient_dict["RNN_vectors"]["data"][0].append(deepcopy(static_for_RNN["data"] + record_data + RE_DATE))
                        patient_dict["RNN_vectors"]["mask"][0].append(deepcopy(static_for_RNN["mask"] + record_mask + [1.]))
                        patient_dict["RNN_vectors"]["concat"][0].append(deepcopy(static_for_RNN["data"] + record_data + RE_DATE + static_for_RNN["mask"] + record_mask + [1.]))
                        patient_dict["RE_DATE"][0].append(RE_DATE)
                # patient_dict["RNN_vectors"]["concat"] = np.array([np.concatenate([patient_dict["RNN_vectors"]["data"][0], patient_dict["RNN_vectors"]["mask"][0]], axis = 1)])
                ## --- list to array.
                for input_ in ["RNN_vectors", "dynamic_vectors_decoder", "static_encoder", "raw"]:
                    for data_type in ["data", "mask", "concat"]:
                        patient_dict[input_][data_type] = self.convert_to_tf_input(patient_dict[input_][data_type])
                patient_dict["RE_DATE"] = self.convert_to_tf_input(patient_dict["RE_DATE"])
                self.dicts.append(patient_dict)
            
            self.num_patients = len(self.dicts) ## Finish this participiant.
            ## --- SANITY TEST for Alz datset: pick participant: 011_S_0002 (valid_index: 0, snp and diagnosis both exists).
            ## static data sanity.
            if len(self.static_data_input_mapping["static_encoder"]) > 0 and self.static_data_input_mapping["static_encoder"][0] == "SNP" and 0 in self.valid_indices["intersect"]:
                assert(self.dicts[0]["static_encoder"]["concat"][0][0] * (self.feature_min_max_dict["rs4846048"]["max"] - self.feature_min_max_dict["rs4846048"]["min"]) + self.feature_min_max_dict["rs4846048"]["min"] == 1.0)
                assert(self.dicts[0]["static_encoder"]["concat"][0][2] * (self.feature_min_max_dict["rs1476413"]["max"] - self.feature_min_max_dict["rs1476413"]["min"]) + self.feature_min_max_dict["rs1476413"]["min"] == 0.0)
            ## dynamic data sanity.
            assert(self.dicts[0]["RNN_vectors"]["concat"][0][0][self.input_to_features_map["RNN_vectors"].index("VBM_mod_LCalcarine")] * (self.feature_min_max_dict["VBM_mod_LCalcarine"]["max"] - self.feature_min_max_dict["VBM_mod_LCalcarine"]["min"]) + self.feature_min_max_dict["VBM_mod_LCalcarine"]["min"] == 0.403619369811371)
            assert(abs(self.dicts[0]["RNN_vectors"]["concat"][0][0][self.input_to_features_map["RNN_vectors"].index("FS_MPavg_RTransvTemporal")] * (self.feature_min_max_dict["FS_MPavg_RTransvTemporal"]["max"] - self.feature_min_max_dict["FS_MPavg_RTransvTemporal"]["min"]) + self.feature_min_max_dict["FS_MPavg_RTransvTemporal"]["min"] - 1.85876451391478) < 1e-5)
            assert(self.dicts[0]["outcome"] == 3.)

        else:
            raise Exception(NotImplementedError)
        
        self.num_patients = len(self.dicts)
        for idx in range(len(self.dicts)):
            self.dicts[idx]["sample_idx"] = idx
        self.idx_key_map = dict()
        self.key_idx_map = dict()
        for idx in range(len(self.dicts)):
            self.idx_key_map[idx] = self.dicts[idx]["patientid"]
            self.key_idx_map[self.dicts[idx]["patientid"]] = idx
        
        # if self.if_create_data_frame_each_bag:
        #     for idx in range(len(self.dicts)):
        #         static_data = self.dicts[idx]["s"]
                    
    def split_train_test(self, train_proportion = None, k_fold = 2, whether_training_is_large = False, shuffle = True, balance_label_distribution = False):
        """Set the indices of test and training set for each k-fold split.
        
        Parameters
        ----------
        k_fold : int
            The number of splits in k-fold cross validation. This is ignored when train_proportion is not None.
        whether_training_is_large : bool
            Whether the training set picks the remaining splits of k-folded indices. his is ignored when train_proportion is not None.
        """

        assert(k_fold > 1)
        self.this_dataset_is_splitted = True
        self.train_proportion = train_proportion
        self.k_fold = k_fold
        self.whether_training_is_large = whether_training_is_large
        self.splits_train_test = [] ## [{"train": [...], "test": [...]}, {"train": [...], "test": [...]}, {"train": [...], "test": [...]}, ...].
        self.most_recent_record_dict = {"x_train": [], "y_train": [], "x_test": [], "y_test": []} ## {"x_train": [k= 0, k= 1, ...], "y_train": [k= 0, k= 1, ...], "x_test": [k= 0, k= 1, ...], "y_test": [k= 0, k= 1, ...]}
        num_patients_each_split = round(self.num_patients / k_fold)
        self.indices_patients_splits = [] ## [[indices], [indices], ...].
        self.indices_patients_shuffled = list(range(self.num_patients)) ## [indices of all]
        if shuffle: np.random.shuffle(self.indices_patients_shuffled)

        if train_proportion is None: ## k_fold cross validation is not applied.
            ## Stack self.indices_patients_splits : [[indices], [indices], ...].
            split_idx = 0
            for k in range(k_fold - 1):
                self.indices_patients_splits.append(self.indices_patients_shuffled[split_idx * num_patients_each_split: (split_idx + 1) * num_patients_each_split])
                split_idx += 1
            if split_idx * num_patients_each_split < self.num_patients: self.indices_patients_splits.append(self.indices_patients_shuffled[split_idx * num_patients_each_split:])
            assert(len(self.indices_patients_splits) == k_fold)

            ## self.splits_train_test : [{"train": [...], "test": [...]}, {"train": [...], "test": [...]}, {"train": [...], "test": [...]}, ...].
            for k in range(k_fold):
                one_piece = self.indices_patients_splits[k]
                remaining_pieces = []
                for k_ in range(k_fold):
                    if k_ != k: remaining_pieces = remaining_pieces + self.indices_patients_splits[k_]
                if whether_training_is_large: self.splits_train_test.append({"train": deepcopy(remaining_pieces), "test": deepcopy(one_piece)})
                else: self.splits_train_test.append({"train": deepcopy(one_piece), "test": deepcopy(remaining_pieces)})

        else: ## k_fold cross validation is applied.
            num_train = round(train_proportion * self.num_patients)
            self.splits_train_test.append({"train": deepcopy(self.indices_patients_shuffled[:num_train]), "test": deepcopy(self.indices_patients_shuffled[num_train:])})

        ## Set self.most_recent_record_dict for baseline models.
        for k in range(len(self.splits_train_test)): ## range(k_fold).
            x_train, y_train, x_test, y_test = [], [], [], []
            for patient_idx_train in self.splits_train_test[k]["train"]:
                x_train.append(np.concatenate([self.dicts[patient_idx_train]["RNN_vectors"]["data"][0, -1], self.dicts[patient_idx_train]["static_encoder"]["data"][0], self.dicts[patient_idx_train]["raw"]["data"][0]], axis = 0))
                y_train.append(self.dicts[patient_idx_train]["predictor label"][0]) ## 0 for dummy batch dimension.
            for patient_idx_test in self.splits_train_test[k]["test"]:
                x_test.append(np.concatenate([self.dicts[patient_idx_test]["RNN_vectors"]["data"][0, -1], self.dicts[patient_idx_test]["static_encoder"]["data"][0], self.dicts[patient_idx_test]["raw"]["data"][0]], axis = 0))
                y_test.append(self.dicts[patient_idx_test]["predictor label"][0]) ## 0 for dummy batch dimension.
            ## List -> Array.
            self.most_recent_record_dict["x_train"].append(np.array(x_train))
            self.most_recent_record_dict["y_train"].append(np.array(y_train))
            self.most_recent_record_dict["x_test"].append(np.array(x_test))
            self.most_recent_record_dict["y_test"].append(np.array(y_test))
        
        if balance_label_distribution: ## For unbalanced dataset, upsample the minor-classes.
            for split in self.splits_train_test:
                list_of_train_indices_orig = split["train"]
                list_of_train_indices_augmented = deepcopy(split["train"])
                counts_dict = utilsforminds.containers.get_items_counts_dict_from_container(container = list_of_train_indices_orig, access_to_item_funct = lambda x: self.dicts[x]["predictor label"])
                num_largest_class = utilsforminds.containers.get_max_with_accessor(counts_dict, accessor = lambda x: counts_dict[x])

                for train_index in list_of_train_indices_orig:
                    predictor_label = tuple(map(tuple, self.dicts[train_index]["predictor label"]))
                    num_copies = (num_largest_class - counts_dict[predictor_label]) / counts_dict[predictor_label]
                    for i in range(round(num_copies)):
                        list_of_train_indices_augmented.append(train_index)
                    if random.random() < num_copies - round(num_copies):
                        list_of_train_indices_augmented.append(train_index)
                if shuffle: np.random.shuffle(list_of_train_indices_augmented)
                split["train"] = deepcopy(list_of_train_indices_augmented)

        for split in self.splits_train_test:
            for type_ in ["train", "test"]:
                for sample_idx in split[type_]:
                    assert(self.dicts[sample_idx]["sample_idx"] == sample_idx)

    def set_observabilities(self, indices_patients_train, indices_patients_test):
        for patient_idx in indices_patients_train:
            self.dicts[patient_idx]["observability"] = self.convert_to_tf_input([[1.]]) ## np.array([[1.]]), in training set.
        for patient_idx in indices_patients_test:
            self.dicts[patient_idx]["observability"] = self.convert_to_tf_input([[0.]]) ## in test set.
    
    def export_to_csv(self, dir_path_to_export = "./datasets/datasets_in_csv/"):
        """Exports ONLY RNN input features, to csv file of each participant.
        
        """

        os.mkdir(dir_path_to_export)
        for patient_dict_idx, patient_dict in zip(range(len(self.dicts)), self.dicts):
            for type_ in ["data", "mask"]:
                df = pd.DataFrame(patient_dict["RNN_vectors"][type_][0], columns = self.input_to_features_map["RNN_vectors"])
                df["outcome"] = [patient_dict["outcome"] for i in range(patient_dict["time_steps"])]
                df.to_csv(f"{dir_path_to_export}{patient_dict_idx}_{type_}.csv")
            # if self.separate_dynamic_static_features == "separate raw" or self.separate_dynamic_static_features == "separate enrich":
            #     df = pd.DataFrame(patient_dict[f'static_features_data'], columns = self.static_feature_names)
            #     df.to_csv(f"{dir_path_to_export}{patient_dict_idx}_static_data.csv")

    def clear_misc(self):
        """Remove non-necessary data"""

        self.dicts = None
    
    def prediction_label_mapping_default_funct(self, label, if_many_to_one = False):
        """If label is scalar -> vector (1, n), else if label is vector (n, ) -> scalar"""

        if self.dataset_kind == "covid" or self.dataset_kind == "toy" or self.dataset_kind == "challenge" or self.dataset_kind == "chestxray":
            if not if_many_to_one:
                if round(label) == 0: return self.convert_to_tf_input([[1., 0.]])
                elif round(label) == 1: return self.convert_to_tf_input([[0., 1.]])
                else: raise Exception(NotImplementedError)
            else:
                if np.argmax(label) == 0: return 0
                elif np.argmax(label) == 1: return 1
                else: raise Exception(NotImplementedError)
        elif self.dataset_kind == "alz":
            if not if_many_to_one:
                if round(label) == 1: return self.convert_to_tf_input([[1., 0., 0.]])
                elif round(label) == 2: return self.convert_to_tf_input([[0., 1., 0.]])
                elif round(label) == 3: return self.convert_to_tf_input([[0., 0., 1.]])
                else: raise Exception(NotImplementedError)
            else:
                if np.argmax(label) == 0: return 1
                elif np.argmax(label) == 1: return 2
                elif np.argmax(label) == 2: return 3
                else: raise Exception(NotImplementedError)
        elif self.dataset_kind == "traffic":
            if not if_many_to_one:
                encoded_label = [0. for i in range(self.labels_info["num_label_divisions"])]
                for i in range(self.labels_info["num_label_divisions"] - 1):
                    if label < self.labels_info["speeds_threshold"][i]:
                        encoded_label[i] = 1.
                        break
                if sum(encoded_label) == 0.: encoded_label[-1] = 1.
                return self.convert_to_tf_input([encoded_label])
            else:
                return np.argmax(label)
                for i in range(self.labels_info["num_label_divisions"]):
                    if self.labels_info["speeds_threshold"][i] < label < self.labels_info["speeds_threshold"][i + 1]:
                        return self.labels_info["speeds_threshold"][i]
                return self.labels_info["speeds_means"][-1]
        elif self.dataset_kind == "colorado_traffic":
            if self.classification_axis is None:
                if not if_many_to_one:
                    return label
                else:
                    return [1.0 if label[i] > 0.5 else 0.0 for i in range(len(label))]
            else:
                if not if_many_to_one:
                    return label
                else:
                    out = []
                    for i in range(len(label)):
                        if i == self.classification_axis:
                            out.append(1.0 if label[i] > 0.5 else 0.0)
                        else:
                            out.append(label[i])
                    return out
        else:
            raise Exception(NotImplementedError)

    
    def reduce_number_of_participants(self, reduced_number_of_participants):
        """Reduce the number of samples in this dataset.
        
        Usually used to create dataset for debugging purpose.

        Args:
            reduced_number_of_participants ([type]): [description]
        """

        assert(not self.this_dataset_is_splitted) ## Should not be k-fold splitted yet.
        if reduced_number_of_participants < self.num_patients:
            # self.dicts = random.sample(self.dicts, reduced_number_of_participants)
            self.dicts = self.dicts[:reduced_number_of_participants]
            print(f"The number of participants is changed from {self.num_patients} to {reduced_number_of_participants}")
            self.num_patients = reduced_number_of_participants
        else:
            print("WARNING: The requested number of participants is larger than current number of participants, so do nothing.")
    
    def get_data_collections_from_all_samples(self, data_access_function):
        """Get the list of collections.
        
        Parameters
        ----------
        data_access_function : callable
            For example, lambda x: x['outcome'].
        """

        collections = []
        for idx in range(len(self.dicts)):
            collections.append(data_access_function(self.dicts[idx]))
        return collections

    def get_measures_of_feature(self, feature_name, group_name = "static", reverse_min_max_scale = False):
        """

        Parameters
        ----------
        feature_access_funct : callable
        Function to access the features given self.groups_of_features_info. For example lambda x: x["static"]["Age"]["observed_numbers"].
        
        Examples
        --------
        """

        observed_numbers = np.array(self.groups_of_features_info[group_name][feature_name]["observed_numbers"])
        if reverse_min_max_scale and self.feature_min_max_dict[feature_name]["max"] > self.feature_min_max_dict[feature_name]["min"]:
            observed_numbers = observed_numbers * (self.feature_min_max_dict[feature_name]["max"] - self.feature_min_max_dict[feature_name]["min"]) + self.feature_min_max_dict[feature_name]["min"]
        return observed_numbers
    
    def convert_to_tf_input(self, data, if_var_shape = False, ragged_kwargs = None):
        if not if_var_shape:
            return np.array(data)
        else:
            if ragged_kwargs is None: ragged_kwargs = {}
            if False:
                if tf.is_tensor(data):
                    return data
                else:
                    return tf.ragged.constant(data, **ragged_kwargs)
            else:
                assert(len(data[0][0].shape) == 3) ## Only support image data
                data_padded = []
                slice_idx = []
                for img in data[0]:
                    slice_idx.append([img.shape[0], img.shape[1]])
                max_size = [max([idx[axis] for idx in slice_idx]) for axis in (0, 1)]
                for img in data[0]:
                    data_padded.append(np.pad(array= img, pad_width= ((0, max_size[0] - img.shape[0]), (0, max_size[1] - img.shape[1]), (0, 0)), mode= 'constant', constant_values= 0))
                
                slice_idx = np.array([slice_idx])
                assert(slice_idx.shape[-1] == 2 and len(slice_idx.shape) == 3) ## (batch, timesteps, 2)
                return np.array([data_padded]), slice_idx
    
    def load_arr(self, sample_idx, input_, data_type = None):
        assert(self.if_separate_img)
        patientid = self.idx_key_map[sample_idx]
        if data_type is not None:
            return np.load(f"{self.obj_dir_path}/{patientid}_{input_}_{data_type}.npy")
        else:
            out_dict = {}
            # for key in ["data", "mask", "concat"]:
            for key in ["data"]:
                out_dict[key] = np.load(f"{self.obj_dir_path}/{patientid}_{input_}_{key}.npy")
            return out_dict
    
    def populate_data_mask_concat(self, dict_populated, row, if_append):
        if if_append:
            dict_populated["data"].append(row.fillna(self.init_number_for_unobserved_entry).tolist())
            dict_populated["mask"].append((1. - row.isna()).tolist())
            dict_populated["concat"].append(deepcopy(dict_populated["data"][-1] + dict_populated["mask"][-1]))
        else:
            dict_populated["data"] = row.fillna(self.init_number_for_unobserved_entry).tolist()
            dict_populated["mask"] = (1. - row.isna()).tolist()
            dict_populated["concat"] = deepcopy(dict_populated["data"] + dict_populated["mask"])

def change_names_from_starting_names(names: list, starting_names: list):
    """Useful when you change column names form the column names changed by pd.get_dummies.

    Examples
    --------
    names = ["gender", "age", "cells"]
    starting_names = ["gender_male", "gender_female", "age", "cells_1", "cells_2", "cells_3", "additional_column"]
    print(change_names_from_starting_names(names, starting_names))
    >>> ['gender_male', 'gender_female', 'age', 'cells_1', 'cells_2', 'cells_3']
    
    """
    changed_names = []
    for name in names:
        if name in starting_names:
            changed_names.append(name)
        else:
            for starting_name in starting_names:
                if starting_name.startswith(name):
                    changed_names.append(starting_name)
    return changed_names


def get_arr_of_image_from_path(path_to_img, shape = None, nopostprocess = False, apply_segment = True):
    # if shape is None: shape = (1051, 1024)
    noHU = True
    
    if path_to_img.endswith("nii.gz"):
        input_image = nib.load(path_to_img)
        input_image = skimage.color.rgb2gray(input_image.get_fdata())
    else:
        input_image = cv2.imread(path_to_img, 0) ## param 0 means gray-mode image load.
    input_image_arr = cv2.equalizeHist(input_image)
    input_image_equalized = sitk.GetImageFromArray(input_image_arr)
    if shape is None:
        shape = input_image_arr.shape
    
    if apply_segment:
        model = mask.get_model("unet", 'R231CovidWeb')
        result = mask.apply(input_image_equalized, model = model, noHU = noHU, volume_postprocessing = not nopostprocess)  # default model is U-net(R231)

    if not apply_segment or type(result) == str:
        if apply_segment: print(f"Segment failure with reason: {result}, segmentation will not applied to image.")
        img_arr_tosave = input_image_arr
        mask_arr = np.ones(shape = shape)
    else:
        if noHU:
            file_ending = path_to_img.split('.')[-1]
            if file_ending in ['jpg','jpeg','png']:
                result = (result/(result.max())*255).astype(np.uint8)
            result = result[0]
        
        if len(input_image_arr.shape) == 3:
            raise Exception(f"Should be gray image already by parameter 0 in cv2.imread(img_path, 0)")
            grey_img = skimage.color.rgb2gray(input_image_arr)
        elif len(input_image_arr.shape) == 2:
            grey_img = input_image_arr
        else:
            raise Exception(f"Unsupported dimension: {input_image_arr.shape}")
        
        if grey_img.max() > 1.0:
            grey_img = grey_img / 255

        mask_arr = result
        img_arr_tosave = grey_img * mask_arr
    
    ## Resize Image
    img_arr_tosave = skimage.transform.resize(img_arr_tosave, list(shape) + [1]) ## Adds [1] for (gray) channel dimension
    mask_arr = skimage.transform.resize(mask_arr, list(shape) + [1]) ## Adds [1] for (gray) channel dimension

    ## Normalize
    if img_arr_tosave.max() > 1.0:
        img_arr_tosave = img_arr_tosave / 255
    if mask_arr.max() > 1.0:
        mask_arr = mask_arr / 255
    
    return img_arr_tosave, mask_arr

def sorted_multiple_lists(list_sort_baseline, *other_lists, reverse = False):
    """
    
    Examples
    --------
    foo = ["c", "b", "a"]
    bar = [1, 2, 3]
    too = [5, 4, 6]
    foo, bar, too = sorted_multiple_lists(foo, bar, too)
    print(foo, bar, too)
    >>> ('a', 'b', 'c') (3, 2, 1) (6, 4, 5)
    foo, bar, too = sorted_multiple_lists(foo, bar, too, reverse = True)
    print(foo, bar, too)
    >>> ('c', 'b', 'a') (1, 2, 3) (5, 4, 6)
    too, bar, foo = sorted_multiple_lists(too, bar, foo)
    print(foo, bar, too)
    >>> ('b', 'c', 'a') (2, 1, 3) (4, 5, 6)
    [foo, bar, too] = sorted_multiple_lists(*[foo, bar, too])
    print(foo, bar, too)
    >>> ('a', 'b', 'c') (3, 2, 1) (6, 4, 5)
    foo, *[bar, too] = sorted_multiple_lists(foo, *[bar, too])
    print(foo, bar, too)
    >>> ('a', 'b', 'c') (3, 2, 1) (6, 4, 5) 
    """

    return zip(*sorted(zip(list_sort_baseline, *other_lists), reverse = reverse, key = lambda x: x[0]))

# def get_valid_ids_chestxray(df):
#     group_df = df.groupby(["patientid"])
#     for name, group in group_df:

if __name__ == "__main__":
    foo = ["c", "b", "a"]
    bar = [1, 2, 3]
    too = [5, 4, 6]
    foo, *[bar, too] = sorted_multiple_lists(foo, *[bar, too])
    print(foo, bar, too)

    print("END")
    