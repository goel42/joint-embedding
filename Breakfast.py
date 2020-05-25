import numpy as np
import os
from torch.utils.data import Dataset
import sys

from word2vec.word2vec import load_word2vec

# SIL must not be used. because text_uniq index 0 contains label 1

# rmSIL edge case?
# stereo02 edge case?

# proper class structure?


def __get_map__(path, inv=False):
    """
    returns
    if inv == False : a dictionary of label_strings to integer
    else: a dictionary of integer to label_strings
    """

    m = {}
    with open(path, "r") as f:
        text = f.readlines()

    for line in text:
        temp = line.strip().split()
        num, activity = int(temp[0]), temp[1]
        if inv:
            m[num] = activity
        else:
            m[activity] = num

    return m


def __get_txt_labels__(path):
    with open(path, "r") as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]
    return labels


def is_src_sync(data):
    first_key = next(iter(data))
    first_label_array = data[first_key]["labels_idx"]
    num_segments, dim = data[first_key]["vis_feats"].shape

    assert data[first_key]["vis_feats"].shape[0] == data[first_key]["labels_idx"].shape[0]

    for key in data:
        if data[key]["vis_feats"].shape != (num_segments, dim):
            return False
        if data[key]["labels_idx"].shape[0] != num_segments:
            return False
        if data[key]["labels_idx"].all() != first_label_array.all():
            print("ERROR! Labels are not identical across sources of same person+activity")
            return False
    return True


class Breakfast(Dataset):

    def __init__(self, mod_dir_path, label_dir_path, map_path, sources, activities, persons, rm_SIL=False):
        self.sources = sources
        self.activities = activities
        self.persons = persons

        self.stoi_map = __get_map__(map_path)
        self.itos_map = __get_map__(map_path, inv=True)

        self.mod_dim = dict([(src, 1024) for src in self.sources]) #change
        self.data = self.__get_data__(mod_dir_path, label_dir_path, rm_SIL)

        self.word2vecs = load_word2vec()
        self.labels_uniq = self.__get_uniq_labels__(map_path, rm_SIL)
        # print("BLAH")

    def __len__(self):
        return len(self.data["labels_idx"])

    def __getitem__(self, idx):
        d = dict()
        for src in self.sources:
            d[src] = self.data[src][idx]
            d[src+"_ind"] = self.data[src+"_ind"][idx]
        d["labels_idx"] = self.data["labels_idx"][idx]
        return d

    def __get_data__(self, mod_dir_path, label_dir_path, rm_SIL):
        aggr_data = {}
        for src in self.sources:
            aggr_data[src] = []
            aggr_data[src+"_ind"] =[]
        aggr_data["labels_idx"] = []

        for person in self.persons:
            for activity in self.activities:
                temp_aggr_data = {}
                for src in self.sources:
                    curr_data = self.__data_helper__(person, src, activity, mod_dir_path, label_dir_path, rm_SIL)
                    if curr_data != {}:
                        temp_aggr_data[src] = curr_data

                if temp_aggr_data == {} or not is_src_sync(temp_aggr_data):
                    print("[Empty or Not Sync] Skipping activity: " + activity + "for Person: " + person)
                else:
                    first_key = next(iter(temp_aggr_data))
                    num_segments, dim = temp_aggr_data[first_key]["vis_feats"].shape

                    labels_idx = temp_aggr_data[first_key]["labels_idx"]
                    aggr_data["labels_idx"].extend(labels_idx)

                    for src in self.sources:
                        if src in temp_aggr_data:
                            aggr_data[src].extend(temp_aggr_data[src]["vis_feats"])
                            aggr_data[src + "_ind"].extend(np.ones(num_segments))
                        else:
                            # print("DDEBUG:")
                            # print(src, num_segments, self.mod_dim[src])
                            aggr_data[src].extend(np.zeros([num_segments, self.mod_dim[src]]))
                            aggr_data[src +"_ind"].extend(np.zeros(num_segments))

        for key in aggr_data:
            aggr_data[key] = np.array(aggr_data[key])

        return aggr_data

    def __data_helper__(self, person, src, activity, mod_dir_path, label_dir_path, rm_SIL):
        """
        returns,
        -> {} when the destination does not exist
        -> dict containing features [num_segments x dim] and "indices of labels" (0-indexed)

        if if rm_SIL == False (i.e. SILs not removed), then labels are already 0-indexed
        """

        file_name = person + "_" + activity + "_" + src
        mod_file = file_name + ".npy"
        mod_path = os.path.join(mod_dir_path, mod_file)

        label_file = file_name + ".txt"
        label_path = os.path.join(label_dir_path, label_file)

        if not os.path.exists(mod_path) and not os.path.exists(label_path):
            return {}

        feats = np.load(mod_path)
        labels = __get_txt_labels__(label_path)
        labels = [self.stoi_map[txt_label] for txt_label in labels]
        labels = np.array(labels)

        # thee are both cases where label < feats and feats < labels
        # Temp Debug. #TODO remove after dataset cleaning
        if feats.shape[0] != labels.shape[0]:
            # if labels.shape[0] > feats.shape[0] :
            #     print("FATAL! There are files where frames are more than visual features")
            #     print(feats.shape, labels.shape)
            #     print(vf)
            print("ERROR! Num of frames and corresponding labels are not equal")
            min_len = min(feats.shape[0], labels.shape[0])
            feats = feats[:min_len, :]
            labels = labels[:min_len]
        assert feats.shape[0] == labels.shape[0]

        if rm_SIL:
            mask = labels != 0
            labels = labels[mask]
            feats = feats[mask]
            labels -= 1

        return {"vis_feats": feats, "labels_idx": labels}

    def __get_uniq_labels__(self, map_path, rm_SIL):
        labels_num = []
        labels_txt = []

        labels_num_w2v = []
        labels_txt_w2v = []

        with open(map_path, "r") as f:
            text = f.readlines()

        for line in text:
            temp = line.strip().split()
            num, activity = int(temp[0]), temp[1]
            if rm_SIL and activity == "SIL":
                continue
            labels_num.append(num)
            labels_txt.append(activity)

            labels_num_w2v.append(self.__get_w2v__(str(num)))
            labels_txt_w2v.append(self.__get_w2v__(activity))

        return {"labels_num": np.array(labels_num),
                "labels_txt": np.array(labels_txt),
                "labels_num_w2v": np.array(labels_num_w2v),
                "labels_txt_w2v": np.array(labels_txt_w2v)
                }

    def __get_w2v__(self, caption):

        """if caption == SIL then sends back a vector of 0s"""

        # tokens_count = 2  # TODO
        tokens_count = len(caption.split("_"))
        vec_size = 200
        w2v = np.zeros([tokens_count, vec_size])

        if caption != 'SIL':
            for i, token in enumerate(caption.split("_")):
                w2v[i, :] = self.word2vecs[token]
        max_pooled_w2v = np.max(w2v, axis=0)
        return max_pooled_w2v