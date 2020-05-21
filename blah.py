def __get_uniq_labels__(self, map_path, rm_SIL):
    labels_num = []
    labels_txt =[]

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

        labels_num_w2v.append(self.__get_w2v__(str(num))
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