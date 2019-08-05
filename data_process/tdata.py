import pandas as pd

class tdata(object):
    def __init__(self):
        return

    @staticmethod
    def isnull(data_frame, is_print=False):
        row = data_frame.shape[0]
        count_frame = data_frame.count(axis=0)

        null_info = {}
        for c in list(data_frame.columns):
            null_info[c] = row - count_frame[c]

        if is_print:
            for key, val in null_info.items():
                print('{0}-is has {1} nan values'.format(key, val))

        return null_info

    @staticmethod
    def down_sample(data_frame, frac=None, n=None):
        return data_frame.sample(frac=frac, n=n, random_state=1)
