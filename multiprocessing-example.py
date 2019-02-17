# Inspiration: https://stackoverflow.com/questions/15704010/write-data-to-hdf-file-using-multiprocessing

import multiprocessing as mp
import threading
import warnings
from contextlib import contextmanager

import pandas as pd
import tables
from tqdm import tqdm

from misc import exploader

task = exploader.load('project_x/1902/190212_make_fundamentals_dataset')[0]
SENTINEL = None
NUM_PROCESSES = mp.cpu_count()
HDF_LOCK = threading.Lock()
HDF_FILEPATH = task.output().ensure().local_path


@contextmanager
def locked_store():
    with HDF_LOCK:
        with pd.HDFStore(HDF_FILEPATH) as store:
            yield store


def get_feature(inqueue, outqueue):
    with locked_store() as dataset:
        for feature in iter(inqueue.get, SENTINEL):
            df = pd.DataFrame.from_dict(
                {symbol: dataset['s{}'.format(symbol)][feature]
                 for symbol in symbols}
            )
            outqueue.put(('put', (feature, df)))


def writer(outqueue):
    pbar = tqdm(total=len(features))
    hdf = pd.HDFStore(path='feature_wise_data_v1.h5', mode='w')
    while True:
        args = outqueue.get()
        pbar.update()
        if args:
            method, args = args
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',
                                      tables.NaturalNameWarning)
                getattr(hdf, method)(*args)
        else:
            break
    hdf.close()


if __name__ == '__main__':

    dataset = task.output().ensure().load()
    symbols = list(dataset['/universe'].columns)
    features = list(dataset[dataset.keys()[2]].columns)
    dataset.close()

    outqueue = mp.Queue()
    inqueue = mp.Queue()
    jobs = list()

    writer_process = mp.Process(target=writer, args=(outqueue, ))
    writer_process.start()

    for i in range(NUM_PROCESSES):
        feature_process = mp.Process(
            target=get_feature,
            args=(inqueue, outqueue)
        )
        jobs.append(feature_process)
        feature_process.start()

    for feature in features:
        inqueue.put(feature)

    for i in range(NUM_PROCESSES):
        inqueue.put(SENTINEL)

    for proc in jobs:
        proc.join()

    outqueue.put(None)
    writer_process.join()
