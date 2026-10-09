from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question

dname   = "../data/my_dataset"
fname   = "../data/my_dataset/data.txt"
configf = "../configs/data_config.json"

split_concept(
    dname=dname,
    fname=fname,
    dataset_name="my_dataset",
    configf=configf,
    min_seq_len=3,
    maxlen=200,
    kfold=5
)

split_question(
    dname=dname,
    fname=fname,
    dataset_name="my_dataset",
    configf=configf,
    min_seq_len=3,
    maxlen=200,
    kfold=5
)