from dataclasses import dataclass, field
from typeless_dataclasses import typeless


@dataclass
class TrainArgs(object):
    cfg:str = field(
        default="./Summ-N-ECS/configure/ICSI.cfg",
        metadata={"help": "The path from ./configure to store configure file."})

    dataset_path:str = field(
        default="./Summ-N-ECS/data/ICSI/",
        metadata={"help": "The absolute path to the dataset folder."}
    )

    output_path:str = field(
        default="./Summ-N-ECS/ICSI",
        metadata={"help": "The path to the output folder."}
    )

    save_intermediate:str = field(
        default="True",
        metadata={"help": "Store or not the intermediate files, such as original dataset."}
    )

    model_path:str = field(
        default="./Summ-N-ECS/bart.large.cnn/model.pt",
        metadata={"help": "The path to store the models .pt checkpoint. The models is loaded before training"})

    cuda_devices:str = field(
        default="1",
        metadata={'help': "The index of GPUs used to train BART-large, seperated by , ."}
    )

    mode:str = field(
        default="train",
        metadata={"help": "Train the whole dataset or test on test set."}
    )

    checkpoint_dir:str = field(
        default="./Summ-N-ECS/ICSI",
        metadata={"help": "The directory to save the checkpoints"}
    )