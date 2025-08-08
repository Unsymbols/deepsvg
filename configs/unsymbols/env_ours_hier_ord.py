from configs.deepsvg.default_icons import *
from pathlib import Path

# if DATA_DIR env variable is set, use it
import os

data_dir = os.environ.get("DSVG_TRAIN_DATA_DIR", None)
batch_size = os.environ.get("DSVG_TRAIN_BS", None)
num_gpus = os.environ.get("DSVG_TRAIN_NUM_GPUS", None)


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False
        # self.use_vae = True

NUM_GPUS = 2 if num_gpus is None else int(num_gpus)

class Config(Config):
    def __init__(self, num_gpus=NUM_GPUS)):
        """
        super().__init__(num_gpus=num_gpus)

        # hierarchical_orderer
        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None

        self.learning_rate = 1e-3 * num_gpus
        # self.learning_rate = 2e-4 * num_gpus
        # self.batch_size = 60 * num_gpus
        self.batch_size = int(batch_size) if batch_size else 60 * num_gpus

        # default 500
        self.warmup_steps = 1000  #

        # our changes
        self.val_every = 100
        self.ckpt_every = 100

        # 50 default, we can --resume training
        self.num_epochs = 400

        self.dataloader_module = "deepsvg.svg_dataset"
        self.collate_fn = None  #
        # self.data_dir = "/home/sh/o/unsymbols/DATASETS/deepsvg/full_svg_simplified"             #
        self.data_dir = (
            data_dir
            if data_dir
            else ("/home/sh/o/unsymbols/preprocessing/data/22-0.7-svg-preprocessed/")
        )
        # self.data_dir = "/home/sh/o/unsymbols/preprocessing/data/0.5-svg/"  #
        # self.meta_filepath = "/home/sh/o/unsymbols/DATASETS/deepsvg/meta_full_svg.csv"       #
        self.meta_filepath = Path(self.data_dir) / "meta.csv"
        # (
        #     "/home/sh/o/unsymbols/preprocessing/data/22-0.7-svg-preprocessed/meta.csv"
        # )
        # self.meta_filepath = "/tmp/meta.csv"
