from configs.deepsvg.default_icons import *
from pathlib import Path
from pprint import pp

# if DATA_DIR env variable is set, use it
import os

data_dir = os.environ.get("DSVG_TRAIN_DATA_DIR", None)
batch_size = os.environ.get("DSVG_TRAIN_BS", None)
num_gpus = os.environ.get("DSVG_TRAIN_NUM_GPUS", None)
lr = os.environ.get("DSVG_TRAIN_LR", None) #(that is then multiplied by num_gpus)
warmup_steps = os.environ.get("DSVG_TRAIN_WARMUP_STEPS", None)
checkpoint_and_val_every = os.environ.get("DSVG_TRAIN_CKPT_VAL_EVERY", None)

print("==")
interesting_envs = [x for x in os.environ.keys() if "DSVG_TRAIN" in x]
print("Interesting environment variables found:")
for k in interesting_envs:
    print(f"\t{k}: {os.environ[k]}")
print("==")

class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False
        # self.use_vae = True


NUM_GPUS = 2 if num_gpus is None else int(num_gpus)


class Config(Config):
    def __init__(self, num_gpus=NUM_GPUS):
        super().__init__(num_gpus=num_gpus)

        # hierarchical_orderer
        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None

        env_learning_rate = float(lr) if lr else 1e-3
        self.learning_rate = env_learning_rate * num_gpus
        # self.learning_rate = 2e-4 * num_gpus
        # self.batch_size = 60 * num_gpus
        self.batch_size = int(batch_size) if batch_size else 60 * num_gpus

        # default 500
        self.warmup_steps = int(warmup_steps) if warmup_steps else 1000

        # our changes
        env_checkpoint_and_val_every = (int(checkpoint_and_val_every) if checkpoint_and_val_every else 1000)
        self.val_every = env_checkpoint_and_val_every
        self.ckpt_every = env_checkpoint_and_val_every

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
