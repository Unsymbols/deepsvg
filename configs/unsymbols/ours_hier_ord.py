from configs.deepsvg.default_icons import *


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False


class Config(Config):
    def __init__(self, num_gpus=2):
        super().__init__(num_gpus=num_gpus)

        # hierarchical_orderer
        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None

        self.learning_rate = 1e-3 * num_gpus
        self.batch_size = 60 * num_gpus

        # our changes
        self.val_every = 100
        self.ckpt_every = 100

        self.dataloader_module = "deepsvg.svg_dataset"  #
        self.collate_fn = None                                #
        self.data_dir = "/home/sh/o/unsymbols/DATASETS/deepsvg/cyl_lat_can_simplified"             #
        self.meta_filepath = "/home/sh/o/unsymbols/DATASETS/deepsvg/meta_cyr_lat_can.csv"       #