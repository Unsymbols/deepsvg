from configs.deepsvg.default_icons import *


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False
        # self.use_vae = True


class Config(Config):
    def __init__(self, num_gpus=2):
        super().__init__(num_gpus=num_gpus)

        # hierarchical_orderer
        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None

        self.learning_rate = 1e-3 * num_gpus
        # self.learning_rate = 2e-4 * num_gpus
        # self.batch_size = 60 * num_gpus
        self.batch_size = 80

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
            "/home/sh/o/unsymbols/preprocessing/data/22-0.7-svg-preprocessed/"  #
        )
        # self.data_dir = "/home/sh/o/unsymbols/preprocessing/data/0.5-svg/"  #
        # self.meta_filepath = "/home/sh/o/unsymbols/DATASETS/deepsvg/meta_full_svg.csv"       #
        self.meta_filepath = (
            "/home/sh/o/unsymbols/preprocessing/data/22-0.7-svg-preprocessed/meta.csv"
        )
        # self.meta_filepath = "/tmp/meta.csv"
