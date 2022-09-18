from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import NeptuneLogger
from datamodule import RWF2000DataModule
import neptune.new as neptune
from model import FlowGatedNetwork, FlowGatedNetworkV2, TrainingModel
from config import read_args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
ar = parser.parse_args()

args = read_args(ar.config)

seed_everything(42, workers=True)

val_acc_callback = ModelCheckpoint(
    monitor = 'val_acc',
    dirpath = args.DIR.CHECKPOINT_DIR,
    filename = 'fgn-{epoch:02d}-{val_acc:.2f}-{train_acc:.2f}',
    every_n_epochs = 1,
    save_top_k = 3, # 4 best ckp based on val_acc
    mode = "max",
    save_last=True
)

# last_ckp = ModelCheckpoint(
#     dirpath = args.DIR.CHECKPOINT_DIR,
#     filename = "fgn-lastest-{epoch:02d}",
#     every_n_epochs = 1
# )

# Initialize neptune AI
run = neptune.init(
    api_token="#",
    project='lamnguyenvu/RWF2000',
    tags=['rwf2000', 'flow-gated-net', 'ver3', 'relu']
    # run='RWF-2' # This is to resume last run
)

neptune_logger = NeptuneLogger(
    run=run
)

train_model = TrainingModel(
    learning_rate = args.TRAIN.LEARNING_RATE, 
    momentum = args.TRAIN.MOMENTUM, 
    weight_decay = args.TRAIN.WEIGHT_DECAY, 
    step_size = args.SCHEDULER.STEP_SIZE, 
    gamma = args.SCHEDULER.GAMMA
    )
    
datamodule = RWF2000DataModule(
                dirpath = args.DIR.DATA_DIR,
                target_frames = args.TRAIN.NUM_FRAMES,
                batch_size = args.TRAIN.BATCH_SIZE
            )

trainer = Trainer(
    max_epochs=args.TRAIN.EPOCHS,
    gpus=args.SETTINGS.GPU,
    default_root_dir=args.DIR.LOG_DIR,
    accumulate_grad_batches=args.TRAIN.ACCUMULATE_BATCH,
    precision=args.SETTINGS.PRECISION,
    callbacks=[val_acc_callback],
    logger=neptune_logger
)

# log model summary
# neptune_logger.log_model_summary(model=train_model.model, max_depth=-1)

# # log params
neptune_logger.log_hyperparams(params=dict(train_model.hparams))

if args.SETTINGS.RESUME:
    trainer.fit(train_model, datamodule=datamodule, ckpt_path=args.DIR.RESUME_CHECKPOINT)
else:
    trainer.fit(train_model, datamodule=datamodule)