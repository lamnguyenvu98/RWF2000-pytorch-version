from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import NeptuneLogger
from datamodule import RWF2000DataModule
import neptune.new as neptune
from model import FlowGatedNetwork, TrainingModel
from config import read_args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
parser.add_argument('--resume', '-r', default=None, type=str, help='resume checkpoint to continue training process')
ar = parser.parse_args()

args = read_args(ar.config)

val_acc_callback = ModelCheckpoint(
    monitor = 'val_acc',
    dirpath = args.SETTINGS.CHECKPOINT_DIR,
    filename = 'fgn-{epoch:02d}-{val_acc:.2f}-{train_acc:.2f}',
    every_n_epochs = 1,
    save_top_k = 3, # 4 best ckp based on val_acc
    mode = "max",
    save_last=True
)

last_ckp = ModelCheckpoint(
    dirpath = args.SETTINGS.CHECKPOINT_DIR,
    filename = "fgn-lastest-{epoch:02d}",
    every_n_epochs = 1
)

# Initialize neptune AI
run = neptune.init(
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxODUwZmMzMC0yM2FiLTQ0MTctYTJkMi1hZmEzYzM5YWIwY2EifQ==",
    project='lamnguyenvu/RWF2000', 
    # run='RWF-2' # This is to resume last run
)

neptune_logger = NeptuneLogger(
    run=run
)

train_model = TrainingModel(
    lr = args.TRAIN.LEARNING_RATE, 
    momentum = args.TRAIN.MOMENTUM, 
    weight_decay = args.TRAIN.WEIGHT_DECAY, 
    step_size = args.SCHEDULER.STEP_SIZE, 
    gamma = args.SCHEDULER.GAMMA
    )
    
datamodule = RWF2000DataModule(
                dirpath = args.SETTINGS.DATA_DIR,
                target_frames = args.TRAIN.NUM_FRAMES,
                batch_size = args.TRAIN.BATCH_SIZE
            )

trainer = Trainer(
    max_epochs=args.TRAIN.EPOCHS,
    gpus=args.SETTINGS.GPU,
    default_root_dir=args.SETTINGS.LOG_DIR,
    accumulate_grad_batches=args.TRAIN.ACCUMULATE_BATCH,
    precision=args.SETTINGS.PRECISION,
    callbacks=[val_acc_callback, last_ckp],
    logger=neptune_logger
)

# log model summary
# neptune_logger.log_model_summary(model=train_model.model, max_depth=-1)

# # log params
neptune_logger.log_hyperparams(params=dict(train_model.hparams))

if ar.resume is None:
    trainer.fit(train_model, datamodule=datamodule)
else:
    trainer.fit(train_model, datamodule=datamodule, ckpt_path=ar.resume)

run.stop()