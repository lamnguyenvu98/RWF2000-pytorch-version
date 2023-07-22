from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import NeptuneLogger
import neptune
from src.models import FGN
from src.config import read_args
from src.data import RWF2000DataModule
from src.models.callbacks import ModelMetricsCallback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
ar = parser.parse_args()

args = read_args(ar.config)

seed_everything(42, workers=True)

val_acc_callback = ModelCheckpoint(
    monitor = 'val_epoch_accuracy',
    dirpath = args.DIR.CHECKPOINT_DIR,
    filename = 'fgn-{epoch:02d}-{val_epoch_accuracy:.2f}-{train_epoch_accuracy:.2f}',
    every_n_epochs = 1,
    save_top_k = args.VALIDATION.TOP_K, # 4 best ckp based on val_acc
    mode = "max",
    save_last=args.VALIDATION.SAVE_LAST
)

model_metric_callback = ModelMetricsCallback(num_classes = 2, task = "multiclass")

# last_ckp = ModelCheckpoint(
#     dirpath = args.DIR.CHECKPOINT_DIR,
#     filename = "fgn-lastest-{epoch:02d}",
#     every_n_epochs = 1
# )

print()
if (args.NEPTUNE_LOGGER.API_TOKEN is not None) or (args.NEPTUNE_LOGGER.PROJECT is not None):
    # Initialize neptune AI
    run = neptune.init_run(
        api_token=args.NEPTUNE_LOGGER.API_TOKEN,
        project=args.NEPTUNE_LOGGER.PROJECT,
        tags=args.NEPTUNE_LOGGER.TAGS,
        with_id=args.NEPTUNE_LOGGER.WITH_ID # This is to resume last run
    )

    logger = NeptuneLogger(
        run=run
    )

else:
    logger = False

train_model = FGN(
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
    accelerator=args.SETTINGS.ACCELERATOR,
    devices=args.SETTINGS.DEVICES,
    default_root_dir=args.DIR.LOG_DIR,
    accumulate_grad_batches=args.TRAIN.ACCUMULATE_BATCH,
    precision=args.SETTINGS.PRECISION,
    callbacks=[val_acc_callback, model_metric_callback],
    logger=logger
)

# log model summary
# neptune_logger.log_model_summary(model=train_model.model, max_depth=-1)

# # log params
if logger:
    logger.log_hyperparams(params=dict(train_model.hparams))

if args.SETTINGS.RESUME:
    trainer.fit(train_model, datamodule=datamodule, ckpt_path=args.DIR.RESUME_CHECKPOINT)
else:
    trainer.fit(train_model, datamodule=datamodule)