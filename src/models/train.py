from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import NeptuneLogger, TensorBoardLogger, MLFlowLogger
import neptune
from src.models import FGN
from src.utils import read_args
from src.data import RWF2000DataModule
from src.models.callbacks import ModelMetricsCallback
import argparse
import sys

parser = argparse.ArgumentParser(
    prog="Training FGN model",
    epilog="Execute <command> --help for more information. Execute <command> <sub-commands> --help for more information about sub-command"
)

required = parser.add_argument_group("required arguments")
required.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")

optional = parser.add_argument_group("optional arguments", description="Optional arguments for loggers: Neptune AI, MLFlow, TensorBoard")

optional.add_argument('--loggers', 
                      required=False, 
                      choices=["neptune", "tensorboard", "mlflow"], 
                      type=str,
                      help="Log to neptune AI")
optional.add_argument('--log-config', 
                      required="--loggers" in sys.argv,
                      type=str,
                      help="Path to log config")
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

logger = False

if ar.logger is None:
    logger = False
else:
    log_args = read_args(ar.log_config)
    if ar.logger == "neptune":
        # Initialize neptune AI
        run = neptune.init_run(
            api_token=log_args.API_TOKEN,
            project=log_args.PROJECT,
            tags=log_args.TAGS,
            with_id=log_args.WITH_ID # This is to resume last run 
        )

        logger = NeptuneLogger(
            run=run,
            log_model_checkpoints=log_args.LOG_MODEL_CHECKPOINT,
            prefix=log_args.PREFIX
        )
    
    elif ar.logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=log_args.SAVE_DIR,
            name=log_args.NAME,
            version=log_args.VERSION,
            log_graph=log_args.LOG_GRAPH,
            prefix=log_args.PREFIX,
            sub_dir=log_args.SUB_DIR
        )
        
    elif ar.logger == "mlflow":
        logger = MLFlowLogger(
            experiment_name=log_args.EXPERIMENT_NAME,
            run_name=log_args.RUN_NAME,
            tags=log_args.TAGS,
            save_dir=log_args.SAVE_DIR,
            log_model=log_args.LOG_MODEL,
            prefix=log_args.PREFIX,
            artifact_location=log_args.ARTIFACT_LOCATION,
            run_id=log_args.RUN_ID
        )
        

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
    logger=logger,
    log_every_n_steps=args.LOGGER.LOG_N_STEP
)

# # log params
# if logger:
#     logger.log_hyperparams(params=dict(train_model.hparams))

def main():
    if args.SETTINGS.RESUME:
        trainer.fit(train_model, datamodule=datamodule, ckpt_path=args.DIR.RESUME_CHECKPOINT)
    else:
        trainer.fit(train_model, datamodule=datamodule)

if __name__ == '__main__':
    main()
