import logging, os, torch, torchmanager, transferring_lottery_ticket as prune
from transferring_lottery_ticket import callbacks, schedulers
from typing import List

import applications, data
from applications.deeplabv3 import DeepLabV3, DeepLabV3Backbone, metrics, scheduler
from configs import SegmentationConfig as Config

if __name__ == "__main__":
    # get config
    config: Config = Config.from_arguments()  # type: ignore

    # load dataset
    dataset = data.load(config.dataset_name, config.batch_size, root_dir=config.dataset_dir)
    training_dataset = dataset.train_loader
    validation_dataset = dataset.val_loader
    testing_dataset = dataset.test_loader

    # initialize checkpoint and data dirs
    experiment_dir = os.path.join("experiments", config.experiment)
    best_ckpt_dir = os.path.join(experiment_dir, "best.model")
    data_dir = os.path.join(experiment_dir, "data")
    init_ckpt_dir = os.path.join(experiment_dir, "init.model")
    last_ckpt_dir = os.path.join(experiment_dir, "last.model")
    os.makedirs(experiment_dir, exist_ok=True)

    # compile manager
    model = applications.deeplabv3(backbone=DeepLabV3Backbone.RESNET50, num_classes=dataset.num_classes)
    optimizer = torch.optim.SGD(
        [
            {"params": model.backbone.parameters(), "lr": 0.1 * config.learning_rate},
            {"params": model.classifier.parameters(), "lr": config.learning_rate},
        ],
        config.learning_rate,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )
    loss_fn = torchmanager.losses.FocalCrossEntropy(ignore_index=255, calculate_average=True)
    metric_fns = {
        "overall_acc": metrics.SegmentationOverallAccuracy(dataset.num_classes),
        "mean_acc": metrics.SegmentationMeanAccuracy(dataset.num_classes),
        "mIoU": metrics.MeanIoU(),
    }
    manager: torchmanager.Manager[DeepLabV3] = torchmanager.Manager(model, optimizer, loss_fn=loss_fn, metrics=metric_fns)

    # save initialized model
    init_ckpt = torchmanager.train.Checkpoint(model)
    init_ckpt.save(0, init_ckpt_dir)

    # initialize lr scheduler
    lr_scheduler = scheduler.MultiStepLR(optimizer, [49, 60], gamma=0.1)

    # pruning
    if config.pruning_ratio > 0:
        # initialize first pruning ratio
        p = 0 if config.use_dpf_scheduler else config.pruning_ratio

        # initialize pruning method
        if prune.is_pruned(model):
            prune.remove(model)
        pruning_method = prune.TransferringLotteryTicket.apply(model.backbone, amount=p, start=2, end=None, is_dynamic=(not config.disable_dynamic))

        # initialize pruning scheduler
        if config.disable_dynamic:
            pruning_scheduler = None
        else:
            pruning_scheduler = schedulers.DPFScheduler(pruning_method, config.pruning_ratio, final_step=60) if config.use_dpf_scheduler else schedulers.ConstantScheduler(pruning_method)
    else:
        pruning_method = pruning_scheduler = None

    # initialize callbacks
    tensorboard_callback = callbacks.TensorBoard(data_dir)
    best_ckpt_callback = callbacks.BestCheckpoint("mIoU", model, best_ckpt_dir)
    last_ckpt_callback = callbacks.LastCheckpoint(model, last_ckpt_dir)
    lr_scheduler_callback = callbacks.LrSchedueler(lr_scheduler, tf_board_writer=tensorboard_callback.writer)
    callbacks_list: List[callbacks.Callback] = [tensorboard_callback, best_ckpt_callback, last_ckpt_callback, lr_scheduler_callback]
    if pruning_scheduler is not None:
        callbacks_list.append(callbacks.PruningRatio(pruning_scheduler))

    # train
    manager.fit(training_dataset, config.epochs, show_verbose=config.show_verbose, val_dataset=validation_dataset, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list)

    # restore best checkpoint
    best_ckpt = torchmanager.train.Checkpoint.from_saved(best_ckpt_dir)

    # remove pruning wrap
    if prune.is_pruned(best_ckpt.model):
        prune.remove(best_ckpt.model)

    # export model
    torch.save(best_ckpt.model, config.output_model_path)
    summary = manager.test(testing_dataset, use_multi_gpus=config.use_multi_gpus)
    logging.info(summary)
