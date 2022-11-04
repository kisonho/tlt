import logging, os, torch, torchmanager, transferring_lottery_ticket as prune
from transferring_lottery_ticket import callbacks, schedulers
from typing import List

import applications, data
from configs import ClassificationConfig as Config

if __name__ == "__main__":
    # get config
    config: Config = Config.from_arguments()

    # load dataset
    dataset = data.load(config.dataset_name, config.batch_size, root_dir=config.dataset_dir)
    steps_per_epoch = dataset.steps_per_epoch

    # initialize checkpoint and data dirs
    experiment_dir = os.path.join("experiments", config.experiment)
    best_ckpt_dir = os.path.join(experiment_dir, "best.model")
    data_dir = os.path.join(experiment_dir, "data")
    init_ckpt_dir = os.path.join(experiment_dir, "init.model")
    last_ckpt_dir = os.path.join(experiment_dir, "last.model")
    os.makedirs(experiment_dir, exist_ok=True)

    # compile manager
    model = applications.resnet50(config.model_path, num_classes=dataset.num_classes, input_channels=dataset.input_channels)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    loss_fn = torchmanager.losses.CrossEntropy()
    metric_fns = {'accuracy': torchmanager.metrics.SparseCategoricalAccuracy()}
    manager = torchmanager.Manager(model, optimizer, loss_fn=loss_fn, metrics=metric_fns)

    # save initialized model
    init_ckpt = torchmanager.train.Checkpoint(model)
    init_ckpt.save(0, init_ckpt_dir)

    # initialize lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [91, 136], 0.1, verbose=True)

    if config.pruning_ratio > 0:
        # initialize first pruning ratio
        p = 0 if config.use_dpf_scheduler else config.pruning_ratio

        # initialize pruning method
        if prune.is_pruned(model):
            prune.remove(model)
        pruning_method = prune.TransferringLotteryTicket.apply(model, amount=p, is_dynamic=(not config.disable_dynamic))

        # initialize pruning scheduler
        pruning_scheduler = schedulers.DPFScheduler(pruning_method, config.pruning_ratio, final_step=136) if config.use_dpf_scheduler else schedulers.ConstantScheduler(pruning_method)
    else:
        # pruning method and pruning scheduler are None if pruning ratio is 0
        pruning_method = pruning_scheduler = None

    # initialize callbacks
    tensorboard_callback = callbacks.TensorBoard(data_dir)
    best_ckpt_callback = callbacks.BestCheckpoint('accuracy', model, best_ckpt_dir)
    last_ckpt_callback = callbacks.LastCheckpoint(model, last_ckpt_dir)
    callbacks_list: List[callbacks.Callback] = [tensorboard_callback, best_ckpt_callback, last_ckpt_callback]
    if pruning_scheduler is not None and not config.disable_dynamic:
        callbacks_list.append(callbacks.PruningRatio(pruning_scheduler))

    # train
    manager.fit(dataset.train_loader, config.epochs, lr_scheduler=lr_scheduler, show_verbose=config.show_verbose, val_dataset=dataset.val_loader, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list)

    # restore best checkpoint
    best_ckpt = torchmanager.train.Checkpoint.from_saved(best_ckpt_dir)

    # remove pruning wrap
    if prune.is_pruned(best_ckpt.model):
        prune.remove(best_ckpt.model)

    # export model
    torch.save(best_ckpt.model, config.output_model_path)
    summary = manager.test(dataset.test_loader, use_multi_gpus=config.use_multi_gpus)
    logging.info(summary)