"""
Training script for pre-training tasks
"""

import os, torch, torchmanager, torchvision
from torch.nn.utils import prune
from typing import Dict, List

from configs import PretrainingConfig as Config
from data import datasets

if __name__ == "__main__":
    # get configurations
    config = Config.from_arguments()

    # load pretrained model
    model = torchvision.models.resnet50(pretrained=True)

    # prune model
    prunable_params: List[tuple[torch.nn.Module, str]] = [(m, "weight") for m in list(model.modules())[1:-2] if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear)]
    prune.global_unstructured(prunable_params, prune.L1Unstructured, amount=config.pruning_ratio)

    # initialize optimizer, loss function, and metrics
    sgd = torch.optim.SGD(model.parameters(), config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    cross_entropy = torchmanager.losses.CrossEntropy()
    metrics: Dict[str, torchmanager.metrics.Metric] = {'accuracy': torchmanager.metrics.SparseCategoricalAccuracy()}

    # initialize manager
    manager = torchmanager.Manager(model, optimizer=sgd, loss_fn=cross_entropy, metrics=metrics)

    # initialize dataset
    imagenet = datasets.ImageNet(batch_size=config.batch_size)
    training_dataset = imagenet.train_loader
    validation_dataset = imagenet.val_loader

    # initialize checkpoint and data dirs
    experiment_dir = os.path.join("experiments", config.experiment)
    best_ckpt_dir = os.path.join(experiment_dir, "best.model")
    data_dir = os.path.join(experiment_dir, "data")
    last_ckpt_dir = os.path.join(experiment_dir, "last.model")
    os.makedirs(experiment_dir, exist_ok=True)

    # initialize callbacks
    tensorboard_callback = torchmanager.callbacks.TensorBoard(data_dir)
    best_ckpt_callback = torchmanager.callbacks.BestCheckpoint('accuracy', model, best_ckpt_dir)
    last_ckpt_callback = torchmanager.callbacks.Checkpoint(model, last_ckpt_dir)
    callbacks_list: List[torchmanager.callbacks.Callback] = [tensorboard_callback, best_ckpt_callback, last_ckpt_callback]

    # train model
    manager.fit(training_dataset, config.epochs, show_verbose=config.show_verbose, val_dataset=validation_dataset, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list)

    # remove pruning wrap
    for m in manager.model.modules():
        if prune.is_pruned(m) and hasattr(m, "weight"):
            prune.remove(m, "weight")

    # export
    torch.save(manager.model, config.output_model_path)
    summary = manager.test(imagenet.test_loader, use_multi_gpus=config.use_multi_gpus)