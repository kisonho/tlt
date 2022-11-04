import torch, torchmanager
from typing import Any, OrderedDict, Union

from configs import PretrainingTestingConfig as Config
from data import datasets

if __name__ == "__main__":
    # get config
    config = Config.from_arguments()

    # load dataset
    dataset = datasets.ImageNet(config.batch_size).test_loader

    # initialize function
    loss_fn = torchmanager.losses.CrossEntropy()
    metrics = {'accuracy': torchmanager.metrics.SparseCategoricalAccuracy()}

    # initialize manager
    model: Union[torch.nn.Module, OrderedDict[str, Any]] = torch.load(config.model_path)
    if isinstance(model, torch.nn.Module):
        manager = torchmanager.Manager(model, loss_fn=loss_fn, metrics=metrics)
    else:
        ckpt = torchmanager.train.Checkpoint(**model)
        manager = torchmanager.Manager(ckpt.model, loss_fn=loss_fn, metrics=metrics)

    # test
    summary = manager.test(dataset, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    print(f"Final results: loss={summary['loss']:.4f}, acc={summary['accuracy']:.4f}")