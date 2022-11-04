import torch, torchmanager
from typing import Any, OrderedDict, Union

import data
from applications.deeplabv3 import metrics
from configs import TestingConfig as Config

if __name__ == "__main__":
    # get config
    config = Config.from_arguments()

    # load dataset
    dataset = data.load(config.dataset_name, config.batch_size, root_dir=config.dataset_dir)
    steps_per_epoch = dataset.steps_per_epoch

    # initialize function
    loss_fn = torchmanager.losses.CrossEntropy(ignore_index=255, reduction='mean')
    metric_fns = {
        'overall_acc': metrics.SegmentationOverallAccuracy(dataset.num_classes),
        'mean_acc': metrics.SegmentationMeanAccuracy(dataset.num_classes),
        'mIoU': metrics.MeanIoU()
    }

    # initialize manager
    model: Union[torch.nn.Module, OrderedDict[str, Any]] = torch.load(config.model_path)
    if isinstance(model, torch.nn.Module):
        manager = torchmanager.Manager(model, loss_fn=loss_fn, metrics=metric_fns)
    else:
        ckpt = torchmanager.train.Checkpoint(**model)
        manager = torchmanager.Manager(ckpt.model, loss_fn=loss_fn, metrics=metric_fns)

    # test
    summary = manager.test(dataset.test_loader, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    print(f"Final results: loss={summary['loss']:.4f}, overall_acc={summary['overall_acc']:.4f}, mean_acc={summary['mean_acc']:.4f}, mIoU={summary['mIoU']:.4f}")