import argparse, logging, os, torch, torchmanager, tlt as prune
from typing import Any

from .pruning import Config as _PruningConfig


class ClassificationConfig(_PruningConfig):
    """Classification Configurations"""

    dataset_dir: str
    dataset_name: str
    model_path: str
    output_model_path: str

    batch_size: int
    epochs: int
    experiment: str
    learning_rate: float
    show_verbose: bool
    use_multi_gpus: bool
    weight_decay: float

    def __init__(self, dataset_dir: str, dataset_name: str, model_path: str, output_model_path: str, batch_size: int = 256, epochs: int = 182, experiment: str = "test.exp", learning_rate: float = 0.1, show_verbose: bool = False, use_multi_gpus: bool = False, weight_decay: float = 2e-4, **kwargs: Any) -> None:
        """Constructor"""
        super().__init__(**kwargs)
        self.dataset_dir = os.path.normpath(dataset_dir)
        self.dataset_name = dataset_name
        self.model_path = os.path.normpath(model_path)
        self.output_model_path = os.path.normpath(output_model_path)

        self.batch_size = batch_size
        self.epochs = epochs
        self.experiment = experiment
        self.learning_rate = learning_rate
        self.show_verbose = show_verbose
        self.use_multi_gpus = use_multi_gpus
        self.weight_decay = weight_decay

        # initialize log
        log_path = os.path.join("logs", self.experiment.replace(".exp", ".log"))
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(level=logging.INFO, filename=log_path, format="%(message)s")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

        assert self.batch_size > 0, f"Batch size must be positive, got {self.batch_size}."
        assert self.epochs > 0, f"Training epochs must be positive, got {self.epochs}."
        assert self.weight_decay >= 0, f"The weight decay must be a non-negative number, got {self.weight_decay}."
        self.show_settings()

    def show_settings(self) -> None:
        logging.info(f"Experiment {self.experiment}: output_model_path={self.output_model_path}")
        logging.info(f"Dataset '{self.dataset_name}': batch_size={self.batch_size}")
        logging.info(f"Pre-trained model: {self.model_path}")
        logging.info(f"Training settings: epochs={self.epochs}, lr={self.learning_rate}, weight_decay={self.weight_decay}")
        super().show_settings()
        logging.info(f"Device settings: use_multi_gpus={self.use_multi_gpus}")
        logging.info(f"TLT Version: {prune.VERSION}")
        logging.info("---------------------------------------")

    @staticmethod
    def set_arguments(parser: argparse.ArgumentParser) -> None:
        required_args = parser.add_argument_group("Required arguments")
        required_args.add_argument("dataset_name", type=str, help="Dataset name, must be specified.")
        required_args.add_argument("model_path", type=str, help="Pretrained model directory, must be specified.")
        required_args.add_argument("output_model_path", type=str, help="Output file path, must be specified.")

        training_args = parser.add_argument_group("Training arguments")
        training_args.add_argument("-b", "--batch_size", type=int, default=256, help="The number of batch size, default is 256.")
        training_args.add_argument("-e", "--epochs", type=int, default=182, help="The number of epochs, default is 182.")
        training_args.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Initial learning rate, default is 0.1.")
        training_args.add_argument("--weight_decay", type=float, default=2e-4, help="Weight decay of the model, default is 2e-4.")
        training_args.add_argument("-d", "--dataset_dir", type=str, default="~/Documents/Data", help="Dataset root directory, default is '~/Documents/Data'.")

        _PruningConfig.set_arguments(parser)

        exp_args = parser.add_argument_group("Experiment arguments")
        exp_args.add_argument("--experiment", type=str, default="test.exp", help="Name of experiment, default is 'test.exp'.")
        exp_args.add_argument("--show_verbose", action="store_true", default=False, help="Flag to show verbose.")
        exp_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="Flag to use multi gpus.")


class SegmentationConfig(_PruningConfig):
    """Segmentation Configurations"""

    dataset_dir: str
    dataset_name: str
    model_path: str
    output_model_path: str

    batch_size: int
    epochs: int
    experiment: str
    learning_rate: float
    show_verbose: bool
    use_multi_gpus: bool
    warmup_start_lr: float
    warmup_steps: int
    weight_decay: float

    def __init__(self, dataset_dir: str, dataset_name: str, model_path: str, output_model_path: str, batch_size: int = 4, epochs: int = 82, experiment: str = "test.exp", learning_rate: float = 0.01, show_verbose: bool = False, use_multi_gpus: bool = False, warmup_start_lr: float = 5e-4, warmup_steps: int = 1000, weight_decay: float = 1e-4, **kwargs: Any) -> None:
        """Constructor"""
        super().__init__(**kwargs)
        self.dataset_dir = os.path.normpath(dataset_dir)
        self.dataset_name = dataset_name
        self.model_path = os.path.normpath(model_path)
        self.output_model_path = os.path.normpath(output_model_path)
        self.batch_size = batch_size
        self.epochs = epochs
        self.experiment = experiment
        self.learning_rate = learning_rate
        self.show_verbose = show_verbose
        self.use_multi_gpus = use_multi_gpus
        self.warmup_start_lr = warmup_start_lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        # initialize log
        log_path = os.path.join("logs", self.experiment.replace(".exp", ".log"))
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(level=logging.INFO, filename=log_path, format="%(message)s")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

        assert torchmanager.version >= "1.0.3", f"Required torchmanager version >= 1.0.3, got {torchmanager.version}."
        assert self.batch_size > 0, f"Batch size must be positive, got {self.batch_size}."
        assert self.epochs > 0, f"Training epochs must be positive, got {self.epochs}."
        assert self.learning_rate > 0, f"Learning rate must be positive, got {self.learning_rate}."
        assert self.warmup_start_lr > 0, f"Warmup starting learning rate must be positive, got {self.warmup_start_lr}."
        assert self.warmup_steps >= 0, f"Warmup steps must be non-negative, got {self.warmup_steps}."
        assert self.weight_decay >= 0, f"The weight decay must be a non-negative number, got {self.weight_decay}."
        self.show_settings()

    def show_settings(self) -> None:
        logging.info(f"Experiment {self.experiment}: output_model_path={self.output_model_path}")
        logging.info(f"Dataset '{self.dataset_name}': batch_size={self.batch_size}")
        logging.info(f"Pre-trained model: {self.model_path}")
        logging.info(f"Training settings: epochs={self.epochs}, lr={self.learning_rate}, weight_decay={self.weight_decay}")
        super().show_settings()
        logging.info(f"Device settings: use_multi_gpus={self.use_multi_gpus}")
        logging.info(f"TLT Version: {prune.VERSION}")
        logging.info("---------------------------------------")

    @staticmethod
    def set_arguments(parser: argparse.ArgumentParser) -> None:
        """Get configuration from arguments"""
        required_args = parser.add_argument_group("Required arguments")
        required_args.add_argument("dataset_name", type=str, help="Dataset name, must be specified.")
        required_args.add_argument("model_path", type=str, help="Pretrained model directory, must be specified.")
        required_args.add_argument("output_model_path", type=str, help="Output file path, must be specified.")

        training_args = parser.add_argument_group("Training arguments")
        training_args.add_argument("-b", "--batch_size", type=int, default=4, help="The number of batch size, default is 4.")
        training_args.add_argument("-e", "--epochs", type=int, default=82, help="The number of epochs, default is 82.")
        training_args.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Initial learning rate, default is 0.01.")
        training_args.add_argument("--warmup_start_lr", type=float, default=5e-4, help="Initial warmup learning rate, default is 5e-4.")
        training_args.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps, default is 1000.")
        training_args.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay of the model, default is 1e-4.")
        training_args.add_argument("-d", "--dataset_dir", type=str, default="~/Documents/Data", help="Dataset root directory, default is '~/Documents/Data'.")

        _PruningConfig.set_arguments(parser)

        exp_args = parser.add_argument_group("Experiment arguments")
        exp_args.add_argument("--experiment", type=str, default="test.exp", help="Name of experiment, default is 'test.exp'.")
        exp_args.add_argument("--show_verbose", action="store_true", default=False, help="Flag to show verbose.")
        exp_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="Flag to use multi gpus.")


class TestingConfig(argparse.Namespace):
    """Configurations"""

    dataset_name: str
    dataset_dir: str
    model_path: str

    batch_size: int
    show_verbose: bool
    use_multi_gpus: bool

    def __init__(self, dataset_dir: str, dataset_name: str, model_path: str, batch_size: int = 256, show_verbose: bool = False, use_multi_gpus: bool = False) -> None:
        super().__init__()
        self.dataset_dir = os.path.normpath(dataset_dir)
        self.dataset_name = dataset_name
        self.model_path = os.path.normpath(model_path)

        self.batch_size = batch_size
        self.show_verbose = show_verbose
        self.use_multi_gpus = torch.cuda.is_available() if self.use_multi_gpus else self.use_multi_gpus

        assert self.batch_size > 0, f"Batch size must be positive, got {self.batch_size}."

    @classmethod
    def from_arguments(cls):
        """Get configuration from arguments"""
        parser = argparse.ArgumentParser()
        parser.add_argument("dataset_name", type=str, help="Dataset name, must be specified.")
        parser.add_argument("model_path", type=str, help="Target model directory, must be specified.")

        parser.add_argument("-b", "--batch_size", type=int, default=256, help="The number of batch size, default is 256.")
        parser.add_argument("-d", "--dataset_dir", type=str, default="~/Documents/Data/", help="The root directory of dataset, default is '~/Documents/Data'.")
        parser.add_argument("--show_verbose", action="store_true", default=False, help="Flag to show progress bar.")
        parser.add_argument("--use_multi_gpus", action="store_true", default=False, help="Flag to use multi gpus.")
        arguments = parser.parse_args().__dict__
        return cls(**arguments)
