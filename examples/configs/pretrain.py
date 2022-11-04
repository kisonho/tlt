import argparse, logging, os, torch

class Config(argparse.Namespace):
    '''The config class'''
    output_model_path: str

    batch_size: int
    epochs: int
    experiment: str
    learning_rate: float
    pruning_ratio: float
    show_verbose: bool
    use_multi_gpus: bool
    weight_decay: float

    def __init__(self, output_model_path: str, batch_size: int = 128, epochs: int = 10, experiment: str = "prune", learning_rate: float = 1e-4, pruning_ratio: float = 0.8, show_verbose: bool = False, use_multi_gpus: bool = False, weight_decay: float = 1e-4) -> None:
        '''
        Constructor
        
        - Parameters:
            - output_model_path: A `str` of output model path
        '''
        super().__init__()
        self.output_model_path = os.path.normpath(output_model_path)
        self.batch_size = batch_size
        self.epochs = epochs
        self.experiment = experiment
        self.learning_rate = learning_rate
        self.pruning_ratio = pruning_ratio
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

        # arguments assertion
        assert self.batch_size > 0, f'Batch size must be positive, got {self.batch_size}.'
        assert self.epochs > 0, f'Epochs must be positive, got {self.epochs}.'
        assert self.learning_rate > 0, f'Learning rate must be positive, got {self.learning_rate}.'
        assert self.pruning_ratio > 0 and self.pruning_ratio < 1, f"Pruning ratio must between (0,1), got {self.pruning_ratio}."
        assert self.weight_decay > 0, f'Weight decay must be positive, got {self.weight_decay}.'
        self.show_settings()

    @classmethod
    def from_arguments(cls):
        '''Get configurations from arguments'''
        parser = argparse.ArgumentParser()
        parser.add_argument("output_model_path", type=str, help="The output model path, must be specified")

        training_args = parser.add_argument_group("Training arguments")
        training_args.add_argument("-e", "--epochs", type=int, default=10, help="The number of epochs, default is 10.")
        training_args.add_argument("-b", "--batch_size", type=int, default=128, help="The number of batch size, default is 128.")
        training_args.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="The learning rate, default is 1e-4.")
        training_args.add_argument("--weight_decay", type=float, default=1e-4, help="The l2 weight decay rate, default is 1e-4.")
        training_args.add_argument("-p", "--pruning_ratio", type=float, required=True, help="The pruning ratio, must be specified.")

        exp_args = parser.add_argument_group("Experiment arguments")
        exp_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="The flag to use multi-gpus.")
        exp_args.add_argument("--show_verbose", action="store_true", default=False, help="The flag to show progress bar.")
        exp_args.add_argument("--experiment", type=str, default="prune", help="The experiment name, default is \'prune\'.")
        arguments = parser.parse_args().__dict__
        return cls(**arguments)

    def show_settings(self) -> None:
        logging.info(f"Experiment {self.experiment}: output_model_path={self.output_model_path}")
        logging.info(f"Dataset: batch_size={self.batch_size}")
        logging.info(f"Training settings: epochs={self.epochs}, lr={self.learning_rate}, weight_decay={self.weight_decay}")
        logging.info(f"Pruning settings: pruning_ratio={self.pruning_ratio:.2f}")
        logging.info(f"Device settings: use_multi_gpus={self.use_multi_gpus}")
        logging.info("---------------------------------------")

class TestingConfig(argparse.Namespace):
    """Configurations"""
    batch_size: int = 128
    model_path: str
    show_verbose: bool = False
    use_multi_gpus: bool = False

    def __init__(self, model_path: str, batch_size: int = 128, show_verbose: bool = False, use_multi_gpus: bool = False) -> None:
        super().__init__()
        self.model_path = os.path.normpath(model_path)

        self.batch_size = batch_size
        self.show_verbose = show_verbose
        self.use_multi_gpus = torch.cuda.is_available() if use_multi_gpus else False

        assert self.batch_size > 0, f"Batch size must be positive, got {self.batch_size}."

    @classmethod
    def from_arguments(cls):
        """Get configuration from arguments"""
        parser = argparse.ArgumentParser()
        parser.add_argument("model_path", type=str, help="Target model directory, must be specified.")

        parser.add_argument("-b", "--batch_size", type=int, default=128, help="The number of batch size, default is 128.")
        parser.add_argument("--show_verbose", action="store_true", default=False, help="Flag to show progress bar.")
        parser.add_argument("--use_multi_gpus", action="store_true", default=False, help="Flag to use multi gpus.")
        arguments = parser.parse_args().__dict__
        return cls(**arguments)