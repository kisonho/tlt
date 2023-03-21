import argparse, logging

class Config(argparse.Namespace):
    """Configurations"""
    pruning_ratio: float = 0.8
    disable_dynamic: bool = False
    use_dpf_scheduler: bool = False

    def __init__(self, pruning_ratio: float, disable_dynamic: bool, use_dpf_scheduler: bool) -> None:
        """Constructor"""
        super().__init__()
        self.pruning_ratio = pruning_ratio
        self.disable_dynamic = disable_dynamic
        self.use_dpf_scheduler = use_dpf_scheduler
        assert 0 <= self.pruning_ratio < 1, f"[Argument Error]: The pruning ratio must between [0,1), got {self.pruning_ratio}."

    def show_settings(self) -> None:
        logging.info(f"Pruning settings: pruning_ratio={self.pruning_ratio:.2f}, use_dynamic={not self.disable_dynamic}, scheduler={'DPF' if self.use_dpf_scheduler else 'Constant'}")

    @staticmethod
    def set_arguments(parser: argparse.ArgumentParser) -> None:
        """Get configuration from arguments"""
        pruning_args = parser.add_argument_group("Pruning arguments")
        pruning_args.add_argument("-p", "--pruning_ratio", type=float, required=True, help="A percentage of pruning ratio, must be specified.")
        pruning_args.add_argument("--use_dpf_scheduler", action="store_true", default=False, help="Flag to use DPF scheduler of pruning ratio.")
        pruning_args.add_argument("--disable_dynamic", action="store_true", default=False, help="Flag to disable dynamic pruning.")

    @classmethod
    def from_arguments(cls):
        parser = argparse.ArgumentParser()
        cls.set_arguments(parser)
        arguments = parser.parse_args().__dict__
        return cls(**arguments)