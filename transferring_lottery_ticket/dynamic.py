from torch.nn.parameter import Parameter
from torch.nn.utils.prune import L1Unstructured, global_unstructured, is_pruned, remove as _remove
from torchmanager_core import abc, torch
from torchmanager_core.typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from torchmanager_core.view import logging

def _get_prunable_params(modules: Iterable[torch.nn.Module], name: str = "weight", skip_types: List[Type[torch.nn.Module]] = [torch.nn.BatchNorm2d]) -> List[Tuple[torch.nn.Module, str]]:
    """
    Method to get available prunable parameters in a `torch.nn.Module`

    - Parameters:
        - modules: A `list` of target `torch.nn.Module`
        - name: A `str` of the name of parameter
        - skip_types: A `list` of skipped prunable types
    - Returns: A `list` of `tuple` with `torch.nn.Module` and its target weight name in `str`
    """
    # initialize finding
    target_params: List[Tuple[torch.nn.Module, str]] = []

    # loop for modules
    for m in modules:
        if hasattr(m, name) and not type(m) in skip_types:
            target_params.append((m, name))
    return target_params

class DynamicPruningMethod(abc.ABC):
    """
    A basic dynamic pruning method

    * implements `.schedulers.DynamicPruning`

    - Properties:
        - amount: Either an `int` of total numbers of weights to be pruned, or a `float` of percentage of weights to be pruned
        - compressible_modules: A `list` of `torch.nn.Module` to be compressed
        - is_dynamic: A `bool` flag to control if using dynamic pruning to update the masks. With `False` flag, the method `compute_mask` will not update current masks.
    """
    # properties
    amount: Union[float, int]
    is_dynamic: bool

    @property
    @abc.abstractmethod
    def compressible_modules(self) -> List[torch.nn.Module]:
        raise NotImplementedError

    def __init__(self, amount: Union[int, float] = 0.8, is_dynamic: bool = True) -> None:
        """
        Constructor

        - Parameters:
            - amount: Either an `int` of total numbers of weights to be pruned, or a `float` of percentage of weights to be pruned
            - is_dynamic: A `bool` flag of if using dynamic pruning
        """
        self.amount = amount
        assert amount >= 0, "[Argument Error]: The pruning amount must be non-negative, got {}.".format(amount)
        self.is_dynamic = is_dynamic

    @classmethod
    @abc.abstractmethod
    def apply(cls, model: torch.nn.Module, amount: Union[int, float], name: str = "weight", is_dynamic: bool = True):
        """
        Applies GlobalL1Unstructured pruning method to a model

        - Parameters:
            - model: A `torch.nn.Module` to be pruned
            - amount: A `int` or a `float` of pruning amount
            - name: A `str` of parameter name
            - is_dynamic: A `bool` flag of if using dynamic pruning
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prune(self, target_params: List[Tuple[torch.nn.Module, str]], amount: Union[int, float]) -> None:
        raise NotImplementedError

    def remove(self, name: str = "weight") -> None:
        """
        Remove the compression wrap in compressible modules

        - Parameters:
            - name: A `str` of the name of parameter to be removed
        """
        for m in self.compressible_modules:
            if is_pruned(m):
                remove(m, name)
                self.compressible_modules.remove(m)

    def update_mask(self, new_amount: Optional[Union[int, float]] = None, name: str = "weight") -> List[torch.Tensor]:
        """
        Recalculate mask along all the model

        - Parameters:
            - new_amount: An optional `int` or `float` of new pruning amount
            - name: A `str` of the name of parameters in compressible modules
        - Returns: A `list` of the masks for parameters
        """
        # return when not dynamic pruning
        if self.is_dynamic is not True:
            logging.warn("[Pruning Warning]: Dynamic pruning disabled.", RuntimeWarning)
            return []

        # check new amount value
        if new_amount is not None:
            self.amount = new_amount
            assert new_amount >= 0, "[Argument Error]: The pruning amount must be non-negative, got {}.".format(
                new_amount)

        # loop for all compressible modules
        with torch.no_grad():
            for m in self.compressible_modules:
                # recover orig weight
                assert is_pruned(m), f'Module is not pruned: {m}'
                orig: torch.Tensor = getattr(m, f"{name}_orig")
                orig = orig.clone()
                remove(m, name)
                var: Parameter = getattr(m, f"{name}")
                var.copy_(orig)

                # reapplying pruning locally
                if new_amount is None:
                    mask: torch.Tensor = getattr(m, f"{name}_mask")
                    amount = int(mask.eq(0).sum())
                    self.prune([(m, name)], amount)

        # reapplying pruning globally
        if new_amount is not None:
            target_params = _get_prunable_params(self.compressible_modules, name=name)
            self.prune(target_params, self.amount)
        return [getattr(m, f"{name}_mask") for m in self.compressible_modules]

class TransferringLotteryTicket(DynamicPruningMethod):
    """
    The Dynamic Pruning with Feedback pruning method

    * https://arxiv.org/abs/2006.07253

    - Properties:
        - compressible_modules: A `list` of the target `torch.nn.Module`
    """
    __compressible_modules: List[torch.nn.Module]

    @property
    def compressible_modules(self) -> List[torch.nn.Module]:
        return self.__compressible_modules

    def __init__(self, compressible_modules: List[torch.nn.Module], amount: Union[int, float], is_dynamic: bool = True) -> None:
        """
        Constructor

        - Parameters:
            - compressible_modules: A `list` of the target `torch.nn.module` to be pruned
            - amount: Either an `int` of total numbers of weights to be pruned, or a `float` of percentage of weights to be pruned
            - is_dynamic: A `bool` flag of if using dynamic pruning
        """
        super().__init__(amount, is_dynamic)
        self.__compressible_modules = compressible_modules

    @classmethod
    def apply(cls, model: torch.nn.Module, amount: Union[int, float], name: str = "weight", is_dynamic: bool = True, start: Optional[int] = 2, end: Optional[int] = -2):
        """
        Applies GlobalL1Unstructured pruning method to a model

        - Parameters:
            - model: A `torch.nn.Module` to be pruned
            - amount: A `int` or a `float` of pruning amount
            - name: A `str` of parameter name
            - is_dynamic: A `bool` flag of if using dynamic pruning
            - start: An `int` of the first module to prune
            - end: An `int` of the last module to prune
        """
        # prune model
        target_params = _get_prunable_params(list(model.modules())[start:end], name=name)
        global_unstructured(target_params, STEL1Unstructured if is_dynamic else L1Unstructured, amount=amount)
        compressible_modules = [m for m, _ in target_params]
        return cls(compressible_modules, amount, is_dynamic=is_dynamic)

    def prune(self, target_params: List[Tuple[torch.nn.Module, str]], amount: Union[int, float]) -> None:
        return global_unstructured(target_params, STEL1Unstructured, amount=amount)

class STEMaskOp(torch.autograd.Function):
    """The STE Mask operation"""
    apply: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad, None

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x * mask

class STEL1Unstructured(L1Unstructured):
    """The L1 Unstructured pruning method with Straight-Throw Estimator (STE)"""
    def apply_mask(self, module: torch.nn.Module) -> torch.Tensor:
        assert self._tensor_name is not None, f"Module {module} has to be pruned"
        mask: torch.Tensor = getattr(module, self._tensor_name + "_mask")
        orig: torch.Tensor = getattr(module, self._tensor_name + "_orig")
        pruned_tensor = STEMaskOp.apply(orig, mask.to(dtype=orig.dtype))
        return pruned_tensor

def remove(model: torch.nn.Module, name: str = "weight") -> None:
    for m in model.modules():
        if is_pruned(m) and hasattr(m, name):
            _remove(m, name)