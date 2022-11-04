from torchmanager_core import abc, torch
from torchmanager_core.typing import List, Optional, Protocol, Union


class DynamicPruning(Protocol):
    @property
    @abc.abstractmethod
    def amount(self) -> Union[float, int]:
        return NotImplemented

    @abc.abstractmethod
    def update_mask(self, new_amount: Optional[Union[float, int]] = None, name: str = "weight") -> List[torch.Tensor]:
        pass
