import abc
import logging
import time
from functools import partial
from typing import Callable, Union, Generator, Literal, Dict, Any

import torch.nn
import torchinfo
from prettytable import PrettyTable
from torch import nn
from torchinfo import summary

from seagull.utils.torch_utils import remove_compiled_model_prefix_from_model_state_dict


class Module(nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.hooks = {"hooks": {}, "outputs": {}}  # dict to store outputs from registered hooks

    def __str__(self) -> str:
        return f"module: {self._get_name()}, num_params: {sum(param.numel() for param in self.parameters())}"

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_weights(self, module: nn.Module) -> None:
        """This method must be overridden by the derived class."""
        pass

    def save_pretrained(self, model_filepath: str) -> None:
        torch.save(self.state_dict(), model_filepath)

    def from_pretrained(self, model_filepath: str) -> None:
        state_dict = torch.load(model_filepath, map_location=self.device)
        state_dict = remove_compiled_model_prefix_from_model_state_dict(state_dict)
        self.load_state_dict(state_dict)

    def from_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.load_state_dict(state_dict)

    def from_checkpoint(self, checkpoint_path: str) -> None:
        assert not hasattr(self, "module"), "DDP model is being used; please use a non-DDP model"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = remove_compiled_model_prefix_from_model_state_dict(checkpoint["model_state_dict"])
        self.from_state_dict(state_dict)

    def from_pretrained_or_checkpoint(self, model_or_checkpoint_path: str) -> None:
        if model_or_checkpoint_path.endswith(".ckpt"):
            self.from_checkpoint(model_or_checkpoint_path)
        elif model_or_checkpoint_path.endswith(".pt"):
            self.from_pretrained(model_or_checkpoint_path)
        else:
            raise ValueError(f"{model_or_checkpoint_path} not supported")

    def summary(self) -> torchinfo.ModelStatistics:
        return summary(self)

    def get_trainable_params(self) -> Generator:
        return (param for param in self.parameters() if param.requires_grad)

    def print_params(self) -> None:
        params_table = PrettyTable(["module", "num_params", "requires_grad"])
        total_trainable_params = 0
        for name, param in self.named_parameters():
            params_table.add_row([name, param.numel(), param.requires_grad])
            if param.requires_grad:
                total_trainable_params = total_trainable_params + param.numel()
        print(params_table)
        if total_trainable_params >= 1e5:
            print(f"total trainable params: {(total_trainable_params / 1e6):0.2f}M")
        else:
            print(f"total trainable params: {total_trainable_params}")

    def _get_hook_name(self, hook: Union[Callable, partial[Callable]]) -> str:
        try:
            hook_name = hook.__name__
        except AttributeError:
            # Partial functions.
            hook_name = hook.func.__name__

        if hook_name in self.hooks["hooks"]:
            hook_name = hook_name + "_" + str(int(time.time()))
            logging.warning(f"another hook with the same name exists, using {hook_name} instead")
        return hook_name

    def attach_hook(
        self, module: nn.Module, hook: Union[Callable, partial[Callable]], hook_type: Literal["forward", "backward"]
    ) -> None:
        hook_name = self._get_hook_name(hook)
        if hook_type == "forward":
            self.hooks["hooks"][hook_name] = module.register_forward_hook(hook)
        elif hook_type == "backward":
            self.hooks["hooks"][hook_name] = module.register_full_backward_hook(hook)

    def detach_hook(self, hook: Callable) -> None:
        logging.warning(f"detach_hook doesn't remove the output from self.hooks['outputs']")
        hook_name = self._get_hook_name(hook)
        self.hooks["hooks"][hook_name].remove()
        self.hooks["hooks"].pop(hook_name)

    def detach_all_hooks(self) -> None:
        for handle in self.hooks["hooks"].values():
            handle.remove()
        self.hooks = {"hooks": {}, "outputs": {}}

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError
