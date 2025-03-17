import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(BaseModel, self).__init__()

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        raise NotImplementedError

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress + torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        return torch.cat(self.get_grads_list())

    def get_grads_list(self):
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads
