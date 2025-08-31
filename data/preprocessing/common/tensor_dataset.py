from torch.utils.data import IterableDataset
from torch import Tensor

# this is the only type of datasets we are going to work with
class TensorIterableDataset(IterableDataset[Tensor]):
    pass
