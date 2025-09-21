import datasets
import torch
from preprocessing.common import TensorIterableDataset

class GiftEvalWindowedDataset(TensorIterableDataset):
    def __init__(self, dataset_name: str, file_names: list[str], window_size: int, 
        prediction_depth: int, step: int, pre_batch_timeseries_count: int):
        super().__init__()

        self.dataset_name = dataset_name
        self.file_names = file_names
        self.window_size = window_size
        self.prediction_depth = prediction_depth
        self.step = step
        self.pre_batch_timeseries_count = pre_batch_timeseries_count
        self.dataset = self._build_dataset()

    def _build_dataset(self):
        return (
            datasets.load_dataset(
                self.dataset_name,
                split='train',
                data_files=self.file_names,
                streaming=True,
            )
            .select_columns(["target"])
            .batch(self.pre_batch_timeseries_count)
            .map(self._to_window_tensors)
        )
        
    def _item_to_window_tensor(self, target):
        target_tensor = torch.tensor(target)
        target_tensor_shape = target_tensor.shape

        unfold_dimension = len(target_tensor_shape) - 1 
        
        window_tensor = target_tensor.unfold(dimension=unfold_dimension, size=self.window_size+self.prediction_depth, step=self.step)
        
        window_tensor_shape = window_tensor.shape
        result_tensor = window_tensor if len(window_tensor_shape) == 2 else window_tensor.flatten(0, 1)

        return result_tensor

    def _to_window_tensors(self, batch):
        tensors = [self._item_to_window_tensor(item) for item in batch['target']]
        total_tensor = torch.cat(tensors)
        return {
            'target': total_tensor
        }
        
    def __iter__(self):
        for item in self.dataset:
            total_tensor = item['target']
            X = total_tensor[:, :-self.prediction_depth]
            y = total_tensor[:, -self.prediction_depth:]
            yield X, y
