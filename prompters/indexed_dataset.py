from torch.utils.data.dataset import Dataset


class IndexedDataset(Dataset):
    def __init__(self, data, prompter):
        self.data = data
        self.prompter = prompter

    def to_multi_inputs(self, datapoint,):
        result = {}
        if "reasoning" in datapoint:
            datapoint = self.prompter.to_chat_example(datapoint)
            user_inputs = self.prompter.totensor_multiple(datapoint,)
            result.update(user_inputs)
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        element = self.data[idx]
        train_data_id = idx
        return element | {"train_data_id": train_data_id}

    def get_with_profiles(self, idx, reasoning_str):
        element = self.data[idx]
        multi_inputs = self.to_multi_inputs(
            element | {"reasoning": reasoning_str},)

        return multi_inputs
