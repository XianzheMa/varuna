import torch
from torch.utils.data import Dataset


class HuggingFaceDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, tokenizer, seq_length):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        curr_tokens = self.tokenizer(self.hf_dataset[idx]['text'])['input_ids']
        # pad to seq_length
        if len(curr_tokens) < self.seq_length:
            curr_tokens += [self.tokenizer.pad_token_id] * (self.seq_length - len(curr_tokens))
        else:
            curr_tokens = curr_tokens[:self.seq_length]
        input_ids = torch.tensor(curr_tokens)
        return input_ids, input_ids.detach().clone()



