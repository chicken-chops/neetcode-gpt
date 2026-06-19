import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        pass
        vocab=set()

        for sentence in positive + negative:
            for word in sentence.split():
                vocab.add(word)
        
        print(vocab)
        vocab = sorted(vocab)

        print(vocab)
        dic = {}

        for i, word in enumerate(vocab):
            dic[word] = i + 1
        
        print(dic)
        tensors = []

        for sentence in positive + negative:
            encoded = []

            for word in sentence.split():
                encoded.append(dic[word])

            tensors.append(torch.tensor(encoded))

        padded = nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True
        )

        return padded
