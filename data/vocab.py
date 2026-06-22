from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        
        l=sorted(set(text))
        stoi={}
        itos={}
        for i,j in enumerate(l):
            itos[i]=j
            stoi[j]=i
        return (stoi,itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        l=[]
        for i in text:
            l.append(stoi[i])
        return l

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        s=""
        for i in ids:
            s=s+itos[i]
        return s
       