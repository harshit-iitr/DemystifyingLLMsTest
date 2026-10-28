"""Special tokens handling (Clean)."""
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
BOS_TOKEN = "<|endoftext|>"
EOS_TOKEN = BOS_TOKEN

SPECIAL_TOKENS = list(dict.fromkeys([PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]))
