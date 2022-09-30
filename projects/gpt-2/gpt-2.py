import os
import sys
import torch
from mingpt.model import GPT
from mingpt.bpe import get_encoder, BPETokenizer
from mingpt.trainer import Trainer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 1024
        return C

    def __init__(self, config, data):
        self.encoder = get_encoder()
        self.config = config

        encoded = self.encoder.encode(data)
        data_size = len(encoded)
        vocab_size = len(self.encoder.encoder)
        print('data: %d tokens, vocab_size: %d' % (data_size, vocab_size))

        self.vocab_size = vocab_size
        self.data = encoded

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def get_item(self, idx):
        chunk = self.data[idx:idx + self.config.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        return self.get_item(idx)

# -----------------------------------------------------------------------------

print('loading data')
text = open('input.bak.txt', 'r').read()
print('data loaded')

print('preparing dataset')
train_dataset = CharDataset(CharDataset.get_default_config(), text)

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = train_dataset.vocab_size
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

train_config = Trainer.get_default_config()
# with this you can train your model on 8GB VRAM, as far i know OpenAI used 512
train_config.batch_size = 1
trainer = Trainer(train_config, model, train_dataset)

tokenizer = BPETokenizer()

def batch_end_callback(trainer):
    if trainer.iter_num % 10 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

    if trainer.iter_num % 500 == 0:
        # evaluate both the train and test score
        model.eval()
        with torch.no_grad():
            # sample from the model...
            context = "This is our starter text"
            x = tokenizer(context).to(trainer.device)
            y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
            decoded = tokenizer.decode(y)
            print(decoded)
        # save the latest model
        print("saving model")
        ckpt_path = os.path.join('./', "model.pt")
        torch.save(model.state_dict(), ckpt_path)
        # revert model to training mode
        model.train()

trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()
