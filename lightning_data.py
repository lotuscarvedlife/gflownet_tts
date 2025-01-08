from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
import warnings
import torch
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text
)

warnings.filterwarnings("ignore", ".*does not have many workers.*")

# 准备 prompt 数据
class PromptDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        audio_tokenizer,
        text_tokenizer,
        phn2num,
        train_size=0.95,
        limit_prompts=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["audio_tokenizer", "text_tokenizer", "phn2num"])
        self.audio_tokenizer = audio_tokenizer
        self.text_tokenizer = text_tokenizer
        self.phn2num = phn2num
        # self.data_path = data_path 
        self.train_data = None
        self.val_data = None
    
    # 设置函数
    def setup(self, stage):
        # 读取 prompt 数据文件
        with open(self.hparams.data_path, "r") as f:
            prompts = f.readlines()
        # 去除每行末尾的换行符
        prompts = [line.rstrip("\n") for line in prompts]
        # 限制提示数量，取前1000个提示
        if self.hparams.limit_prompts is not None:
            prompts = prompts[: self.hparams.limit_prompts]
        # 计算训练集数量
        num_train = int(len(prompts) * self.hparams.train_size)
        # 划分对应的训练集和验证集，并生成对应的 pipe 实例
        # TODO: 进行数据集适配
        self.train_data = PromptDataPipe(prompts[:num_train], self.audio_tokenizer, self.text_tokenizer, self.phn2num, self.hparams.data_path)
        self.val_data = PromptDataPipe(prompts[num_train:], self.audio_tokenizer, self.text_tokenizer, self.phn2num, self.hparams.data_path)

    # 返回训练集对应的 dataloader，随机打乱顺序，每次返回一个样本，加载的工作线程数默认为0
    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=None, num_workers=0)

    # 返回验证集对应的 dataloader，每次返回一个样本
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None, num_workers=0)

# prompt 的数据 pipe 类，返回编码后每个句子的 token id 列表
# prompts 为包含多个句子的列表
# TODO: 进行数据集适配
class PromptDataPipe(Dataset):
    def __init__(self, prompts, audio_tokenizer, text_tokenizer, phn2num, data_path) -> None:
        super().__init__()
        self.audio_tokenizer = audio_tokenizer
        self.text_tokenizer = text_tokenizer
        self.phn2num = phn2num
        self.prompts = prompts
        self.data_path = data_path

    def __len__(self):
        return len(self.prompts)

    # 会返回对应索引的句子经过 tokenizer 处理后的结果，返回的类型为 pt（即pytorch张量），返回其中的 token id 列表。
    def __getitem__(self, index):
        prompt = self.prompts[index]
        prompt_audio_path, prompt_text = prompt.split(maxsplit=1)
        prompt_audio_path = "/".join(self.data_path.split("/")[:2]+["audio/"+prompt_audio_path+".flac"])
        text_tokens = [self.phn2num[phn] for phn in
            tokenize_text(
                self.text_tokenizer, text=prompt_text.strip()
            ) if phn in self.phn2num
        ]
        text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
        audio_tokens = tokenize_audio(
            tokenizer=self.audio_tokenizer,
            audio_path=prompt_audio_path
        )[0][0] # [1, codebook, time]

        # if audio_tokens.shape[2] == 793 and torch.all(audio_tokens[0,0,-3:] == torch.tensor([131,825,433]).to(audio_tokens.device)):
        #     print(prompt_audio_path)

        return {"x_encoded": text_tokens,
                "y_encoded": audio_tokens}
        # prompt = self.tokenizer(
        #     self.prompts[index],
        #     return_tensors="pt",
        # )["input_ids"]
        return prompt
