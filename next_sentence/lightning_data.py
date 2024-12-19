from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

# 准备 prompt 数据
class PromptDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer,
        train_size=0.95,
        limit_prompts=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        self.tokenizer = tokenizer
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
        self.train_data = PromptDataPipe(prompts[:num_train], self.tokenizer)
        self.val_data = PromptDataPipe(prompts[num_train:], self.tokenizer)

    # 返回训练集对应的 dataloader，随机打乱顺序，每次返回一个样本，加载的工作线程数默认为0
    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=None, num_workers=0)

    # 返回验证集对应的 dataloader，每次返回一个样本
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None, num_workers=0)

# prompt 的数据 pipe 类，返回编码后每个句子的 token id 列表
# prompts 为包含多个句子的列表
class PromptDataPipe(MapDataPipe):
    def __init__(self, prompts, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    # 会返回对应索引的句子经过 tokenizer 处理后的结果，返回的类型为 pt（即pytorch张量），返回其中的 token id 列表。
    def __getitem__(self, index):
        prompt = self.tokenizer(
            self.prompts[index],
            return_tensors="pt",
        )["input_ids"]
        return prompt
