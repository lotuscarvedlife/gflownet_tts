from types import MethodType
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from utils import (
    FrozenModelSentenceGivenPrompt,
    RuleSentenceValidator,
    ModelSentenceValidator,
    ReplayBuffer,
)
from lightning_module import NextSentenceGFNTask
from lightning_data import PromptDataModule
from models import voicecraft
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
import os
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text
)
from argparse import Namespace
from hydra import initialize, compose

# 让我看看你到底是什么东西
def print_all_and_exit(stop: bool, *args, **kwargs):
    if args:
        print("位置参数:")
        for i, arg in enumerate(args, start=1):
            if isinstance(arg, list):
                print(f"  参数 {i}: {arg} (类型: {type(arg)}, 长度：{len(arg)})")
            elif isinstance(arg, torch.Tensor):
                print(f"  参数 {i}: {arg} (类型: {type(arg)}, 形状：{arg.shape})")
            else:
                print(f"  参数 {i}: {arg} (类型: {type(arg)})")
    if kwargs:
        print("关键字参数:")
        for key, value in kwargs.items():
            if isinstance(value, list):
                print(f"  {key}: {value} (类型: {type(value)}, 长度：{len(value)})")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: {value} (类型: {type(value)}, 形状：{value.shape})")
            else:
                print(f"  {key}: {value} (类型: {type(value)})")
    if stop:
        import sys
        sys.exit("调用打印后退出，进程正常停止。")


def get_config():
    with initialize(version_base=None, config_path="./configs/"):
        cfg = compose(config_name="train")
        return cfg


# @hydra.main(version_base=None, config_path="./configs/", config_name="train")
def train(config: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 设置全局随机种子
    pl.seed_everything(config.seed, workers=True)

    # 加载模型和tokenizer，加载参数
    model, audio_tokenizer, text_tokenizer = get_model(config)
    model.to(device)
    model_args = Namespace(**vars(model.args))
    phn2num = model_args.phn2num

    # example_audio = "./demo/output_trump3.wav"
    # example_target_transcript = "Because our leaders are stupid, our politicians are stupid."
    # example_target_transcript = "BECAUSE OUR LEADERS ARE STUPID, OUR POLITICIANS ARE STUPID."
    
    # try: 
    #     original_audio = tokenize_audio(
    #         tokenizer=audio_tokenizer,
    #         audio_path=example_audio
    #     )[0][0] # [1, codebook, time]
    #     original_audio_lens = torch.LongTensor([original_audio.shape[2]])
    #     # assert original_audio.ndim==3 and original_audio.shape[0] == 1 and original_audio.shape[2] == model_args.n_codebooks, original_audio.shape
        
    #     text_tokens = [phn2num[phn] for phn in
    #         tokenize_text(
    #             text_tokenizer, text=example_target_transcript.strip()
    #         ) if phn in phn2num
    #     ]
    #     text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
    #     text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])
    #     assert text_tokens.ndim ==2, text_tokens.shape
    #     assert text_tokens_lens.ndim == 1, text_tokens_lens.shape
        
    # except:
    #     raise NotImplementedError("Can't get end of sentence token id")
    
    # print_all_and_exit(True, text_tokens=text_tokens)
    # 尝试获取音频结束 token
    end_of_audio_token_id = model_args.eos if model_args.eos>0 else model_args.eog

    # batch = {"x":text_tokens.to(device), "x_lens":text_tokens_lens.to(device), "y":original_audio.to(device), "y_lens":original_audio_lens.to(device)}
    # out = model(batch)
    # print_all_and_exit(True, logits = out["logits"])
    # print_all_and_exit(True, out=out)





    # batch = {"x":text_tokens, "x_lens":text_tokens_lens, "y":original_audio, "y_lens":original_audio_lens}
    # results = model(batch)
    # print_all_and_exit(True, results)
    # # 创建一个布尔数组表示哪些Token被认为是非法的，并将其转换为NumPy数组以便后续使用。
    # # vocab_size是tokenizer的词汇表大小
    # illegal_token_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
    # # 获取设置里提供的非法的Token，并转为 python 容器。
    # illegal_tokens = OmegaConf.to_container(config.task.constraints.illegal_tokens)
    # # 遍历非法 token 列表，如果是整数则直接添加 [t]，否则（应该是字符串形式的）将其编码为Token ID。
    # illegal_tokens = [
    #     [t] if isinstance(t, int) else tokenizer.encode(t, add_special_tokens=False)
    #     for t in illegal_tokens
    # ]
    # # 验证每个子列表长度都为1
    # assert all(len(t) == 1 for t in illegal_tokens)
    # # 获取非法 token 的 ID，直接组成一个列表
    # illegal_tokens = [t[0] for t in illegal_tokens]
    # # 将掩码中非法 token 的 ID 对应的位置标记为True
    # illegal_token_mask[illegal_tokens] = True
    # # 将掩码转换为NumPy数组
    # illegal_token_mask = illegal_token_mask.numpy()

    illegal_token_mask = None

    # 设置奖励函数
    reward = get_reward(config, end_of_audio_token_id, illegal_token_mask)
    # 创建一个奖励缓存，用于存储 token 和对应的奖励。
    reward_buffer = ReplayBuffer(
        buffer_size=config.task.reward.buffer_size,         # 缓存大小，设定为50
        termination_token_id=end_of_audio_token_id,      # 句子结束 token 的 ID
    )

    # 根据配置加载数据集并划分训练集和验证集。
    data = PromptDataModule(
        data_path=config.task.data.path,                # 数据存储路径，存储的是 prompt
        audio_tokenizer=audio_tokenizer,                
        text_tokenizer=text_tokenizer,
        phn2num=phn2num,
        train_size=config.task.data.train_size,         # 训练集大小比例，设定为 0.95
        # val_size=config.task.data.val_size,             
        limit_prompts=config.task.data.limit_prompts,   # 限制 prompt 数据集大小
    )
    # 进行 fit 阶段，即开始准备训练集和验证集，不过原本的函数中没有使用就是了
    data.setup("fit")
    # TODO: 对训练和测试探针进行适配
    # 训练集和测试集探针，为列表，提取前 config.task.eval.n_probes 个 prompt 编码 token_id 列表（[0] 是消除批次维度）
    # train_probes = [data.train_data[i][0] for i in range(config.task.eval.n_probes)]
    # val_probes = [data.val_data[i][0] for i in range(config.task.eval.n_probes)]
    train_probes = None
    val_probes = None

    # 创建一个训练任务实例，包含了模型、分词器、奖励机制等所有需要的信息。
    task = NextSentenceGFNTask(
        model=model,
        audio_tokenizer=audio_tokenizer,
        text_tokenizer=text_tokenizer,
        reward=reward,
        reward_buffer=reward_buffer,
        n_samples=config.task.training.n_samples,
        lr=config.task.training.lr,
        subtb_lambda=config.task.training.subtb_lambda,
        pf_temp_high=config.task.training.pf_temp_high,
        pf_temp_low=config.task.training.pf_temp_low,
        pf_temp_prob=config.task.training.pf_temp_prob,
        use_buffer_prob=config.task.training.use_buffer_prob,
        min_sentence_len=config.task.constraints.min_sentence_len,
        max_sentence_len=config.task.constraints.max_sentence_len,
        reward_temp_start=config.task.reward.temp_start,
        reward_temp_end=config.task.reward.temp_end,
        reward_temp_horizon=config.task.reward.temp_horizon,
        illegal_token_mask=illegal_token_mask,
        # train_probes=train_probes,
        # val_probes=val_probes,
        diversity_metric=config.task.eval.diversity_metric,
        use_4bit=config.task.training.use_4bit,
    )

    # 创建一个PyTorch Lightning Trainer实例，配置了加速器、最大轮数、梯度累积步数、日志记录器和回调函数等。
    trainer = pl.Trainer(
        accelerator=config.device.accelerator,                                          # 运行命令指定gpu.yaml，加速器为 cuda
        max_epochs=config.task.training.epochs,                                         # config 文件中指定为 300
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,           # 运行命令有指定为 32，指定梯度累计的步数
        logger=config.logger                                                            # 在 train.yaml 中指定了 wandb
        if isinstance(config.logger, bool)                                              # 意思是如果 config.logger 是布尔值，则不加载 logger
        else hydra.utils.instantiate(config.logger),                                    # 否则就加载指定的 logger
        callbacks=[hydra.utils.instantiate(c) for c in config.task.callbacks],          # 指定训练过程中使用的回调函数，包括保存模型权重，监控指标提早停止训练等函数
    )

    # Fix a bug that arises when using 4-bit quantized models.
    # It's caused by different operations being on different devices,
    # so we'll just deactivate lightning's automatic device placement
    # and let huggingface handle the dynamic device placement
    if config.task.training.use_4bit:
        task.to = MethodType(lambda s, _: s, task)
        task.cuda = MethodType(lambda s: s, task)

    trainer.fit(model=task, datamodule=data)


# 根据配置加载预训练模型和分词器，并应用4位量化和LoRA技术。
def get_model(config: DictConfig):
    # Use 4-bit quantization for lower memory use，在本次训练中没有使用这个技术，技术是用来降低内存使用的
    if config.task.training.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Get the model
    # 加载分词器，名字为 gpt2-xl，默认不添加 beginning of sentence token (BOS)
    from models import voicecraft
    model = voicecraft.VoiceCraft.from_pretrained(f"pyp1/VoiceCraft_{config.task.model.name}")

    encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system(f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=config.device.accelerator) # will also put the neural codec model on gpu
    text_tokenizer = TextTokenizer(backend="espeak")

    # tokenizer = AutoTokenizer.from_pretrained(
    #     config.task.model.name, add_bos_token=False
    # )
    # 加载预训练模型，名字为 gpt2-xl，自动分配设备（cpu或gpu），配置4位量化
    # model = AutoModelForCausalLM.from_pretrained(
    #     config.task.model.name, device_map="auto", quantization_config=bnb_config
    # )

    # Prepare model for k-bit training
    # 准备模型以进行k-bit训练。
    if config.task.training.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,  # Doesn't save memory when generating autoregressively compared to caching
        )

    # Wrap using Lora
    # LoRA: Low Rank Adaptation of Large Language Models; peft: parameter efficient fine-tuning
    # 一种用于微调预训练模型的技术，通过引入低秩矩阵，以减少模型的参数数量，从而降低模型的训练时间和推理时间。
    model = get_peft_model(
        model, hydra.utils.instantiate(config.task.model.lora_config)
    )

    # Remove dropout
    # 移除模型中所有 dropout 层，但好像也把 lora 的 dropout 层也移除了？
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0

    return model, audio_tokenizer, text_tokenizer

# 根据配置选择合适的句子验证器并初始化奖励机制。
def get_reward(config: DictConfig, sentence_token_id, illegal_token_mask):
    # 根据配置选择句子验证器 sentence validator，本次配置为 null
    if config.task.reward.sentence_validator is None:
        sentence_validator, valid_sentence_alpha = None, None
    elif config.task.reward.sentence_validator == "rule":
        sentence_validator, valid_sentence_alpha = (
            RuleSentenceValidator(sentence_token_id=sentence_token_id),
            config.task.reward.valid_sentence_alpha,
        )
    elif config.task.reward.sentence_validator == "model":
        sentence_validator, valid_sentence_alpha = (
            ModelSentenceValidator(sentence_token_id=sentence_token_id),
            config.task.reward.valid_sentence_alpha,
        )
    else:
        raise ValueError(
            f"Invalid sentence validator: {config.task.reward.sentence_validator}"
        )

    reward = FrozenModelSentenceGivenPrompt(
        sentence_token_id=sentence_token_id,                    # 输入语句结束 token id（“.”）
        min_len=config.task.constraints.min_sentence_len,       # 最小句子长度，为1
        vocab_alpha=config.task.reward.vocab_alpha,             # 词汇表偏好的概率补偿
        vocab_naughty_mask=illegal_token_mask,                  # 非法词汇表掩码
        sentence_validator=sentence_validator,                  # 句子验证器
        valid_sentence_alpha=valid_sentence_alpha,              # 验证句子的有效性权重参数
    )

    return reward


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
