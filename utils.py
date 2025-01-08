import torch
import heapq
import pickle
import gzip
import editdistance
import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

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


def lora_to_base(model):
    model.base_model.disable_adapter_layers()
    model.eval()


def base_to_lora(model):
    model.base_model.enable_adapter_layers()
    model.train()

"""
得到的分数是每个采样句子中，在大于等于最小长度的句子中的任意一个位置截止的概率的家和做成的分数
"""
@torch.no_grad()
def score_fast(
    model,                      # base model，非微调模型
    encoded_input,              # 编码后的输入（batch），在后面使用的时候就是已经采样好的多条语句的 token id 序列
    termination_token_id,       # 句号 token id
    min_len,                    # 最小句子长度
    skip_first,                 # 为 prompt_length, 注意这里的 prompt_length 相当于 x_encoded 长度加一（最开始有empty token）
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    prompt_cache=None,      
):
    y_generated_encoded = encoded_input["y_generated_encoded"]  # -> [B,K,L]
    x_encoded = encoded_input["x_encoded"]
    # 再次获取模型输出下一个 token 的得分
    if prompt_cache is None:
        logits = model.logits_forward(x_encoded, y_generated_encoded)  # [B K L+4 card]
    else:
        # NOTE: 因为不会用到所以没有做适配
        raise NotImplementedError("Prompt_cache is not implemented")
        # prompt_cache[1] contains past_key_values which need to be reshaped to the right batch size from encoded_input
        batched_prompt_cache = tuple(
            tuple(
                [
                    prompt_cache[1][i][j].repeat(encoded_input.shape[0], 1, 1, 1)
                    for j in range(len(prompt_cache[1][i]))
                ]
            )
            for i in range(len(prompt_cache[1]))
        )
        logits = model(encoded_input, past_key_values=batched_prompt_cache).logits
    # 去除 prompt 部分的得分（保留最后一个）以及最后的小尾巴
    # get rid of the first few tokens
    logits = logits[:, :, skip_first - 1 :-3]
    # 根据词汇偏好（好与坏）进行概率补偿
    # score the log probability of the input sequence while ignoring termination and padding tokens
    # NOTE: 因为这里用不到所以没有进行适配
    if vocab_nice_mask is not None:
        # add vocab_alpha to the logits of the unmasked vocab items
        raise NotImplementedError
        logits[:, :, ~vocab_nice_mask] += vocab_alpha
    elif vocab_naughty_mask is not None:
        raise NotImplementedError
        # add vocab_alpha to the logits of the masked vocab items
        logits[:, :, vocab_naughty_mask] += vocab_alpha
    # softmax 转化成每组词汇的概率
    logprob = logits.log_softmax(-1)    
    # XXX: 估计这里会出问题，维度什么的可能对不上
    # 提取原来输入的采样后的语句中的生成部分的 token id 序列，并进行维度扩展，由于有 delayed pattern，prompt 前面多了一个 empty token，因此需要减1
    # 这里对延迟 token 做了适配
    token_ids = y_generated_encoded[:, :, skip_first-1:]
    delayed_token_ids = []
    for jj in range(model.args.n_codebooks):
        delayed_token_ids.append(torch.cat([torch.full((logits.shape[0],jj),model.args.empty_token).to(token_ids.device),
                                            token_ids[:, jj, :],
                                            torch.full((logits.shape[0],model.args.n_codebooks-1-jj),termination_token_id).to(token_ids.device)], 
                                            dim=-1))
    delayed_token_ids = torch.stack(delayed_token_ids, dim=1).unsqueeze(-1)
    delayed_token_ids = delayed_token_ids[:, :, :-3]  # 去掉小尾巴
    # 收集之前生成的每句话的 token_ids 对应的概率的对数，并按照码本维度进行相加
    logPF = logprob[:, :, :-1].gather(-1, delayed_token_ids).squeeze(-1).sum(1)        # [B,K,S,1] -> [B,S]
    # 逐步累加每句话的采样的所有词汇的概率，即每步可以停止时，当前生成的句子的概率之和
    logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)，即给定 prompt 下，生成的句子的概率之和
    # 获取每句话所有词汇位置的终止标记的概率，并作为初始 reward
    # XXX: 对应到前面的终止概率记录，这里也仅记录第一个码本的终止概率
    reward = logprob[
        :, 0, :, termination_token_id
    ]  # logP(generated[i+1]=term | prompt + generated[:i+1])，在i+1处停止时的概率       [B,S]
    # 加上之前停止时候的概率，就得到了在任意一个地方停止时整个句子的生成概率
    reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)
    # 标识哪些位置不是终止令牌，标识从生成的位置开始，一旦遇到终止令牌标记则标志为 false，否则为 true
    non_term_mask = (delayed_token_ids.squeeze(-1)[:, 0] != termination_token_id)   # [B, L]
    
    # print_all_and_exit(False, non_term_mask=non_term_mask, y_encoded=y_encoded, delayed_token_ids)

    # 在每一段句子中的最开始添加一个 true（即添加一列 true）
    non_term_mask = torch.cat(
        (
            non_term_mask.new_ones(non_term_mask.shape[0], 1),
            non_term_mask,
        ),
        dim=-1,
    )  # Start (i.e., empty) state has never terminated（即还未生成任何东西一定不是终止符）
    # 将实际中的终止标记位置后续的奖励都设置为 0
    reward[~non_term_mask] = 0.0
    reward_unpenalized = reward.clone()
    # 将小于最小句子长度的句子奖励设置为 -99，防止被选择。
    reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward)
    return reward, reward_unpenalized

# TODO: 完成 reward 的适配,已完成
class FrozenModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_token_id,
        temperature=1.0,
        min_len=1,
        vocab_alpha=-50.0,
        vocab_nice_mask=None,
        vocab_naughty_mask=None,
        sentence_validator=None,
        valid_sentence_alpha=None,
    ):
        assert (
            sentence_validator is None
            and valid_sentence_alpha is None
            or sentence_validator is not None
            and valid_sentence_alpha is not None
        )

        self.temperature = temperature
        self.sentence_token_id = sentence_token_id
        self.vocab_nice_mask = vocab_nice_mask
        self.vocab_naughty_mask = vocab_naughty_mask
        self.vocab_alpha = vocab_alpha
        self.min_len = min_len
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha
    
    # 计算分数函数
    def score(self, input_batch, prompt_length, model, audio_tokenizer, text_tokenizer):
        # 将模型从 lora 切换到 base 模式，lora模式为 low-rank adaptation
        # 这是确保在评分过程中使用的是基础模型而不是微调模型
        lora_to_base(model)
        # 保存当前训练状态并设置为评估模式
        training = model.training
        model.eval()
        # 计算奖励分数
        reward, reward_unpenalized = score_fast(
            model=model,                                    # 基础模型实例
            encoded_input=input_batch,                      # 编码后的输入（batch）
            termination_token_id=self.sentence_token_id,    # 句子结束标记的 token id
            skip_first=prompt_length,                       # prompt 长度，这部分不评分，会跳过
            vocab_nice_mask=self.vocab_nice_mask,           # 词汇表掩码（nice），好像在本代码中没有指定
            vocab_naughty_mask=self.vocab_naughty_mask,     # 非法词汇表掩码
            vocab_alpha=self.vocab_alpha,                   # 词汇偏好（好与坏）概率补偿
            min_len=self.min_len,                           # 最小句子长度约束
        )
        reward /= self.temperature
        reward_unpenalized /= self.temperature
        base_to_lora(model)
        if training:
            model.train()

        # NOTE: 注意，这里任务没有指定，因此没有做适配！！
        if self.sentence_validator is not None:
            raise NotImplementedError("sentence_validator is not implemented!")
            # invalid = self.sentence_validator(input_batch[:, prompt_length:], tokenizer)
            invalid = invalid * self.valid_sentence_alpha
            reward = torch.min(reward, invalid)

        return reward, reward_unpenalized


class SentenceValidator:
    def __init__(self, sentence_token_id) -> None:
        self.sentence_token_id = sentence_token_id

    def __call__(self, sentences, tokenizer):
        pass


class RuleSentenceValidator(SentenceValidator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nlp = spacy.load("en_core_web_lg")

    def __call__(self, sentences, tokenizer):
        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            dtype=torch.bool,
            device=sentences.device,
        )
        invalid[:, 0] = True  # Empty sentence is never valid
        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                if sentences[i, j] == self.sentence_token_id:
                    break  # Only unterminated sentences get a reward
                sent = tokenizer.decode(sentences[i, : j + 1])
                sent = self.nlp(sent).sents
                tokens = []
                for s in sent:
                    for t in s:
                        tokens.append(t)
                if not (len(tokens) >= 2 and tokens[0].is_space and tokens[1].is_title):
                    invalid[i, j + 1] = True  # Must start with a space and capital
                    continue
                has_noun = 1
                has_verb = 1
                for token in tokens:
                    if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                        has_noun -= 1
                    elif token.pos_ in ["VERB", "AUX"]:
                        has_verb -= 1
                if has_noun > 0 or has_verb > 0:
                    invalid[i, j + 1] = True  # Must have a noun and a verb
        return invalid


class ModelSentenceValidator(SentenceValidator):
    def __init__(self, *args, model_name=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if model_name is None:
            model_name = "textattack/roberta-base-CoLA"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, device_map="auto"
        )

    @torch.no_grad()
    def __call__(self, sentences, tokenizer):
        sentences = sentences.to(self.model.device)
        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            dtype=torch.bool,
            device=self.model.device,
        )
        invalid[:, 0] = True  # Empty sentence is never valid
        done = torch.zeros(sentences.shape[0]).bool().to(self.model.device)
        for i in range(sentences.shape[1]):
            sent = sentences[:, : i + 1]
            done |= sent[:, -1] == self.sentence_token_id
            if done.all():
                break
            sent = self.tokenizer(
                tokenizer.batch_decode(sent),
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            invalid_probs = self.model(**sent).logits.softmax(dim=-1)[:, 0]
            invalid[~done, i + 1] = invalid_probs[~done] > 0.2
        return invalid

# 这个地方需要做成 VoiceCraft 的单次采样的情况，即将 inference_tts 移植到此处
def generate_and_return_termination_logprob(
    model,
    encoded_prompt,
    termination_token_id,
    reward_fn,
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    max_len=10,
    min_len=0,
    temperature=1.0,
    top_k=999999,
    top_p=1.0,
    action_seq=None,
    skip_rewards=False,
    n_samples = None,
):
    # 每一步都生成并返回句子的终止概率
    # generate and return the probability of terminating at every step
    # 表示哪些序列仍在生成状态，初始时所有序列为活跃状态。
    active_seqs = torch.ones(n_samples).bool().to(encoded_prompt["y_encoded"].device)
    # 存储目前生成的内容
    cur_generated = [[] for _ in range(n_samples)]
    # 存储前向和终止的概率
    log_pf = []
    log_pterm = []
    # 存储句子的生成状态
    token_ids = None  # For caching hidden states during generation
    past = None  # For caching hidden states during generation
    prompt_length = 0 # 存储提示长度

    # 获取 x 和 y
    x = encoded_prompt["x_encoded"] # A 2-D tensor of shape (1, L)
    x_lens = torch.LongTensor([x.shape[-1]]) # A 1-D tensor of shape (1,). It contains the number of tokens in `x` before padding
    y = encoded_prompt["y_encoded"] # A 3-D tensor of shape (1, K, T)
    # y = y.transpose(2,1) # [1,T,K] -> [1,K,T]

    # 制作注意力掩码和制作带有位置信息的目标文本嵌入
    x_attention_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x.device)
    x_input = model.text_embedding(x)
    x_input = model.text_positional_embedding(x_input)

    # 提取 prompt 时长，做多样本
    y_len = y.shape[2]
    y_lens = torch.LongTensor([y_len]).to(y.device)
    rearranged_y = [[y[0]]]         # -> [1,1,K,T]
    assert rearranged_y[0][0].shape[0] == model.args.n_codebooks, rearranged_y[0][0].shape

    # shift y to create the delayed pattern, 做了 y 的间隔漂移
    shifted_y, patterns = model.shift(rearranged_y) # each element [K S], patterns is not used, as we directly use the original input y
    assert shifted_y[0][0].shape[0] == model.args.n_codebooks, shifted_y[0][0].shape
    assert len(shifted_y[0]) == 1, len(shifted_y[0])

    shifted_y[0][0] = shifted_y[0][0][:, :-(model.args.n_codebooks-1)]    # -> [4, 189] 切了每个 K 的后三个，这样后面的就不会出现 empty_tokens 了
    assert not (shifted_y[0][0][model.args.n_codebooks:] == model.args.empty_token).any() and not (shifted_y[0][0][model.args.n_codebooks:] == model.args.eog).any(), shifted_y[0][0]

    cated_y = shifted_y[0][0].unsqueeze(-1) #[K,S]->[K,S,B]
    new_y_lens = torch.LongTensor([cated_y.shape[1]]).to(cated_y.device)    # 新长度
    assert cated_y.shape == torch.Size((model.args.n_codebooks, cated_y.shape[1], 1))  # -> [4, 189, 1]
    assert not (cated_y == model.args.audio_pad_token).any(), cated_y

    # 这里经过 audio_embedding 模块的处理，将每个码本分别通过对应 embedding 模块，然后再堆积起来。
    embedded_y = torch.stack([model.audio_embedding[k](cated_y[k]) for k in range(model.args.n_codebooks)], dim=0) # [K, S, B, D] torch.Size([4, 189, 1, 2048])
    assert embedded_y.shape[0] == model.args.n_codebooks, embedded_y.shape
    assert embedded_y.shape[-1] == model.args.d_model, embedded_y.shape
    embedded_y = embedded_y.sum(dim=0) # [K,S,B,D]->[S,B,D]  torch.Size([189, 1, 2048]) ，将码本相加
    embedded_y = embedded_y.transpose(1,0) # [S,B,D]->[B,S,D]  torch.Size([1, 189, 2048])

    # positional embedding, 经过 audio_positional_embedding 模块, 添加位置信息
    y_input = model.audio_positional_embedding(embedded_y)  # torch.Size([1, 189, 2048])

    # make attention mask and padding mask, 创建注意力掩码和填充掩码
    y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y.device)
    x_padding_mask = torch.full((1,x_lens[0]), False).to(x.device)
    y_padding_mask = torch.full((1,new_y_lens[0]), False).to(y.device)

    # 标记每个生成序列的码本是否产生了结束符号
    codebook_eog = torch.tensor([[False] * model.args.n_codebooks for _ in range(n_samples)])

    # 存储当前生成的状态
    # state = encoded_prompt.clone()
    state = y.repeat(n_samples, 1, 1)
    assert state.shape==torch.Size([n_samples, model.args.n_codebooks, y_len]), state.shape

    # XXX: 这里 past 为什么这样初始化，也不太清楚
    past = torch.ones([model.args.num_decoder_layers, 2, x.shape[0]], device=x.device, dtype=torch.float32) # -> [16,2,1]

    # 已修改为单次采样（对 n_sample 的概率进行采样）
    def sample_helper(n_sample, n_eog, logits, codebook_eog, top_k, top_p, temperature, cur_num_gen):
        # 没有码本进入结束阶段
        if n_eog == 0:
            prob = logits[n_sample].softmax(dim=-1)
            logits_adjust = logits[n_sample].clone().detach() # [K, card]
            # 对于除了第一个码本之外的所有码本，将 EOG 和空标记的概率设置为极低值（-10000），以防止它们被选中，因为肯定是第一个码本先结束。
            for jj in range(1,model.args.n_codebooks):
                logits_adjust[jj,termination_token_id] = -torch.inf
                logits_adjust[jj,model.args.empty_token] = -torch.inf
            # 处理早期停止，如果生成的 token 数量非常少（小于编码器采样率除以5），则禁止所有码本中的 EOG 标记，确保模型不会过早停止。
            # if cur_num_gen <= model.args.encodec_sr // 5: # this shouldn't happen, but just in case the model stopped too early
            #     logits_adjust[:,:,termination_token_id] = -10000
            ##################### silence repetition handling #####################
            # for b in range(batch_size):
            #     prev_token = prev_tokens[b]
            #     consec_silence_count = consec_silence_counts[b]
            #     if stop_repetition > 0 and prev_token in silence_tokens and consec_silence_count > stop_repetition:
            #         if logits_adjust[b, 0, prev_token] < 0:
            #             logits_adjust[b, 0, prev_token] = logits_adjust[b, 0, prev_token] * (consec_silence_count - (stop_repetition-1))
            #         else:
            #             logits_adjust[b, 0, prev_token] = logits_adjust[b, 0, prev_token] / (consec_silence_count - (stop_repetition-1))
            ##################### silence repetition handling #####################
            # 使用 topk_sampling 函数从调整后的 logits 中抽取样本。样本形状调整为 [batch_size, n_codebooks, 1]。
            # samples = topk_sampling(
            #         logits_adjust.reshape(batch_size * self.args.n_codebooks, logits_adjust.shape[-1]), top_k=top_k, top_p=top_p, temperature=temperature
            #     ) # [B*K, 1]

            if top_k < 999999:
                logits_adjust[prob >= prob.topk(top_k)] = -torch.inf
            if top_p < 1.0:
                sorted_probs, _ = torch.sort(prob, dim=-1, descending=True)
                cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumsum_prob < top_p
                nucleus = torch.cat(
                    [
                        nucleus.new_ones(nucleus.shape[:-1] + (1,)),
                        nucleus[..., :-1],
                    ],
                    dim=-1,
                )
                logits_adjust[~nucleus] = -torch.inf
            if cur_num_gen < min_len:
                # if we haven't reach the minimum length, set the probability of terminating to 0
                logits_adjust[:, termination_token_id] = -torch.inf
            # 如果此时已经达到最大长度限制，则将第一个码本的终止 token 的概率设置为1，其他设置为无穷小
            elif cur_num_gen >= max_len:
                # if we've reached the maximum length, set the probability of terminating to 1
                mask = [True] * logits_adjust.shape[1]
                mask[termination_token_id] = False
                logits_adjust[0, mask] = -torch.inf
            
            # 进行温度处理，让分布更尖或者更平缓
            prob = (logits_adjust / temperature).softmax(dim=-1)
            # 根据概率采样下一个token，生成每一句的下一个 token id
            token_ids = torch.multinomial(prob, num_samples=1) # -> [K, 1]

            # for b in range(batch_size):
            # 在早期阶段，由于 delayed pattern 的存在，因此需要先填充填充 empty_token
            if cur_num_gen < model.args.n_codebooks-1:
                for jj in range(1, model.args.n_codebooks - cur_num_gen):
                    token_ids[-jj, 0] = model.args.empty_token
            # 检查是否有 EOG 标记被选中，如果有，则确定选中，并更新 codebook_eog
            if (
                token_ids[0,0] == termination_token_id
                #  or torch.argmax(logits[n_sample, 0], dim=-1) == termination_token_id
            ): 
                token_ids[0,0] = termination_token_id
                codebook_eog[n_sample, 0] = True

            return token_ids, codebook_eog
        else:
            # 确保 codebook_eog 中标记为 True 的码本数量等于 n_eog
            assert sum(codebook_eog[n_sample, n_eog_i] for n_eog_i in range(n_eog)) == n_eog, f"codebook_eog: {codebook_eog}, but n_eog: {n_eog}"
            prob = logits[n_sample].softmax(dim=-1)
            logits_adjust = logits[n_sample].clone().detach()
            # 这回轮到 n_eog 的码本结束了，因此在此之后的码本先不结束
            for jj in range(n_eog+1,model.args.n_codebooks):
                logits_adjust[jj,termination_token_id] = -torch.inf
                logits_adjust[jj,model.args.empty_token] = -torch.inf
            # 采样
            if top_k < 999999:
                logits_adjust[prob >= prob.topk(top_k)] = -torch.inf
            if top_p < 1.0:
                sorted_probs, _ = torch.sort(prob, dim=-1, descending=True)
                cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumsum_prob < top_p
                nucleus = torch.cat(
                    [
                        nucleus.new_ones(nucleus.shape[:-1] + (1,)),
                        nucleus[..., :-1],
                    ],
                    dim=-1,
                )
                logits_adjust[~nucleus] = -torch.inf
            # 进行温度处理，让分布更尖或者更平缓
            prob = (logits_adjust / temperature).softmax(dim=-1)
            # 根据概率采样下一个token，生成每一句的下一个 token id
            token_ids = torch.multinomial(prob, num_samples=1)   # -> [K,1]
            # 在尾声阶段，由于 delayed pattern 的存在，因此需要再次填充 empty_token
            # NOTE: 这里由于后续算分的需要，因此将填充改成结束符号
            for jj in range(n_eog):
                token_ids[jj, 0] = termination_token_id
            # 强制该码本结束
            token_ids[n_eog, 0] = termination_token_id
            codebook_eog[n_sample, n_eog] = True
            return token_ids, codebook_eog

    # num_gen = [[] for _ in range(n_samples)] # 记录最终生成了多长的语句
    num_gen = 0
    # 生成循环，生成次数为最大句子长度
    for i in range(max_len + 1):
        # 第一次生成
        if i == 0:
            assert x_input.ndim == 3 and x_input.shape[0] == 1, x_input.shape
            assert x_padding_mask.ndim == 2 and x_padding_mask.shape[0] == 1, x_padding_mask.shape
            assert y_input.ndim == 3 and y_input.shape[0] == 1 and y_input.shape[1] == new_y_lens[0], y_input.shape
            assert embedded_y.ndim == 3 and embedded_y.shape[0] == 1 and embedded_y.shape[1] == new_y_lens[0], embedded_y.shape
            x_input = x_input.repeat(n_samples, 1, 1)
            x_lens = x_lens.repeat(n_samples)
            x_padding_mask = x_padding_mask.repeat(n_samples, 1)
            y_input = y_input.repeat(n_samples, 1, 1)
            new_y_lens = new_y_lens.repeat(n_samples)
            prompt_length = new_y_lens[0]  # 记录一下 prompt 长度，用于后面的 score 函数，注意这里是 shift 后的长度
            y_padding_mask = y_padding_mask.repeat(n_samples, 1)
            embedded_y = embedded_y.repeat(n_samples, 1, 1) # will be used to concat with newly generated token embedding
            past = past.repeat(1, 1, n_samples) if past != None else None
        else:
            assert x_input.shape[0] == n_samples and x_padding_mask.shape[0] == n_samples and y_input.shape[0] == n_samples and new_y_lens.shape[0] == n_samples, f"x_input.shape: {x_input.shape}, x_padding_mask.shape: {x_padding_mask.shape}, y_input.shape: {y_input.shape}, new_y_lens.shape: {new_y_lens.shape}"

        y_out, present = model.dec_forward(
                            x_input,                # 编码后的文本输入（带 batch_size）
                            x_lens,                 # 文本长度（带 batch_size）
                            x_attention_mask,       # 文本注意力掩码
                            x_padding_mask,         # 文本填充掩码（带 batch_size）
                            y_input,                # 编码后的音频输入
                            new_y_lens,             # 编码后的音频长度
                            y_attention_mask,       # 音频注意力掩码
                            y_padding_mask,         # 音频填充掩码
                            past=past               # 应该是表示缓存
                        )
        if past != None:
            past = torch.cat([past, present.to(past.dtype)], dim=-2) if past.ndim > 3 else present.to(past.dtype)
        # 输入prompt的 token id 序列和 past key values，获取当前步的输出，这是最重要的
        # output = model(input_ids=token_ids, past_key_values=past_key_values)
        # 更新 past key values
        # past_key_values = output.past_key_values
        # 获取最后一层的 logits，即每个 token 的预测得分，相当于每次都往前推进一个单词（token）
        y_out = y_out[:, -1:] # only take the last token
        logits = torch.stack([model.predict_layer[i](y_out) for i in range(model.args.n_codebooks)], dim=1) # [B K S card], S==1, so [B K 1 card]，因为只取了第一层
        logits = logits.squeeze(2) # [B K card]
        assert logits.shape == torch.Size((n_samples, model.args.n_codebooks, model.n_audio_tokens[0])), f"{logits.shape}"
        
        # XXX: 将 eog 概率设置最小，因为在当前任务中就不应该出现这个 token，不过我不知道这一步放出来会不会有什么其他神奇的影响，还是说放在采样里会好一些
        if model.args.eos > 0:  
            for jj in range(model.args.n_codebooks):
                logits[:,jj,model.args.eog] = -torch.inf
        # 没有动作序列，则进行自动生成
        if action_seq is None:
            with torch.no_grad():
                # 为了代码编写方便，每个生成样本都单独处理
                for n_sample in range(n_samples):
                    n_eog = sum(codebook_eog[n_sample])
                    # 如果当前的样本还没有生成完的话
                    if n_eog != model.args.n_codebooks:
                        token_ids, codebook_eog = sample_helper(n_sample=n_sample, 
                                                                n_eog=n_eog, 
                                                                logits=logits, 
                                                                codebook_eog=codebook_eog, 
                                                                top_k=top_k, 
                                                                top_p=top_p,
                                                                temperature=temperature,
                                                                cur_num_gen=i)
                        cur_generated[n_sample].append(token_ids.squeeze(-1))
                        # if n_eog == 1: # 进入结束阶段
                        #     active_seqs[n_sample] = False
                            # num_gen[n_sample] = i+3 # 记录该条语句最终生成的长度
                    else:
                        # 根据 active_seqs 标记，将已经终止的语句的此时的 token_id 替换为 empty_token，后面就不再让他生成了，全加的是empty_token
                        # active_seqs[n_sample] = False
                        # NOTE: 为了后续算分，这里填充改成结束符号
                        cur_generated[n_sample].append(torch.tensor([termination_token_id]*model.args.n_codebooks).to(x.device))

                # # 根据概率采样下一个token，生成每一句的下一个 token id
                # token_ids = torch.multinomial(prob, num_samples=1)

        # TODO√: 适配奖励缓冲区, 已完成
        else: # 一般是从奖励缓冲区中进行采样就会有固定序列
            # 首先将 action_seq 转为 delayed pattern
            delayed_token_ids = []
            for jj in range(model.args.n_codebooks):
                delayed_token_ids.append(torch.cat([torch.full((action_seq.shape[0],jj),model.args.empty_token).to(action_seq.device),
                                                    action_seq[:, jj, :],
                                                    torch.full((action_seq.shape[0],model.args.n_codebooks-1-jj),termination_token_id).to(action_seq.device)], 
                                                    dim=-1))
            action_seq_delayed = torch.stack(delayed_token_ids, dim=1)
            if i >= action_seq_delayed.size(-1):  # action_seq_delayed [B,K,T]
                token_ids = (
                    torch.ones_like(action_seq[:, :, 0]) * termination_token_id
                ).unsqueeze(-1).to(x.device)     # [B,K,1]
            else:
                token_ids = action_seq_delayed[:, :, i].unsqueeze(-1).to(x.device)       # [B,K,1]
        # 根据 active_seqs 标记，将已经终止的语句的此时的 token_id 替换为终止 token_id，后面就不再让他生成了，全加的是终止token_id
        # token_ids = torch.where(
        #     active_seqs.unsqueeze(-1),
        #     token_ids,
        #     termination_token_id,
        # )
        # 对原来的 logits 根据对应的 mask 进行处理
        # if vocab_nice_mask is not None:
        #     logits[:, ~vocab_nice_mask] += vocab_alpha
        # if vocab_naughty_mask is not None:
        #     logits[:, vocab_naughty_mask] += vocab_alpha
        # 计算对应概率
        logprob = logits.log_softmax(dim=-1) # [B, K, card]
        # 记录终止概率，只记录 alive 的样本的第一个码本的终止概率，死掉的这一步记为0
        # 终止概率即下一步输出终止 token 的概率
        # XXX: 不知道这里的终止概率需不需要乘以码本数量，也不知道这里合不合理
        log_pterm.append(
            torch.where(
                active_seqs,
                logprob[:, 0, termination_token_id],
                0,
            )
        )
        # 更新 active_seqs 标记，如果 token_id 是终止 token id，则标记为 False
        # active_seqs = active_seqs * (token_ids != termination_token_id).squeeze(-1)
        codebook_eog_num = codebook_eog.sum(dim=-1)
        for idx, eog_num in enumerate(codebook_eog_num):
            if eog_num == 1:
                active_seqs[idx] = False
        # 记录前向概率，只记录 alive 的样本的前向概率
        # 前向概率即本次采样选取的 token 对应的概率，死掉的这一步记为0
        # 这个是本次时间步生成的所有 token id
        all_token_ids = torch.stack([gs[-1] for gs in cur_generated]).unsqueeze(-1) if action_seq is None else token_ids # -> [B,K,1]
        log_pf.append(
            torch.where(
                active_seqs,
                logprob.gather(-1, all_token_ids).squeeze(-1).sum(dim=-1),  # 取每一步的概率之和
                0,
            )
        )

        # 制作 samples 的嵌入
        samples_emb = torch.stack([model.audio_embedding[k](all_token_ids[:, k]) for k in range(model.args.n_codebooks)], dim=1) # [B, K,1,D]
        assert samples_emb.shape == torch.Size([n_samples, model.args.n_codebooks, 1, model.args.d_model])
        # 通过沿码本维度求和（dim=1），将不同码本的嵌入向量合并为单个嵌入向量。这样每个样本就只有一个嵌入向量了
        samples_emb = samples_emb.sum(dim=1,keepdim=False) # [B,1,D]
        assert samples_emb.shape == torch.Size((n_samples,1,model.args.d_model))

        embedded_y = torch.cat([embedded_y, samples_emb], dim=1)
        y_input = model.audio_positional_embedding(embedded_y) # [B T D]
        # make attention mask and padding mask
        y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y.device)
        new_y_lens = torch.LongTensor([y_input.shape[1]]).to(y.device).repeat(n_samples)
        y_padding_mask = torch.full((n_samples, new_y_lens[0]), False).to(y.device)

        # 拼接 token_id 到 state 中，最新更新，由于最后才恢复delayedtoken，因此最后再生成state
        # state = torch.cat([state, samples_emb], dim=1)

        num_gen = i+1

        # check if all sequences have terminated
        # 如果所有句子都结束了那就结束
        if torch.all(codebook_eog):
            codebook_eog = torch.tensor([[False] * model.args.n_codebooks for _ in range(n_samples)])
            break

    
    # 现在每个句子都已经生成完了，这里对列表中的 tensor 进行拼接，变成一整个 tensor
    log_pf = torch.stack(log_pf, dim=1)
    log_pf = log_pf[:, :-3]
    log_pterm = torch.stack(log_pterm, dim=1)
    log_pterm = log_pterm[:, :-3]

    # 恢复 delayed pattern
    flatten_gen = []
    # print(num_gen)
    for l, orig_span in enumerate(cur_generated):
        # 堆叠每步生成的 token
        span = torch.stack(orig_span, dim=0) # [T, K]
        span = span.transpose(1,0) # [K, T]
        
        # 移除多余的空白向量，新改动，打算将这个移到后续的分数计算中
        # end_idx = torch.nonzero(span[-1] == termination_token_id)
        # assert end_idx.shape[0]==1, f"end_token have {end_idx.shape[0]} !"
        # span = span[:, :end_idx[0]+1]

        assert span.shape[0] == model.args.n_codebooks, span.shape
        unshifted_span = []
        for j, s in enumerate(span):
            start_from = j
            end_at = - (model.args.n_codebooks - start_from)    # 这里理论上也会将最长的句子的结束符号删除
            unshifted_span.append(s[start_from:end_at])
        unshifted_span = torch.stack(unshifted_span, dim=0) # [K,T]

        assert unshifted_span.shape[1] == num_gen - model.args.n_codebooks, f"len(unshifted_spans[0]): {len(unshifted_span[0])}, num_gen: {num_gen}"
        flatten_gen.append(unshifted_span)
    
    # 衔接 prompt 部分，这里就是所有生成的语句接上之前的 prompt
    state = torch.cat([state, torch.stack(flatten_gen, dim=0)], dim=2)
    expected_y_len = y_len + num_gen - model.args.n_codebooks
    assert state.shape == torch.Size((n_samples, model.args.n_codebooks, expected_y_len)), f"state.shape: {state.shape}, expected_y_len: {expected_y_len}. y_len + num_gen - model.args.n_codebooks: {y_len} + {num_gen} - {model.args.n_codebooks}"
    # expected_y_len = torch.tensor([y_len]*n_samples) + torch.tensor([item - model.args.n_codebooks for item in num_gen])
    # assert state.shape == torch.Size((n_samples, model.args.n_codebooks, expected_y_len)), f"res.shape: {res.shape}, expected_y_len: {expected_y_len}. y_len + sum([item - self.args.n_codebooks for item in num_gen]): {y_len} + {sum([item - self.args.n_codebooks for item in num_gen])}"


    # 计算奖励，forward 中，skip_rewards=False
    if skip_rewards:
        log_r, log_r_unpenalized = None, None
    else:
        input_batch = {
            "y_generated_encoded":state,      # -> [B, K, L]
            "x_encoded":x.repeat(n_samples, 1),      # 注意这里要重复
        }
        # 对得到的所有采样语句的 token_id 序列计算奖励分数，剔除最后一个 token，因为这里可以确保是
        # Reward for all intermediate states (except the last one,
        # which is guaranteed to be the termination token)
        log_r, log_r_unpenalized = reward_fn(input_batch=input_batch, prompt_length=prompt_length)
    # add a termination token to the end of the sequence
    cur_generated = [[] for _ in range(n_samples)]
    return state, log_pf, log_pterm, log_r, log_r_unpenalized


# def generate_and_return_termination_logprob(
#     model,
#     encoded_prompt,
#     termination_token_id,
#     reward_fn,
#     vocab_nice_mask=None,
#     vocab_naughty_mask=None,
#     vocab_alpha=-99,
#     max_len=10,
#     min_len=0,
#     temperature=1.0,
#     top_k=999999,
#     top_p=1.0,
#     action_seq=None,
#     skip_rewards=False,
# ):
#     # 每一步都生成并返回句子的终止概率
#     # generate and return the probability of terminating at every step
#     # 表示哪些序列仍在生成状态，初始时所有序列为活跃状态。
#     active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
#     # 存储当前生成的状态
#     state = encoded_prompt.clone()
#     # 存储前向和终止的概率
#     log_pf = []
#     log_pterm = []
#     # 存储句子的生成状态
#     token_ids = state  # For caching hidden states during generation
#     past_key_values = None  # For caching hidden states during generation
#     # 生成循环，生成次数为最大句子长度
#     for i in range(max_len + 1):
#         # 输入prompt的 token id 序列和 past key values，获取当前步的输出，这是最重要的
#         output = model(input_ids=token_ids, past_key_values=past_key_values)
#         # 更新 past key values
#         past_key_values = output.past_key_values
#         # 获取最后一层的 logits，即每个 token 的预测得分，相当于每次都往前推进一个单词（token）
#         logits = output.logits[:, -1, :]
#         # 没有动作序列，则进行自动生成
#         if action_seq is None:
#             with torch.no_grad():
#                 # 使用 softmax 转化为概率
#                 prob = logits.softmax(dim=-1)
#                 modified_logits = logits.clone().detach()
#                 # 进行 top-k 采样，只保留概率最高的 K 个 token，其余 token 的概率被设置为零
#                 # implement top-k by getting the top-k largest values and setting the rest to 0
#                 if top_k < 999999:
#                     modified_logits[prob >= prob.topk(top_k)] = -torch.inf
#                 # 进行 top-p 采样，只保留概率和前 K 个 token 的和为 top_p 的 token，其余 token 的概率被设置为零
#                 # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
#                 if top_p < 1.0:
#                     sorted_probs, _ = torch.sort(prob, dim=-1, descending=True)
#                     cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
#                     nucleus = cumsum_prob < top_p
#                     nucleus = torch.cat(
#                         [
#                             nucleus.new_ones(nucleus.shape[:-1] + (1,)),
#                             nucleus[..., :-1],
#                         ],
#                         dim=-1,
#                     )
#                     modified_logits[~nucleus] = -torch.inf
#                 # 如果此时还处于最小长度限制内，则将终止 token 的概率设置为无穷小
#                 if i < min_len:
#                     # if we haven't reach the minimum length, set the probability of terminating to 0
#                     modified_logits[:, termination_token_id] = -torch.inf
#                 # 如果此时已经达到最大长度限制，则将终止 token 的概率设置为1，其他设置为无穷小
#                 elif i >= max_len:
#                     # if we've reached the maximum length, set the probability of terminating to 1
#                     mask = [True] * modified_logits.shape[1]
#                     mask[termination_token_id] = False
#                     modified_logits[:, mask] = -torch.inf
#                 # 将非 nice 的词汇的概率设置为负数
#                 if vocab_nice_mask is not None:
#                     # add vocab_alpha to the logits of the unmasked vocab items
#                     modified_logits[:, ~vocab_nice_mask] += vocab_alpha
#                 # 将非法词汇的概率设置为负数
#                 if vocab_naughty_mask is not None:
#                     # add vocab_alpha to the logits of the masked vocab items
#                     modified_logits[:, vocab_naughty_mask] += vocab_alpha
#                 # 进行温度处理，让分布更尖或者更平缓
#                 prob = (modified_logits / temperature).softmax(dim=-1)
#                 # 根据概率采样下一个token，生成每一句的下一个 token id
#                 token_ids = torch.multinomial(prob, num_samples=1)
#         else: # 一般是从奖励缓冲区中进行采样就会有固定序列 
#             if i >= action_seq.size(-1):
#                 token_ids = (
#                     torch.ones_like(action_seq[:, 0]) * termination_token_id
#                 ).unsqueeze(-1)
#             else:
#                 token_ids = action_seq[:, i].unsqueeze(-1)
#         # 根据 active_seqs 标记，将已经终止的语句的此时的 token_id 替换为终止 token_id，后面就不再让他生成了，全加的是终止token_id
#         token_ids = torch.where(
#             active_seqs.unsqueeze(-1),
#             token_ids,
#             termination_token_id,
#         )
#         # 对原来的 logits 根据对应的 mask 进行处理
#         if vocab_nice_mask is not None:
#             logits[:, ~vocab_nice_mask] += vocab_alpha
#         if vocab_naughty_mask is not None:
#             logits[:, vocab_naughty_mask] += vocab_alpha
#         # 计算对应概率
#         logprob = logits.log_softmax(dim=-1)
#         # 记录终止概率，只记录 alive 的样本的终止概率，死掉的这一步记为0
#         # 终止概率即下一步输出终止 token 的概率
#         log_pterm.append(
#             torch.where(
#                 active_seqs,
#                 logprob[:, termination_token_id],
#                 0,
#             )
#         )
#         # 更新 active_seqs 标记，如果 token_id 是终止 token id，则标记为 False
#         active_seqs = active_seqs * (token_ids != termination_token_id).squeeze(-1)
#         # 记录前向概率，只记录 alive 的样本的前向概率
#         # 前向概率即本次采样选取的 token 对应的概率，死掉的这一步记为0
#         log_pf.append(
#             torch.where(
#                 active_seqs,
#                 logprob.gather(-1, token_ids).squeeze(-1),
#                 0,
#             )
#         )
#         # 拼接 token_id 到 state 中
#         state = torch.cat([state, token_ids], dim=-1)
#         # check if all sequences have terminated
#         # 如果所有句子都结束了那就结束
#         if torch.all(~active_seqs):
#             break
#     # 现在每个句子都已经生成完了，这里对列表中的 tensor 进行拼接，变成一整个 tensor
#     log_pf = torch.stack(log_pf, dim=1)
#     log_pterm = torch.stack(log_pterm, dim=1)
#     # 计算奖励，forward 中，skip_rewards=False
#     if skip_rewards:
#         log_r, log_r_unpenalized = None, None
#     else:
#         # 对得到的所有采样语句的 token_id 序列计算奖励分数，剔除最后一个 token，因为这里可以确保是
#         # Reward for all intermediate states (except the last one,
#         # which is guaranteed to be the termination token)
#         log_r, log_r_unpenalized = reward_fn(state[:, :-1])
#     # add a termination token to the end of the sequence
#     return state, log_pf, log_pterm, log_r, log_r_unpenalized

# 本函数为 GFN 的子轨迹平衡损失
def modified_subtb_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_audio,
    termination_token_id,
    prompt_len,
    subtb_lambda=1.0,
):
    # 在末尾添加结束符号用以适配
    generated_audio = torch.cat([generated_audio, torch.full((generated_audio.shape[0],1),termination_token_id).to(generated_audio.device)], dim=-1)

    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_audio.shape[1] - prompt_len
    ), f"log_pf.shape[1]: {log_pf.shape[1]}, log_r.shape[1]: {log_r.shape[1]}, log_pterm.shape[1]: {log_pterm.shape[1]}, generated_audio.shape[1]: {generated_audio.shape[1]}, prompt_len: {prompt_len}"
    assert (
        log_pf.shape[1] > 1
    )  # With modified-style losses, we need at least one transition before terminating

    delta = (
        log_r[:, :-1]
        + log_pf[:, :-1]
        + log_pterm[:, 1:]
        - log_r[:, 1:]
        - log_pterm[:, :-1]
    )
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    # Get a mask for tokens after the termination token in the generated_audio
    mask = (generated_audio[:, prompt_len:-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_audio.shape[1] - prompt_len
    for subtraj_len in range(1, generated_len):
        subtb_term = (
            delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]
        ) ** 2
        subtb_term[mask[:, subtraj_len - 1 :]] = 0
        batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
        total_lambda += (
            subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
        )
    batch_loss /= total_lambda

    return batch_loss

# 用于计算生成文本的终止位置相关的值，包括累积的前向概率 (log_pfs)、奖励 (log_r) 和未惩罚的奖励 (log_r_unpenalized)
def get_termination_vals(
    generated_audio,    # 只传入了第一个码本
    log_pf,
    log_pterm,
    log_r,
    log_r_unpenalized,
    termination_token_id,
    prompt_len,
):
    # 在末尾添加结束符号用以适配
    generated_audio = torch.cat([generated_audio, torch.full((generated_audio.shape[0],1),termination_token_id).to(generated_audio.device)], dim=-1)

    # batch idx 为每个生成的文本的索引
    batch_idx = torch.arange(generated_audio.size(0))
    # 获取每个文本生成部分的长度（不包含终止标记）
    # try:
    #     gen_len = (
    #         (generated_audio[:, prompt_len:] == termination_token_id).byte().argmax(dim=-1)
    #     )
    # except:
    #     print_all_and_exit(True, 
    #                        generated_audio=generated_audio, 
    #                        prompt_len=prompt_len, 
    #                        log_pf=log_pf, 
    #                        log_pterm=log_pterm,
    #                        log_r=log_r
    #                        )
    gen_len = (
        (generated_audio[:, prompt_len:] == termination_token_id).byte().argmax(dim=-1)
    )
    if log_pf is None and log_pterm is None:
        log_pfs = None
    else:
        # 在 log_pf 中插入一个 0 列，用于计算 cumsum，顺便删除最后一列
        log_pf = torch.cat([torch.zeros_like(log_pf[:, :1]), log_pf], dim=-1)[:, :-1]
        # 计算累积前向概率，并添加当前位置终止的概率
        log_pfs = log_pf.cumsum(dim=-1) + log_pterm
        # 得到各个生成的语句的累计前向概率加上位置终止的概率
        log_pfs = log_pfs[batch_idx, gen_len]
    # 获取各个生成语句的直接奖励分数和未惩罚奖励分数
    # print_all_and_exit(True, batch_idx=batch_idx, gen_len=gen_len, log_r=log_r)
    log_r = log_r[batch_idx, gen_len]
    # # 检查一下是什么情况
    # if batch_idx < log_r.size(0) and gen_len < log_r.size(1):
    #     log_r_slice = log_r[batch_idx, gen_len]
    # else:
    #     raise IndexError(f"Index out of bounds for log_r tensor. batch_idx={batch_idx}, gen_len={gen_len}, log_r.shape={log_r.shape}")
    log_r_unpenalized = log_r_unpenalized[batch_idx, gen_len]
    return log_pfs, log_r, log_r_unpenalized, gen_len


class SequenceDiversity:
    def __init__(self, method, **kwargs):
        self.method = method
        if method is None:
            pass
        # 本次训练采用这个方案
        elif method == "sequence_embedding":
            # 获取 model name 参数，默认为后面的内容
            model_name = kwargs.get(
                "model_name", "sentence-transformers/all-mpnet-base-v2"
            )
            # 设置 model
            self.model = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unknown sequence diversity method: {method}")

    # 调用当前类的方法如下，接收一个 sequences 参数，应该是多个句子的 token_id 序列
    @torch.no_grad()
    def __call__(self, sequences):
        if self.method is None:
            return None
        elif self.method == "sequence_embedding":
            # 通过编码器获取输入语句的 embeddings
            embeddings = self.model.encode(sequences, show_progress_bar=False)
            # 计算 cosine similarity
            sim = cos_sim(embeddings, embeddings)
            # 使用 torch.triu_indices 获取上三角矩阵的索引（排除对角线），以避免重复计算
            indices = torch.triu_indices(len(sequences), len(sequences), offset=1)
            # 计算非对角线上元素的平均值，并从1中减去得到多样性分数。
            diversity = 1 - sim[indices[0], indices[1]].mean().item()
        else:
            raise ValueError(f"Unknown sequence diversity method: {self.method}")
        return diversity


class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    """
    item 是一个字典，包含：
        "logreward": 当前存储的生成语句的奖励。
        "str_sentence": 生成语音的字符串表示，我计划就是用第一个码本的 token 序列做成字符串
        "tensor_sentence": 生成语音的张量表示，这里我们规定为生成的语音 token_id 序列[K,T]
        "full_logrewards": 完整的生成语句，包含子轨迹的对数奖励序列。

    self._buffer 是一个字典，key 是提示字符串 str_prompt，value 是另一个字典，包含：
        "tensor_prompt": 提示的 token 表示。
        "sentences": 一个列表，包含所有这个提示相关的 item。
        "exists": 一个集合，包含所有与这个提示相关的 item 的生成句子的字符串。
    """
    def __init__(self, buffer_size, termination_token_id, sim_tolerance=0.25):
        self.buffer_size = buffer_size
        self.termination_token_id = termination_token_id
        self.sim_tolerance = sim_tolerance                  # 相似度容忍阈值
        self.reset()

    # 清空缓存
    def reset(self):
        self._buffer = {}

    # 添加一个 item 到缓存中
    def add(self, item):
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # 如果这个 item 对应的生成的字符串已经在 buffer 中，则跳过不进行添加操作
        # if item is already in the buffer, skip it
        str_prompt = item["str_prompt"]
        if item["str_sentence"] in self._buffer[str_prompt]["exists"]:
            return
        # if the edit distance between item and any item in the buffer is small, skip it
        # 计算新项目与缓冲区中每个项目的编辑距离。
        tokenized_sentence = [
            x
            for x in item["tensor_sentence"][0].tolist()    # 这里取第一个码本
            if x != self.termination_token_id
        ]
        for buffer_item in self._buffer[str_prompt]["sentences"]:
            tokenized_existing_sentence = [
                x for x in buffer_item[2][0].tolist() if x != self.termination_token_id  # 也是取第一个码本
            ]
            # 如果遍历的当前项目的编辑距离小于阈值
            if (
                editdistance.eval(tokenized_sentence, tokenized_existing_sentence)
                < (len(tokenized_sentence) + len(tokenized_existing_sentence))
                * self.sim_tolerance
            ):
                # 当前项目的日志奖励大于等于新项目的日志奖励，则跳过新项目。
                if buffer_item[0] >= item["logreward"]:
                    return
                # 新项目的日志奖励大于当前项目的日志奖励，则删除旧项目，并添加新的项目
                else:
                    self._buffer[str_prompt]["exists"].remove(buffer_item[1])
                    self._buffer[str_prompt]["sentences"].remove(buffer_item)
                    heapq.heapify(self._buffer[str_prompt]["sentences"])
                    self._buffer[str_prompt]["exists"].add(item["str_sentence"])
                    heapq.heappush(
                        self._buffer[str_prompt]["sentences"],
                        (
                            item["logreward"],
                            item["str_sentence"],
                            item["tensor_sentence"],
                            item["full_logrewards"],
                        ),
                    )
                    return
        # 已确定添加，注册当前语句，防止后面存在重复计算
        self._buffer[str_prompt]["exists"].add(item["str_sentence"])
        # 如果缓存已满，则弹出最小奖励的项目并添加新项目
        if len(self._buffer[str_prompt]["sentences"]) >= self.buffer_size:
            popped = heapq.heappushpop(
                self._buffer[str_prompt]["sentences"],
                (
                    item["logreward"],
                    item["str_sentence"],
                    item["tensor_sentence"],
                    item["full_logrewards"],
                ),
            )
            self._buffer[str_prompt]["exists"].remove(popped[1])
        # 否则直接添加新的项目
        else:
            heapq.heappush(
                self._buffer[str_prompt]["sentences"],
                (
                    item["logreward"],
                    item["str_sentence"],
                    item["tensor_sentence"],
                    item["full_logrewards"],
                ),
            )

    # 批量添加新的项目（包含新的提示生成集合）
    def add_batch(self, prompt, sentences, logrewards):
        """
        add a batch of items to the buffer
         - prompt: 这里的 prompt 是个字典,包含"x_encoded"[1,L]和"y_encoded"[1,T,K]
         - sentence: 这里为去掉 prompt 部分的，没有delayed pattern的生成语音token，[B,K,T]
        """
        def batch_decode(sentences):
            token_sentences = []
            for i in range(sentences.size(0)):
                str = " ".join(map(sentences[i, 0, :].tolist()))
                str_no_term_token = str.replace(str(self.termination_token_id), "").strip()
                token_sentences.append(str_no_term_token)
            return token_sentences
        
        str_prompt = " ".join([str(x) for x in prompt["x_encoded"][0].tolist()])      # 这里我们规定就是 text prompt
        # 如果这个提示字符串是新的字符串，则直接创建一个条目
        if str_prompt not in self._buffer:
            self._buffer[str_prompt] = {
                "tensor_prompt": prompt,
                "sentences": [],
                "exists": set(),
            }
        # 将终止 token 之后的所有 token 设置为终止 token 。
        sentences[
            (sentences == self.termination_token_id).cumsum(dim=-1) >= 1
        ] = self.termination_token_id
        # 解码出句子字符串
        token_sentences = batch_decode(sentences)
        # 逐个添加对应的句子（会将句号删除）
        for i in range(sentences.size(0)):
            str_sentence = token_sentences[i]
            self.add(
                {
                    "logreward": logrewards[
                        i, (sentences[i][0] != self.termination_token_id).sum()
                    ].item(),
                    "str_prompt": str_prompt,   # 这里就是 text_prompt_token_id 做的字符串
                    "str_sentence": str_sentence,   # 这里是用第一个码本的token id 做的字符串，不包含终止 token，用空格衔接
                    "tensor_sentence": sentences[i],    # [K, T]，没有delayed pattern的生成语音token
                    "full_logrewards": logrewards[i, :],
                }
            )

    # 从缓冲区中均匀采样一批项目，并返回堆叠的张量
    def sample(self, batch_size, prompt):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        # 将提示张量转为字符串形式
        str_prompt = " ".join([str(x) for x in prompt["x_encoded"][0].tolist()])
        # 提示张量不在缓冲区，则返回 None
        if str_prompt not in self._buffer:
            return None, None
        # 从缓冲区中随机采样一批项目
        prompt_buffer = self._buffer[str_prompt]["sentences"]
        # 抽取 batch_size 个随机 idx，允许重复
        idx = np.random.choice(
            len(prompt_buffer),
            batch_size,
            replace=True,
        )
        # 返回采样的 tensor_sentence 和 full_logrewards，使用终止 token 和 0 进行填充
        # 这里由于prompt_buffer[i][2] 返回的是 [K,T] 形状的向量，pad_sequence 可能有问题，因此我们改一下
        return torch.nn.utils.rnn.pad_sequence(     # 这个是 tensor_sentence，修改后就可以返回 [B, K, T] 形状的向量了
            [prompt_buffer[i][2].transpose(0,1) for i in idx],
            batch_first=True,
            padding_value=self.termination_token_id,
        ).transpose(1, 2), torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][3] for i in idx],
            batch_first=True,
            padding_value=0,
        )

    # 打印缓冲区内容
    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["sentences"]:
                print(item[1])
            print("")

    # 保存缓冲区内容
    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)
