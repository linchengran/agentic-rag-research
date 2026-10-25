"""本地大语言模型加载与推理封装。

支持 HuggingFace transformers 原生 GPTQConfig / AwqConfig / BitsAndBytesConfig 三种量化路径,
参考 rag-research-project/src/generator.py 的接口风格。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    """推理参数,默认使用贪心解码保证可复现。"""

    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False


class LocalModel:
    """本地 Qwen2.5 系列模型的薄封装。"""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        quantization: Optional[str] = None,
    ) -> None:
        """加载本地 Qwen2.5 系列模型。

        Args:
            model_name: HF Hub ID 或本地路径(预量化模型名已含 GPTQ/AWQ 后缀)。
            device: ``auto`` / ``cuda`` / ``cpu``。
            quantization: 量化方案(transformers 5.x 原生支持,无需外部量化库)。
                - ``None``: 全精度加载(显存大但通用)
                - ``"gptq"``: 预量化 GPTQ 模型(GPTQConfig,AutoModel 自动识别权重格式)
                - ``"awq"``: 预量化 AWQ 模型(AwqConfig,AutoModel 自动识别权重格式)
                - ``"bitsandbytes"``: 4bit 动态量化(BitsAndBytesConfig,需 bitsandbytes)
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"device_map": device}

        if quantization == "gptq":
            # transformers 5.x 原生 GPTQ 支持;预量化模型 AutoModel 自动识别权重格式
            from transformers import GPTQConfig
            kwargs["quantization_config"] = GPTQConfig(bits=4)
            kwargs["local_files_only"] = True
        elif quantization == "awq":
            from transformers import AwqConfig
            kwargs["quantization_config"] = AwqConfig(bits=4)
            kwargs["local_files_only"] = True
        elif quantization == "bitsandbytes":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["local_files_only"] = True
        elif quantization is not None:
            raise ValueError(
                f"未知 quantization={quantization!r};可选 None/'gptq'/'awq'/'bitsandbytes'"
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """批量生成回复。"""
        cfg = config or GenerationConfig()
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=cfg.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # 仅解码新生成的部分,去除输入前缀
        results: List[str] = []
        for i, out_ids in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            new_ids = out_ids[input_len:]
            results.append(self.tokenizer.decode(new_ids, skip_special_tokens=True))
        return results

    def chat(
        self,
        messages: List[dict],
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """单轮 chat 模板调用。

        messages 形如 [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.generate([prompt], config)[0]