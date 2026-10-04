"""Check that the downloaded model loads and generates on the local GPU."""

from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = r"models/Qwen3-0.6B"


def main():
    start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.float16
    ).to("cuda")
    print(f"模型加载: {perf_counter() - start:.1f} 秒")
    print(f"显存峰值: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一个词回答：北京是哪个国家的首都？"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cuda")
    output = model.generate(
        prompt,
        attention_mask=torch.ones_like(prompt),
        max_new_tokens=24,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    print("模型输出:", tokenizer.decode(output[0][prompt.shape[-1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
