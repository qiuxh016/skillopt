"""Direct local Hugging Face Transformers backend for SkillOpt.

The optimizer and target roles each load one local causal language model on
first use. Generation is serialized per model so concurrent SearchQA workers
do not create multiple simultaneous CUDA generations against the same model.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillopt.model.common import TokenTracker, default_model_for_backend
from skillopt.model.common import CompatAssistantMessage

BACKEND_NAME = "transformers_local"


@dataclass
class LocalModelConfig:
    model_path: str
    dtype: str
    quantization: str
    quantization_type: str
    double_quantization: bool
    temperature: float
    top_p: float
    top_k: int
    seed: int | None
    max_tokens: int
    require_cuda: bool


def _role_env(role: str, key: str, default: str) -> str:
    role_key = f"{role.upper()}_TRANSFORMERS_LOCAL_{key}"
    shared_key = f"TRANSFORMERS_LOCAL_{key}"
    return os.environ.get(role_key) or os.environ.get(shared_key) or default


def _initial_config(role: str) -> LocalModelConfig:
    path_key = f"{role.upper()}_TRANSFORMERS_LOCAL_MODEL_PATH"
    path = (
        os.environ.get(path_key)
        or os.environ.get(f"{role.upper()}_DEPLOYMENT")
        or ""
    )
    seed_text = _role_env(role, "SEED", "")
    return LocalModelConfig(
        model_path=path,
        dtype=_role_env(role, "DTYPE", "bfloat16"),
        quantization=_role_env(role, "QUANTIZATION", "none"),
        quantization_type=_role_env(role, "4BIT_QUANT_TYPE", "nf4"),
        double_quantization=_role_env(role, "4BIT_USE_DOUBLE_QUANT", "true").lower()
        in {"1", "true", "yes", "on"},
        temperature=float(_role_env(role, "TEMPERATURE", "1.0")),
        top_p=float(_role_env(role, "TOP_P", "1.0")),
        top_k=int(_role_env(role, "TOP_K", "0")),
        seed=int(seed_text) if seed_text.strip() else None,
        max_tokens=int(_role_env(role, "MAX_TOKENS", "8192")),
        require_cuda=_role_env(role, "REQUIRE_CUDA", "true").lower()
        in {"1", "true", "yes", "on"},
    )


OPTIMIZER_CONFIG = _initial_config("optimizer")
TARGET_CONFIG = _initial_config("target")
_CONFIG_LOCK = threading.RLock()
_GENERATION_LOCK = threading.RLock()
_ROLE_LOCKS = {"optimizer": threading.RLock(), "target": threading.RLock()}
_MODEL_CACHE: dict[str, tuple[tuple[Any, ...], Any, Any]] = {}
tracker = TokenTracker()


def _config_for(role: str) -> LocalModelConfig:
    return OPTIMIZER_CONFIG if role == "optimizer" else TARGET_CONFIG


def _torch_dtype(name: str):
    import torch

    normalized = str(name).strip().lower()
    choices = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in choices:
        raise ValueError(
            f"Unsupported transformers_local_dtype {name!r}; choose bfloat16, float16, or float32"
        )
    return choices[normalized]


def _normalize_quantization(name: str) -> str:
    normalized = str(name or "none").strip().lower().replace("-", "")
    aliases = {"": "none", "no": "none", "none": "none", "4bit": "4bit"}
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported transformers_local_quantization {name!r}; choose 'none' or '4bit'"
        )
    return aliases[normalized]


def _load_model(role: str):
    config = _config_for(role)
    model_path = str(Path(config.model_path).expanduser())
    quantization = _normalize_quantization(config.quantization)
    cache_key = (
        model_path,
        str(config.dtype),
        quantization,
        str(config.quantization_type),
        bool(config.double_quantization),
    )
    cached = _MODEL_CACHE.get(role)
    if cached and cached[0] == cache_key:
        return cached[1], cached[2]

    if not model_path or not Path(model_path).is_dir():
        raise FileNotFoundError(
            f"{role} local model directory does not exist: {model_path!r}. "
            "Set model.optimizer/model.target to the local Hugging Face directory."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "transformers_local is configured to require a CUDA GPU, but CUDA is "
            "not available in this process. Switch the instance to GPU mode and "
            "run with the skillopt conda environment."
        )

    if quantization == "4bit" and not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit transformers_local loading requires a CUDA GPU. "
            "Use a CUDA-enabled instance or set the role quantization to 'none'."
        )

    dtype = _torch_dtype(config.dtype)
    if dtype is torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "The selected GPU does not support bfloat16. Set "
            "model.transformers_local_dtype to float16 in the SearchQA config."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else "cpu",
        "low_cpu_mem_usage": True,
    }
    if quantization == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "4-bit loading requires bitsandbytes. Install it in the active "
                "environment with: python -m pip install 'bitsandbytes>=0.45,<1'"
            ) from exc
        quant_type = str(config.quantization_type).strip().lower()
        if quant_type not in {"nf4", "fp4"}:
            raise ValueError(
                f"Unsupported 4-bit quantization type {config.quantization_type!r}; "
                "choose 'nf4' or 'fp4'"
            )
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=bool(config.double_quantization),
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    _MODEL_CACHE[role] = (cache_key, tokenizer, model)
    print(
        f"  [transformers_local] loaded {role} model from {model_path} "
        f"(dtype={config.dtype}, quantization={quantization}, "
        f"device_map={'auto' if torch.cuda.is_available() else 'cpu'})"
    )
    return tokenizer, model


def _chat(messages: list[dict[str, Any]], max_completion_tokens: int, stage: str, role: str):
    import torch

    if not messages:
        raise ValueError("messages must contain at least one chat message")

    config = _config_for(role)
    # Keep a model's generate calls serialized. This bounds per-role temporary
    # GPU memory even when the trainer runs multiple SearchQA workers.
    with _GENERATION_LOCK, _ROLE_LOCKS[role]:
        tokenizer, model = _load_model(role)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        try:
            input_device = next(model.parameters()).device
        except StopIteration:
            input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])

        model_context = int(
            getattr(model.config, "max_position_embeddings", config.max_tokens + prompt_tokens)
        )
        remaining_context = model_context - prompt_tokens
        if remaining_context < 1:
            raise ValueError(
                f"Prompt uses {prompt_tokens} tokens, exceeding model context {model_context}"
            )
        max_new_tokens = min(
            max(int(max_completion_tokens), 1),
            max(config.max_tokens, 1),
            remaining_context,
        )

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": config.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        eos_token_id = getattr(model.generation_config, "eos_token_id", None)
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if config.temperature > 0:
            generation_kwargs.update(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )
            if config.seed is not None:
                torch.manual_seed(config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(config.seed)

        with torch.inference_mode():
            output = model.generate(**encoded, **generation_kwargs)
        generated = output[0, prompt_tokens:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        completion_tokens = int(generated.numel())
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        tracker.record(stage, prompt_tokens, completion_tokens)
        return text, usage


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    del retries, reasoning_effort, timeout
    return _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens,
        stage,
        "optimizer",
    )


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    del retries, reasoning_effort, timeout
    return _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens,
        stage,
        "target",
    )


def _chat_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    stage: str,
    role: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    return_message: bool = False,
    **kwargs: Any,
):
    del kwargs
    if tools:
        raise NotImplementedError("transformers_local does not implement tool calling")
    text, usage = _chat(messages, max_completion_tokens, stage, role)
    return (CompatAssistantMessage(content=text) if return_message else text), usage


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
):
    del retries, reasoning_effort, tool_choice, timeout
    return _chat_messages(messages, max_completion_tokens, stage, "optimizer", tools=tools, return_message=return_message)


def chat_target_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
):
    del retries, reasoning_effort, tool_choice, timeout
    return _chat_messages(messages, max_completion_tokens, stage, "target", tools=tools, return_message=return_message)


def configure_transformers_local(
    *,
    dtype: str | None = None,
    quantization: str | None = None,
    quantization_type: str | None = None,
    double_quantization: bool | str | None = None,
    temperature: float | str | None = None,
    top_p: float | str | None = None,
    top_k: int | str | None = None,
    seed: int | str | None = None,
    max_tokens: int | str | None = None,
    require_cuda: bool | str | None = None,
    optimizer_dtype: str | None = None,
    optimizer_quantization: str | None = None,
    optimizer_quantization_type: str | None = None,
    optimizer_double_quantization: bool | str | None = None,
    optimizer_temperature: float | str | None = None,
    optimizer_top_p: float | str | None = None,
    optimizer_top_k: int | str | None = None,
    optimizer_seed: int | str | None = None,
    optimizer_max_tokens: int | str | None = None,
    target_dtype: str | None = None,
    target_quantization: str | None = None,
    target_quantization_type: str | None = None,
    target_double_quantization: bool | str | None = None,
    target_temperature: float | str | None = None,
    target_top_p: float | str | None = None,
    target_top_k: int | str | None = None,
    target_seed: int | str | None = None,
    target_max_tokens: int | str | None = None,
) -> None:
    values = {
        "dtype": dtype,
        "quantization": quantization,
        "quantization_type": quantization_type,
        "double_quantization": double_quantization,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "max_tokens": max_tokens,
        "require_cuda": require_cuda,
    }
    role_values = {
        "optimizer": {
            "dtype": optimizer_dtype,
            "quantization": optimizer_quantization,
            "quantization_type": optimizer_quantization_type,
            "double_quantization": optimizer_double_quantization,
            "temperature": optimizer_temperature,
            "top_p": optimizer_top_p,
            "top_k": optimizer_top_k,
            "seed": optimizer_seed,
            "max_tokens": optimizer_max_tokens,
        },
        "target": {
            "dtype": target_dtype,
            "quantization": target_quantization,
            "quantization_type": target_quantization_type,
            "double_quantization": target_double_quantization,
            "temperature": target_temperature,
            "top_p": target_top_p,
            "top_k": target_top_k,
            "seed": target_seed,
            "max_tokens": target_max_tokens,
        },
    }
    with _CONFIG_LOCK:
        for role, role_cfg in (("optimizer", OPTIMIZER_CONFIG), ("target", TARGET_CONFIG)):
            for key, value in values.items():
                override = role_values[role].get(key, None)
                if override is not None:
                    value = override
                if value is None:
                    continue
                if key == "seed":
                    parsed = None if str(value).strip().lower() in {"", "none", "null"} else int(value)
                elif key == "require_cuda":
                    parsed = str(value).strip().lower() in {"1", "true", "yes", "on"}
                elif key == "double_quantization":
                    parsed = str(value).strip().lower() in {"1", "true", "yes", "on"}
                elif key in {"temperature", "top_p"}:
                    parsed = float(value)
                elif key == "top_k" or key == "max_tokens":
                    parsed = int(value)
                elif key == "quantization":
                    parsed = _normalize_quantization(str(value))
                elif key == "quantization_type":
                    parsed = str(value).strip().lower()
                else:
                    parsed = str(value)
                setattr(role_cfg, key, parsed)
            _MODEL_CACHE.pop(role, None)


def get_max_tokens() -> int:
    return TARGET_CONFIG.max_tokens


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    del effort


def set_target_deployment(deployment: str) -> None:
    with _ROLE_LOCKS["target"]:
        TARGET_CONFIG.model_path = str(deployment or "")
        _MODEL_CACHE.pop("target", None)


def set_optimizer_deployment(deployment: str) -> None:
    with _ROLE_LOCKS["optimizer"]:
        OPTIMIZER_CONFIG.model_path = str(deployment or "")
        _MODEL_CACHE.pop("optimizer", None)

