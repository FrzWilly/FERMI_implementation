from __future__ import annotations

import random
from typing import Any, Dict, List

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.methods.base import BaseMethod, predict_from_profile
from FERMI.src.methods.llm_interface import LLMClient
from FERMI.src.methods.memory_bank import MemoryBank
from FERMI.src.methods.optimizer_loop import collect_misaligned_context, generate_k_prompts, score_prompt
from FERMI.src.prompts.init_prompt import get_initial_prompt


class OPROMethod(BaseMethod):
    name = "opro"

    def __init__(self, task: str, config: Dict | None = None) -> None:
        super().__init__(task, config)
        self.k = int(self.config.get("K", 4))
        self.l = int(self.config.get("L", 5))
        self.t = int(self.config.get("T", 10))
        self.train_ratio = float(self.config.get("optimization_train_ratio", 0.2))
        self.m_temperature = float(self.config.get("M_temperature", self.config.get("model_M_temperature", 0.0)))
        self.mopt_temperature = float(self.config.get("Mopt_temperature", 1.0))
        self.model_m_name = str(self.config.get("model_M_name", "gpt-4.1-mini"))
        self.model_mopt_name = str(self.config.get("model_Mopt_name", "gpt-4o-mini"))
        self.openai_api_key_env = str(self.config.get("openai_api_key_env", "OPENAI_API_KEY"))
        self.openai_max_retries = int(self.config.get("openai_max_retries", 2))
        self.openai_request_timeout = float(self.config.get("openai_request_timeout", 30.0))

        self.llm = LLMClient(
            evaluator_model=self.model_m_name,
            optimizer_model=self.model_mopt_name,
            api_key_env=self.openai_api_key_env,
            max_retries=self.openai_max_retries,
            request_timeout=self.openai_request_timeout,
        )
        self.memory = MemoryBank(size=self.l)
        self.best_prompt: Dict[str, Any] = {"prompt_id": "init", "text": get_initial_prompt(task), "score": 0.0}

    def fit(self, train_samples: List[UnifiedSample]) -> None:
        super().fit(train_samples)
        if not train_samples:
            return

        n_opt = max(1, int(len(train_samples) * self.train_ratio))
        optimize_samples = random.sample(train_samples, k=min(n_opt, len(train_samples)))

        current_pool: List[Dict[str, Any]] = [
            {
                "prompt_id": "init_0",
                "text": get_initial_prompt(self.task, strategy="vanilla"),
            }
        ]

        for step in range(self.t):
            for i, p in enumerate(current_pool):
                score = score_prompt(self.task, p["text"], optimize_samples)
                entry = {
                    "prompt_id": p.get("prompt_id", f"t{step}_p{i}"),
                    "text": p["text"],
                    "score": score,
                    "iteration": step,
                }
                self.memory.add(entry)

            best = self.memory.best()
            misaligned = collect_misaligned_context(self.task, optimize_samples)
            generated = generate_k_prompts(
                seed_prompt=str(best.get("text", "")),
                misaligned_context=misaligned,
                k=self.k,
                generator_fn=lambda prompt, temp: self.llm.generate(
                    prompt,
                    temperature=temp,
                    model=self.model_mopt_name,
                    role="optimizer",
                ),
                temperature=self.mopt_temperature,
            )

            current_pool = [
                {
                    "prompt_id": f"t{step+1}_k{idx}",
                    "text": text,
                }
                for idx, text in enumerate(generated)
            ]

        self.best_prompt = self.memory.best()

    def predict(self, sample: UnifiedSample) -> Dict[str, Any]:
        pred = predict_from_profile(self.task, sample)
        return {
            "prediction": pred,
            "selected_prompt_id": self.best_prompt.get("prompt_id"),
            "rop_neighbor_ids": [],
        }

    def runtime_summary(self) -> Dict[str, Any]:
        return {
            "llm": self.llm.summary(),
            "model_M_name": self.model_m_name,
            "model_M_temperature": self.m_temperature,
            "model_Mopt_name": self.model_mopt_name,
            "model_Mopt_temperature": self.mopt_temperature,
        }
