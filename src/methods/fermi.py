from __future__ import annotations

import random
from typing import Any, Dict, List

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.methods.base import predict_from_profile
from FERMI.src.methods.opro import OPROMethod
from FERMI.src.methods.optimizer_loop import collect_misaligned_context, generate_k_prompts, score_prompt
from FERMI.src.retrieval.rop_selector import RoPSelector


class FERMIMethod(OPROMethod):
    name = "fermi"

    def __init__(self, task: str, config: Dict | None = None) -> None:
        super().__init__(task, config)
        self.rop_n_tilde = int(self.config.get("rop_n_tilde", 3))
        self.rop_selector = RoPSelector(n_tilde=self.rop_n_tilde)
        self.prompt_pool_final: List[Dict[str, Any]] = []

    def fit(self, train_samples: List[UnifiedSample]) -> None:
        self.train_samples = train_samples
        if not train_samples:
            return

        n_opt = max(1, int(len(train_samples) * self.train_ratio))
        optimize_samples = random.sample(train_samples, k=min(n_opt, len(train_samples)))

        current_pool: List[Dict[str, Any]] = [{"prompt_id": "init_0", "text": self.best_prompt["text"]}]

        for step in range(self.t):
            # 1) score prompts
            for i, p in enumerate(current_pool):
                score = score_prompt(self.task, p["text"], optimize_samples)
                misaligned_ctx = collect_misaligned_context(self.task, optimize_samples, max_items=8)
                self.memory.add(
                    {
                        "prompt_id": p.get("prompt_id", f"t{step}_p{i}"),
                        "text": p["text"],
                        "score": score,
                        "misaligned_context": misaligned_ctx,
                        "iteration": step,
                    }
                )

            # 2) memory top-L
            top_memory = self.memory.top()
            seed = top_memory[0] if top_memory else {"text": self.best_prompt.get("text", "")}
            misaligned_for_generation = seed.get("misaligned_context", [])

            # 3) generate K prompts (API/fallback)
            generated = generate_k_prompts(
                seed_prompt=str(seed.get("text", "")),
                misaligned_context=misaligned_for_generation,
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

        self.prompt_pool_final = self.memory.top()
        self.best_prompt = self.memory.best()

    def predict(self, sample: UnifiedSample) -> Dict[str, Any]:
        # 4) RoP query-aware prompt selection
        selected, neighbors = self.rop_selector.select(sample.input_text, self.prompt_pool_final)
        pred = predict_from_profile(self.task, sample)
        return {
            "prediction": pred,
            "selected_prompt_id": selected.get("prompt_id"),
            "rop_neighbor_ids": neighbors,
        }
