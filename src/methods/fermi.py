from __future__ import annotations

from typing import Any, Dict, List

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.methods.base import BaseMethod
from FERMI.src.methods.memory_bank import MemoryBank
from FERMI.src.methods.opro import OPROMethod
from FERMI.src.methods.optimizer_loop import (
    build_fermi_popt_prompt,
    evaluate_prompt_on_samples,
    generate_k_prompts,
)
from FERMI.src.prompts.init_prompt import get_initial_prompt
from FERMI.src.retrieval.rop_selector import RoPSelector


class FERMIMethod(OPROMethod):
    name = "fermi"

    def __init__(self, task: str, config: Dict | None = None) -> None:
        super().__init__(task, config)
        self.rop_n_tilde = int(self.config.get("rop_n_tilde", 3))
        self.rop_backend = str(self.config.get("rop_backend", "mpnet"))
        self.rop_mpnet_model_name = str(
            self.config.get("rop_mpnet_model_name", "sentence-transformers/all-mpnet-base-v2")
        )
        self.rop_selector = RoPSelector(
            n_tilde=self.rop_n_tilde,
            backend=self.rop_backend,
            mpnet_model_name=self.rop_mpnet_model_name,
        )
        self.prompt_pool_final: List[Dict[str, Any]] = []

    def fit(self, train_samples: List[UnifiedSample]) -> None:
        # Call BaseMethod.fit() only (store train_samples); skip OPROMethod.fit() to
        # avoid running a full OPRO optimization loop before FERMI's own loop.
        BaseMethod.fit(self, train_samples)
        if not train_samples:
            return

        # Reset memory bank so no state leaks in from prior calls.
        self.memory = MemoryBank(size=self.l)

        self.tau = float(self.config.get("tau", self.tau))

        optimize_samples, demonstration_samples = self._split_optimization_and_demonstration(train_samples)
        evaluation_samples = optimize_samples
        if self.training_eval_use_llm and self.training_eval_subset_size > 0:
            evaluation_samples = optimize_samples[: min(len(optimize_samples), self.training_eval_subset_size)]

        current_pool: List[Dict[str, Any]] = [
            {"prompt_id": "init_0", "text": get_initial_prompt(self.task, strategy="vanilla")}
        ]

        for step in range(self.t):
            # 1) score prompts
            for i, p in enumerate(current_pool):
                eval_result = evaluate_prompt_on_samples(
                    task=self.task,
                    prompt_text=p["text"],
                    samples=evaluation_samples,
                    tau=self.tau,
                    predictor_fn=self._evaluation_predictor(p) if self.training_eval_use_llm else None,
                )
                score = float(eval_result.get("score", 0.0))
                entry = {
                    "prompt_id": p.get("prompt_id", f"t{step}_p{i}"),
                    "text": p["text"],
                    "score": score,
                    "accuracy": float(eval_result.get("accuracy", 0.0)),
                    "loss": float(eval_result.get("loss", 0.0)),
                    "sample_scores": eval_result.get("sample_scores", {}),
                    "misaligned_records": eval_result.get("misaligned_records", []),
                    "misaligned_indices": eval_result.get("misaligned_indices", set()),
                    "misaligned_count": int(eval_result.get("misaligned_count", 0)),
                    "iteration": step,
                    "surrogate_metric": eval_result.get("surrogate_metric"),
                    "evaluation_backend": eval_result.get("evaluation_backend"),
                }
                self.memory.add(entry)
                self._record_training_entry(entry)

            # 2) memory top-L
            top_memory_desc = self.memory.top()
            memory_asc = list(reversed(top_memory_desc))
            prompt_for_mopt = build_fermi_popt_prompt(
                memory_entries_asc=memory_asc,
                demo_samples=demonstration_samples,
                task_focus=self.task,
                num_questions=len(optimize_samples),
            )

            # 3) generate K prompts (API/fallback)
            generated = generate_k_prompts(
                prompt_for_mopt=prompt_for_mopt,
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

            self._record_iteration_artifacts(step)

        self.prompt_pool_final = []
        for i, p in enumerate(current_pool):
            eval_result = evaluate_prompt_on_samples(
                task=self.task,
                prompt_text=p["text"],
                samples=evaluation_samples,
                tau=self.tau,
                predictor_fn=self._evaluation_predictor(p) if self.training_eval_use_llm else None,
            )
            entry = {
                "prompt_id": p.get("prompt_id", f"t{self.t}_p{i}"),
                "text": p["text"],
                "score": float(eval_result.get("score", 0.0)),
                "accuracy": float(eval_result.get("accuracy", 0.0)),
                "loss": float(eval_result.get("loss", 0.0)),
                "sample_scores": eval_result.get("sample_scores", {}),
                "misaligned_records": eval_result.get("misaligned_records", []),
                "misaligned_indices": eval_result.get("misaligned_indices", set()),
                "misaligned_count": int(eval_result.get("misaligned_count", 0)),
                "iteration": self.t,
                "surrogate_metric": eval_result.get("surrogate_metric"),
                "evaluation_backend": eval_result.get("evaluation_backend"),
            }
            self.prompt_pool_final.append(entry)
            self.memory.add(entry)
            self._record_training_entry(entry)

        if not self.prompt_pool_final:
            self.prompt_pool_final = self.memory.top()

        self.best_prompt = max(self.prompt_pool_final, key=lambda entry: float(entry.get("score", 0.0)))
        self._finalize_training_artifacts()

    def predict(self, sample: UnifiedSample) -> Dict[str, Any]:
        # 4) RoP query-aware prompt selection
        opinion_samples = [
            {
                "id": str(item.get("id", "")),
                "input_text": self.profile_text_fn(item),
            }
            for item in sample.profile
        ]
        selected, neighbors = self.rop_selector.select(
            sample.input_text,
            self.prompt_pool_final,
            opinion_samples=opinion_samples,
        )
        return self._predict_with_selected_prompt(sample=sample, prompt_entry=selected, rop_neighbor_ids=neighbors)

    def _evaluation_predictor(self, prompt_entry: Dict[str, Any]):
        def _predict(sample: UnifiedSample) -> Dict[str, Any]:
            opinion_samples = [
                {
                    "id": str(item.get("id", "")),
                    "input_text": self.profile_text_fn(item),
                }
                for item in sample.profile
            ]
            selected, neighbors = self.rop_selector.select(
                sample.input_text,
                [prompt_entry],
                opinion_samples=opinion_samples,
            )
            return self._predict_with_selected_prompt(
                sample=sample,
                prompt_entry=selected,
                rop_neighbor_ids=neighbors,
                record_event=False,
                count_prediction_source=False,
            )

        return _predict

    def runtime_summary(self) -> Dict[str, Any]:
        base = super().runtime_summary()
        base.update(
            {
                "rop_n_tilde": self.rop_n_tilde,
                "rop_backend": self.rop_backend,
                "rop_mpnet_model_name": self.rop_mpnet_model_name,
                "rop": self.rop_selector.summary(),
                "prompt_pool_size": len(self.prompt_pool_final),
            }
        )
        return base
