from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.methods.base import BaseMethod, predict_from_profile
from FERMI.src.methods.inference_utils import build_personalized_inference_prompt, parse_llm_prediction
from FERMI.src.methods.llm_interface import LLMClient
from FERMI.src.methods.memory_bank import MemoryBank
from FERMI.src.methods.observability import TrainingArtifacts
from FERMI.src.methods.optimizer_loop import (
    build_opro_popt_prompt,
    evaluate_prompt_on_samples,
    generate_k_prompts,
)
from FERMI.src.prompts.init_prompt import get_initial_prompt
from FERMI.src.utils.logging import get_logger


class OPROMethod(BaseMethod):
    name = "opro"

    def __init__(self, task: str, config: Dict | None = None) -> None:
        super().__init__(task, config)
        self.logger = get_logger(f"fermi.method.{self.name}")
        self.k = int(self.config.get("K", 4))
        self.l = int(self.config.get("L", 5))
        self.t = int(self.config.get("T", 10))
        self.train_ratio = float(self.config.get("optimization_train_ratio", 0.8))
        self.demo_ratio = float(self.config.get("demonstration_ratio", 0.2))
        self.split_per_user = bool(self.config.get("split_per_user", True))
        self.tau = float(self.config.get("tau", 1.0))
        self.m_temperature = float(self.config.get("M_temperature", self.config.get("model_M_temperature", 0.0)))
        self.mopt_temperature = float(self.config.get("Mopt_temperature", 1.0))
        self.model_m_name = str(self.config.get("model_M_name", "gpt-4.1-mini"))
        self.model_mopt_name = str(self.config.get("model_Mopt_name", "gpt-4o-mini"))
        self.openai_api_key_env = str(self.config.get("openai_api_key_env", "OPENAI_API_KEY"))
        self.openai_max_retries = int(self.config.get("openai_max_retries", 2))
        self.openai_request_timeout = float(self.config.get("openai_request_timeout", 30.0))
        self.prediction_profile_limit = int(self.config.get("prediction_profile_limit", 20))
        self.prediction_max_tokens = int(self.config.get("prediction_max_tokens", 64))
        self.training_eval_use_llm = bool(self.config.get("training_eval_use_llm", True))
        self.training_eval_subset_size = int(self.config.get("training_eval_subset_size", 24))
        self.training_eval_max_tokens = int(self.config.get("training_eval_max_tokens", self.prediction_max_tokens))

        self.llm = LLMClient(
            evaluator_model=self.model_m_name,
            optimizer_model=self.model_mopt_name,
            api_key_env=self.openai_api_key_env,
            max_retries=self.openai_max_retries,
            request_timeout=self.openai_request_timeout,
        )
        self.memory = MemoryBank(size=self.l)
        self.best_prompt: Dict[str, Any] = {"prompt_id": "init", "text": get_initial_prompt(task), "score": 0.0}
        self.training_artifacts: TrainingArtifacts | None = None
        self.prediction_source_counts = {"llm": 0, "heuristic_fallback": 0}
        self.last_prediction_fallback_reason: str | None = None

    def set_run_context(self, run_dir: Path, split: str) -> None:
        super().set_run_context(run_dir, split)
        self.training_artifacts = TrainingArtifacts(run_dir=run_dir, method_name=self.name)

    def _record_training_entry(self, entry: Dict[str, Any]) -> None:
        if self.training_artifacts is not None:
            self.training_artifacts.record_prompt_evaluation(entry)

    def _record_iteration_artifacts(self, iteration: int) -> None:
        best = self.memory.best()
        if self.training_artifacts is not None:
            self.training_artifacts.record_iteration_summary(iteration, best)
            self.training_artifacts.write_memory_snapshot(iteration, self.memory.top())
        self.logger.info(
            "[%s][iter=%s] best_prompt=%s score=%.4f accuracy=%.4f loss=%.4f misaligned=%s",
            self.name,
            iteration,
            best.get("prompt_id"),
            float(best.get("score", 0.0)),
            float(best.get("accuracy", 0.0)),
            float(best.get("loss", 0.0)),
            int(best.get("misaligned_count", 0)),
        )

    def _finalize_training_artifacts(self) -> None:
        if self.training_artifacts is not None:
            self.training_artifacts.write_final_memory(self.memory.top())
            self.training_artifacts.finalize()

    def _predict_with_selected_prompt(
        self,
        sample: UnifiedSample,
        prompt_entry: Dict[str, Any],
        rop_neighbor_ids: List[str],
        record_event: bool = True,
        count_prediction_source: bool = True,
    ) -> Dict[str, Any]:
        prompt_text = str(prompt_entry.get("text", "")).strip() or get_initial_prompt(self.task)
        selected_prompt_id = str(prompt_entry.get("prompt_id", "none"))
        raw_response = None
        prediction_source = "llm"
        fallback_reason = None

        try:
            llm_prompt = build_personalized_inference_prompt(
                task=self.task,
                optimized_prompt=prompt_text,
                sample=sample,
                max_profile_items=self.prediction_profile_limit,
            )
            raw_response = self.llm.generate(
                llm_prompt,
                temperature=self.m_temperature,
                max_tokens=self.prediction_max_tokens if record_event else self.training_eval_max_tokens,
                model=self.model_m_name,
                role="evaluator",
                allow_fallback=False,
            )
            parsed = parse_llm_prediction(self.task, raw_response)
            if parsed is None:
                raise RuntimeError("llm_parse_error")
            prediction = parsed
            if count_prediction_source:
                self.prediction_source_counts["llm"] += 1
        except Exception as exc:
            prediction_source = "heuristic_fallback"
            fallback_reason = str(exc)
            if count_prediction_source:
                self.last_prediction_fallback_reason = fallback_reason
            prediction = predict_from_profile(self.task, sample)
            if count_prediction_source:
                self.prediction_source_counts["heuristic_fallback"] += 1
            self.logger.warning(
                "[%s] fallback to heuristic for sample=%s prompt=%s reason=%s",
                self.name,
                sample.id,
                selected_prompt_id,
                fallback_reason,
            )

        event = {
            "sample_id": sample.id,
            "split": self.run_split,
            "selected_prompt_id": selected_prompt_id,
            "prediction_source": prediction_source,
            "fallback_reason": fallback_reason,
            "rop_neighbor_ids": list(rop_neighbor_ids),
            "raw_response": raw_response,
        }
        if record_event and self.training_artifacts is not None:
            self.training_artifacts.record_prediction_event(event)

        return {
            "prediction": prediction,
            "selected_prompt_id": selected_prompt_id,
            "rop_neighbor_ids": list(rop_neighbor_ids),
            "prediction_source": prediction_source,
            "fallback_reason": fallback_reason,
            "raw_response": raw_response,
        }

    def _evaluation_predictor(self, prompt_entry: Dict[str, Any]):
        def _predict(sample: UnifiedSample) -> Dict[str, Any]:
            return self._predict_with_selected_prompt(
                sample=sample,
                prompt_entry=prompt_entry,
                rop_neighbor_ids=[],
                record_event=False,
                count_prediction_source=False,
            )

        return _predict

    def _split_optimization_and_demonstration(
        self, train_samples: List[UnifiedSample]
    ) -> tuple[List[UnifiedSample], List[UnifiedSample]]:
        if not train_samples:
            return [], []

        if not self.split_per_user:
            n_opt = max(1, int(len(train_samples) * self.train_ratio))
            optimize_samples = random.sample(train_samples, k=min(n_opt, len(train_samples)))
            remain = [s for s in train_samples if s not in optimize_samples]
            if not remain:
                remain = optimize_samples[:1]
            n_demo = max(1, int(len(train_samples) * self.demo_ratio))
            demonstration_samples = remain[: min(len(remain), n_demo)]
            return optimize_samples, demonstration_samples

        by_user: Dict[str, List[UnifiedSample]] = {}
        for s in train_samples:
            by_user.setdefault(s.user_id, []).append(s)

        optimize_samples: List[UnifiedSample] = []
        demonstration_samples: List[UnifiedSample] = []

        for user_samples in by_user.values():
            shuffled = list(user_samples)
            random.shuffle(shuffled)

            n_user_opt = max(1, int(len(shuffled) * self.train_ratio))
            n_user_opt = min(n_user_opt, len(shuffled))

            opt = shuffled[:n_user_opt]
            demo_pool = shuffled[n_user_opt:]
            if not demo_pool:
                demo_pool = opt[:1]

            n_user_demo = max(1, int(len(shuffled) * self.demo_ratio))
            n_user_demo = min(n_user_demo, len(demo_pool))

            optimize_samples.extend(opt)
            demonstration_samples.extend(demo_pool[:n_user_demo])

        return optimize_samples, demonstration_samples

    def fit(self, train_samples: List[UnifiedSample]) -> None:
        super().fit(train_samples)
        if not train_samples:
            return

        # tau 由實驗設定（task-specific）決定，避免被 methods 設定覆蓋。
        self.tau = float(self.config.get("tau", self.tau))

        optimize_samples, demonstration_samples = self._split_optimization_and_demonstration(train_samples)
        evaluation_samples = optimize_samples
        if self.training_eval_use_llm and self.training_eval_subset_size > 0:
            evaluation_samples = optimize_samples[: min(len(optimize_samples), self.training_eval_subset_size)]

        current_pool: List[Dict[str, Any]] = [
            {
                "prompt_id": "init_0",
                "text": get_initial_prompt(self.task, strategy="vanilla"),
            }
        ]

        for step in range(self.t):
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

            memory_desc = self.memory.top()
            memory_asc = list(reversed(memory_desc))
            prompt_for_mopt = build_opro_popt_prompt(
                memory_entries_asc=memory_asc,
                demo_samples=demonstration_samples,
            )

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

        self.best_prompt = self.memory.best()
        self._finalize_training_artifacts()

    def predict(self, sample: UnifiedSample) -> Dict[str, Any]:
        return self._predict_with_selected_prompt(sample=sample, prompt_entry=self.best_prompt, rop_neighbor_ids=[])

    def artifact_summary(self) -> Dict[str, Any]:
        if self.training_artifacts is None:
            return {}
        return self.training_artifacts.summary()

    def runtime_summary(self) -> Dict[str, Any]:
        return {
            "llm": self.llm.summary(),
            "model_M_name": self.model_m_name,
            "model_M_temperature": self.m_temperature,
            "model_Mopt_name": self.model_mopt_name,
            "model_Mopt_temperature": self.mopt_temperature,
            "tau": self.tau,
            "optimization_train_ratio": self.train_ratio,
            "demonstration_ratio": self.demo_ratio,
            "split_per_user": self.split_per_user,
            "training_eval_use_llm": self.training_eval_use_llm,
            "training_eval_subset_size": self.training_eval_subset_size,
            "best_prompt_id": self.best_prompt.get("prompt_id"),
            "best_prompt_score": self.best_prompt.get("score"),
            "prediction_source_counts": self.prediction_source_counts,
            "last_prediction_fallback_reason": self.last_prediction_fallback_reason,
            "artifacts": self.artifact_summary(),
        }
