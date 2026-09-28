"""Plausibility critics used at SAMPLING time only (see GaussianDiffusion.sample). None of these
ever receive a gradient from the diffusion model -- they are called under torch.no_grad() and only
reweight/resample particles, exactly like a reward model in best-of-N decoding for LLMs."""
import torch


class HeuristicCritic:
    """No external model: scores physical plausibility against the observed context using cheap,
    inspectable rules (continuity at the join, seasonality match, no bound-violating spikes).
    Used for every critic-guided experiment run inside this CPU sandbox; swap for QwenCritic below
    on a GPU machine without changing any calling code (`critic=` is duck-typed)."""

    def __init__(self, period=24, w_cont=1.0, w_season=1.0, w_spike=0.5):
        self.period, self.w_cont, self.w_season, self.w_spike = period, w_cont, w_season, w_spike

    @torch.no_grad()
    def __call__(self, x0, ctx_n, **kw):
        cont = -(x0[:, 0] - ctx_n[:, -1]).pow(2)
        p = self.period
        if ctx_n.shape[1] >= p and x0.shape[1] >= 1:
            ref = ctx_n[:, -p:].mean(1, keepdim=True)
            season = -(x0.mean(1, keepdim=True) - ref).pow(2).squeeze(1)
        else:
            season = torch.zeros_like(cont)
        spike = -x0.diff(dim=1).abs().amax(1)
        return self.w_cont * cont + self.w_season * season + self.w_spike * spike


class QwenTextCritic:
    """Text-only LLM critic: turns (context stats, candidate forecast facts) into a short numeric
    prompt and reads back a 0-10 plausibility rating via constrained generation / regex parse.
    Cheaper than the VLM critic (no image), good default for large critic_every*n_particles sweeps.
    Requires a GPU-hosted Qwen2.5 (e.g. Qwen2.5-7B-Instruct); NOT run in this CPU sandbox."""

    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", device="cuda", dtype="bfloat16"):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self._model = self._tok = None

    def _lazy_load(self):
        if self._model is not None: return
        import torch as T
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=getattr(T, self.dtype),
                                                             device_map=self.device)

    @torch.no_grad()
    def __call__(self, x0, ctx_n, **kw):
        import re
        self._lazy_load()
        scores = []
        for i in range(x0.shape[0]):
            c, f = ctx_n[i].detach().cpu().numpy(), x0[i].detach().cpu().numpy()
            prompt = (f"Context (last 10 values): {list(map(lambda v: round(float(v),2), c[-10:]))}\n"
                      f"Candidate forecast: {list(map(lambda v: round(float(v),2), f))}\n"
                      "Rate how plausible this continuation is on a 0-10 scale given the context's "
                      "recent trend and volatility. Reply with ONLY the number.")
            ids = self._tok(prompt, return_tensors="pt").to(self._model.device)
            out = self._model.generate(**ids, max_new_tokens=6)
            gen = self._tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"-?\d+(\.\d+)?", gen)
            scores.append(float(m.group()) if m else 0.0)
        return torch.tensor(scores, device=x0.device)


class QwenVLMCritic:
    """Vision critic: renders (context, candidate forecast) as a chart and asks Qwen2.5-VL for a
    0-10 plausibility score. Most faithful to the paper's original VLM-critic idea, most expensive.
    NOT run in this CPU sandbox; see docs/VLM.md for the swap-in instructions."""

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda", dtype="bfloat16"):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self._model = self._proc = None

    def _lazy_load(self):
        if self._model is not None: return
        import torch as T
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=getattr(T, self.dtype), device_map=self.device)
        self._proc = AutoProcessor.from_pretrained(self.model_id)

    @torch.no_grad()
    def __call__(self, x0, ctx_n, **kw):
        import re
        from ..explain.render import render_chart
        self._lazy_load()
        scores = []
        for i in range(x0.shape[0]):
            img = render_chart(ctx_n[i].cpu().numpy(), x0[i].cpu().numpy())
            prompt = "Rate 0-10 how plausible this forecast (blue) looks given the context (black). Reply with ONLY the number."
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            text = self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._proc(text=[text], images=[img], return_tensors="pt").to(self._model.device)
            out = self._model.generate(**inputs, max_new_tokens=6)
            gen = self._proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            m = re.search(r"-?\d+(\.\d+)?", gen)
            scores.append(float(m.group()) if m else 0.0)
        return torch.tensor(scores, device=x0.device)
