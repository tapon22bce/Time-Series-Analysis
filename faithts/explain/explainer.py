"""Two interchangeable explainers with the SAME output schema (a FactSheet + free text), so the
benchmark can score either one identically:

  RuleExplainer   - deterministic, fully verifiable, zero external dependency; used for all
                    experiments run in this sandbox and as the "oracle-faithful" reference.
  QwenVLMExplainer - real VLM (Qwen2.5-VL-7B-Instruct by default); receives the SAME exposed
                    retrieval weights as the diffusion model and is asked to (a) state the 5
                    verifiable facts and (b) attribute its reasoning to specific neighbors by
                    index. Requires a GPU; see docs/VLM.md. Not run inside this sandbox.
"""
from dataclasses import dataclass
import torch
from .facts import FACT_NAMES, facts_to_dict, dict_to_facts, extract_facts
from .render import render_chart


@dataclass
class Explanation:
    facts: torch.Tensor          # (5,) predicted facts (from text, for VLM) or ground truth (rule)
    text: str
    neighbor_attrib: list         # list of (neighbor_idx, weight, one-line reason) actually used
    raw: dict = None


class RuleExplainer:
    """Composes a natural-language explanation strictly from: (1) the forecast's own extracted
    facts, (2) the EXPOSED retrieval weights w, and (3) each retrieved neighbor's facts/similarity.
    Because every claim is derived by code from the same numbers that drove the forecast, this
    explainer is faithful by construction -- it is the reference point for how much fidelity a
    learned (VLM) explainer gives up."""

    def __init__(self, weight_thresh=0.05):
        self.weight_thresh = weight_thresh

    def explain(self, ctx_n, fut_n, gate, w, neighbor_facts, neighbor_sim, neighbor_time_ns=None):
        ctx_n, fut_n, gate, w = ctx_n.detach(), fut_n.detach(), gate.detach(), w.detach()
        neighbor_sim = neighbor_sim.detach()
        facts = extract_facts(ctx_n[None], fut_n[None])[0]
        d = facts_to_dict(facts)
        used = [(i, float(w[i]), float(neighbor_sim[i])) for i in range(len(w)) if w[i] > self.weight_thresh]
        used.sort(key=lambda x: -x[1])
        if float(gate) < 0.15 or not used:
            retr_txt = f"Retrieval mostly ignored (gate={float(gate):.2f}); forecast relies on the context window."
        else:
            parts = []
            for i, wi, si in used[:3]:
                nd = facts_to_dict(neighbor_facts[i])
                parts.append(f"neighbor #{i} (weight={wi:.2f}, similarity={si:.2f}, historically {nd['direction']}/"
                              f"{nd['volatility']} volatility)")
            retr_txt = f"Retrieval used (gate={float(gate):.2f}), driven mainly by " + "; ".join(parts) + "."
        text = (f"Forecast: {d['direction']} trend, peak {d['peak']} in the horizon, {d['volatility']} volatility "
                f"than context, {d['magnitude']} range, level shift: {d['level_shift']}. {retr_txt}")
        attrib = [(i, wi, f"similarity={si:.2f}") for i, wi, si in used]
        return Explanation(facts=facts, text=text, neighbor_attrib=attrib,
                            raw=dict(gate=float(gate), w=w.tolist()))


class QwenVLMExplainer:
    """Loads Qwen2.5-VL-7B-Instruct (or any Qwen2-VL-class checkpoint) via `transformers` and asks
    it to fill the SAME fact schema, given: the rendered chart (context+forecast+top-weighted
    neighbors, colour-coded by their exposed weight) and a text block listing the neighbor weights
    explicitly (so the model cannot "cheat" by inventing attributions we cannot check).
    GPU (>=16GB, or 4-bit) required; NOT instantiated in the CPU sandbox used to build this repo."""

    PROMPT = ("You are a time-series analyst. The chart shows a context window (black), a forecast "
              "(blue) and retrieved historical analogues (orange; thicker/more opaque = higher "
              "retrieval weight). Retrieval weights (index: weight): {weights}. "
              "Return STRICT JSON with keys direction[down/flat/up], peak[early/mid/late], "
              "volatility[lower/similar/higher], magnitude[small/medium/large], level_shift[no/yes], "
              "and neighbor_reliance: for each neighbor index you actually relied on, one short reason. "
              "Only report neighbor_reliance for indices you would still rely on if that neighbor's "
              "weight were the ONLY nonzero one -- do not mention neighbors you are not using.")

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda", dtype="bfloat16", max_new_tokens=200):
        self.model_id, self.device, self.dtype, self.max_new_tokens = model_id, device, dtype, max_new_tokens
        self._model = self._proc = None

    def _lazy_load(self):
        if self._model is not None: return
        import torch as T
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=getattr(T, self.dtype), device_map=self.device)
        self._proc = AutoProcessor.from_pretrained(self.model_id)

    def explain(self, ctx_n, fut_n, gate, w, neighbor_facts, neighbor_sim, neighbor_fut_n=None):
        self._lazy_load()
        img = render_chart(ctx_n.cpu().numpy(), fut_n.cpu().numpy(),
                            [nf.cpu().numpy() for nf in (neighbor_fut_n or [])], w.tolist())
        wtxt = ", ".join(f"{i}: {wi:.2f}" for i, wi in enumerate(w.tolist()) if wi > 0.02)
        prompt = self.PROMPT.format(weights=wtxt or "none")
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        text = self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._proc(text=[text], images=[img], return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen = self._proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return self._parse(gen)

    def _parse(self, gen_text):
        import json, re
        m = re.search(r"\{.*\}", gen_text, re.S)
        d, attrib = {}, []
        if m:
            try:
                d = json.loads(m.group(0))
                attrib = [(int(k), 1.0, v) for k, v in (d.get("neighbor_reliance") or {}).items()]
            except Exception:
                pass
        facts = dict_to_facts(d) if d else torch.full((5,), -1)
        return Explanation(facts=facts, text=gen_text, neighbor_attrib=attrib, raw=d)
