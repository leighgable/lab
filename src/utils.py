from contextlib import contextmanager
import torch
from blocks import Gemma3nTransformer

@contextmanager
def capture_activations(model: Gemma3nTransformer,
    pattern: str
) -> torch.Tensor:
    """ pz.select for torch !!!
        Usage (forward-pass):
            import treescope
            model = Gemma3nTransformer(config)
            with capture_activations(model, pattern=".attn") as traces:
                logits, _ = model(input_ids, segment_pos)

            layer_0_attn = traces['h.0.attn']

            treescope.show(
                "Layer 0 Attention:",
                treescope.ndarray.render_array(
                    layer_0_attn, axis_names = ["batch", "seq", "heads", "dim"]
                )
            )
        Usage (local vars):
            import torch.nn.functional as F
            probs_trace = []

            og_softmax = F.softmax
            def softmax_hook(x, dim=-1, **kwargs):
                res = og_softmax(x, dim=dim, ** kwargs)
                probs_trace.append(res)
                return res

            F.softmax = softmax_hook  # patch softmax
            try:
                logits, _ = model(input_ids, segment_pos)
            finally:
                F.softmax = og_softmax

            treescope.show(
                "Attention heatmap (layer 0, head 0)",
                treescope.ndarray.render_array(
                    probs_trace[0][0,0],
                    axis_names=["query", "key"]
                    )
            )
    """
    recordings = {}
    hooks = []

    def get_hook(name: str):
        def hook(module, input, output):
            recordings[name] = output[2] if isinstance(output, tuple) else output
        return hook

    for name, module in model.named_modules():
        if pattern in name:
            hooks.append(
                         module.register_forward_hook(
                                                      get_hook(
                                                               name
                                                           )
                                                  )
                     )
    try:
        yield recordings
    finally:
        for h in hooks:
            h.remove()        

