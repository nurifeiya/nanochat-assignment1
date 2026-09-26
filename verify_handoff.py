import os
os.environ.setdefault("NANOCHAT_DTYPE", "float32")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
from pathlib import Path
import torch
from nanochat.checkpoint_manager import load_model
os.environ.setdefault("NANOCHAT_BASE_DIR", str(Path(__file__).resolve().parent.parent / "model_data"))
model, tokenizer, meta = load_model("base", torch.device("cpu"), phase="eval", model_tag="d2_8k", step=35000)
assert meta["step"] == 35000
assert tokenizer.get_vocab_size() == 8192
assert sum(p.numel() for p in model.parameters()) == 3538986
text = "Hello world!"
assert tokenizer.decode(tokenizer.encode(text)) == text
with torch.no_grad():
    ids = torch.tensor([tokenizer.encode(text, prepend="<|bos|>")])
    assert torch.isfinite(model(ids)).all()
print("HANDOFF LOAD CHECK PASSED: step 35000, vocab 8192, finite forward pass")
