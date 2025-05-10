# compare_models

## install Dependencies:
```bash
pip install transformers torch matplotlib seaborn tqdm
```
## Run the Script:
```bash
python compare_models.py -model1 "Qwen/Qwen3-0.6B" --model2 "huihui-ai/Qwen3-0.6B-abliterated" --plot
```
Compares weights of Qwen/Qwen3-0.6B and huihui-ai/Qwen3-0.6B-abliterated.
--plot enables visualization, generating heatmaps saved as a PNG (e.g., Qwen-Qwen3-0.6B_vs_huihui-ai-Qwen3-0.6B-abliterated_diffs.png).
