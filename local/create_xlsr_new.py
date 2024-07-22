# https://github.com/facebookresearch/fairseq/issues/3741

from omegaconf import DictConfig, OmegaConf, open_dict
import torch

cp_path = 'model/xlsr_53_56k.pt'
cp = torch.load(cp_path)
cfg = DictConfig(cp['cfg'])
dd = OmegaConf.to_container(cfg, resolve=True)
for k,v in dd.items():
    if not isinstance(v, dict):
        continue
    for key, _ in v.items():
        if key.split("_")[:2] == ["eval", "wer"]:
            print(k,key)
with open_dict(cfg):
    cfg.task.pop('eval_wer')
    cfg.task.pop('eval_wer_config')
    cfg.task.pop('eval_wer_tokenizer')
    cfg.task.pop('eval_wer_post_process')
    cfg.task.pop('autoregressive')
cp['cfg'] = cfg
torch.save(cp, 'model/xlsr_53_56k_new.pt')