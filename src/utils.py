from omegaconf import DictConfig, OmegaConf

def set_preload_false(cfg):
    """
    Imposta preload=False ricorsivamente in tutta la configurazione.
    
    Args:
        cfg: OmegaConf DictConfig
    """
    if isinstance(cfg, DictConfig):
        # Se ha il parametro preload, impostalo a False
        if "preload" in cfg:
            OmegaConf.update(cfg, "preload", False, merge=False)
        
        # Ricorsione su tutti i sotto-dizionari
        for key in cfg:
            if isinstance(cfg[key], (DictConfig, dict)):
                set_preload_false(cfg[key])
    
    elif isinstance(cfg, dict):
        # Caso normale dict
        if "preload" in cfg:
            cfg["preload"] = False
        
        for key in cfg:
            if isinstance(cfg[key], (DictConfig, dict)):
                set_preload_false(cfg[key])


def check_tensor_stats(tensor):
    mean = tensor.mean()
    max_val = tensor.max()
    min_val = tensor.min()
    
    # Soglie basate sui tuoi esempi
    max_threshold = 2  # valori ok sono < 2, anomali > 3
    min_threshold = -2
    
    if abs(max_val) > max_threshold or abs(min_val) > abs(min_threshold):
        pass
    else:
        print(f"⚠️  -  WARNING: tensor has extreme values! Looks it has NOT been normalized")
        print(f"   Mean: {mean:.4f}, Max: {max_val:.4f}, Min: {min_val:.4f}")
        return False
    
    return True
