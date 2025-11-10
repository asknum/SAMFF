import logging
logger = logging.getLogger('base')
import torch
import torch.nn as nn
import models.samff as SAMFF

def create_CD_model(opt):
    cd_model = opt['model']['name'].upper()
    
    if cd_model == "SAMFF":
        decoder_type = opt['model'].get('decoder_type', 'ifa')
        use_pfem = opt['model'].get('use_pfem', True)
        model = SAMFF.BiFA(backbone='mit_b0', use_pfem=use_pfem, decoder_type=decoder_type)
        
    else:
        raise NotImplementedError(f"Model {opt['model']['name']} not implemented!")
        
    return model