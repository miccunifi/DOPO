
from .t2m_dataset import HumanML3D, KIT, HumanML3D_no_manipulation, HumanML3D_no_posture, HumanML3D_no_performance, \
    KIT_generated_momask, KIT_22, HumanML3D_generated_momask, KIT_generated_motiongpt, HumanML3D_generated_motiongpt, \
    KIT_generate, KITML_SMPL, HumanML3D_SMPL, BABEL_SMPL, MOTIONX_SMPL

from os.path import join as pjoin
__all__ = [
    'HumanML3D', 'KIT',  'get_dataset',]

def get_dataset(opt, split='train', mode='train', accelerator=None):
    if opt.dataset_name == 't2m' :
        dataset = HumanML3D(opt, split, mode, accelerator)
    elif opt.dataset_name == 'kit' :
        dataset = KIT(opt,split, mode, accelerator)
    elif opt.dataset_name == 'kit_generated_momask':
        dataset = KIT_generated_momask(opt,split, mode, accelerator)
    elif opt.dataset_name == 'KIT_generate':
        dataset = KIT_generate(opt, split, mode, accelerator)
    elif opt.dataset_name == 'kit_generated_motiongpt':
        dataset = KIT_generated_motiongpt(opt,split, mode, accelerator)
    elif opt.dataset_name == 'kit_22'or opt.dataset_name == 'kitml':
        dataset = KIT_22(opt,split, mode, accelerator)
    elif opt.dataset_name == 'manipulation' :
        dataset = HumanML3D_no_manipulation(opt,split, mode, accelerator)
    elif opt.dataset_name == 'posture':
        dataset = HumanML3D_no_posture(opt,split, mode, accelerator)
    elif opt.dataset_name == 'performance' :
        dataset = HumanML3D_no_performance(opt,split, mode, accelerator)
    elif opt.dataset_name == 'humanml3d_generated_momask':
        dataset = HumanML3D_generated_momask(opt,split, mode, accelerator)
    elif opt.dataset_name == 'humanml3d_generated_motiongpt':
        dataset = HumanML3D_generated_motiongpt(opt,split, mode, accelerator)
    elif opt.dataset_name == 'humanml3d_smpl':
        dataset = HumanML3D_SMPL(opt,split, mode, accelerator)
    elif opt.dataset_name == 'kitml_smpl':
        dataset = KITML_SMPL(opt,split, mode, accelerator)
    elif opt.dataset_name == 'babel_smpl':
        dataset = BABEL_SMPL(opt,split, mode, accelerator)
    elif opt.dataset_name == 'motionx_smpl':
        dataset = MOTIONX_SMPL(opt,split, mode, accelerator)
    else:
        raise KeyError('Dataset Does Not Exist')
    
    if accelerator:
        accelerator.print('Completing loading %s dataset' % opt.dataset_name)
    else:
        print('Completing loading %s dataset' % opt.dataset_name)
    
    return dataset

