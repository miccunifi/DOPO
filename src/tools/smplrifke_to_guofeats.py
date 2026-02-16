import numpy as np
import torch
from src.tools.smpl_layer import SMPLH
from src.tools.extract_joints import extract_joints
from src.tools.guofeats.motion_representation import joints_to_guofeats


smplh = SMPLH(
        path="deps/smplh",
        jointstype="both",
        input_pose_rep="axisangle",
        gender="male",
)


def smpl_to_guofeats(smpl):
    guofeats = []

    if isinstance(smpl, list):
        device = smpl[0].device
    else:
        device = smpl.device
    
    is_tesnor = isinstance(smpl, torch.Tensor)

    smplh.to(device)

    for i in smpl:
        i_output = extract_joints(
            i,
            'smplrifke',
            fps=20,
            value_from='smpl',
            smpl_layer=smplh,
        )
        i_joints = i_output["joints"]  # tensor(N, 22, 3)

        if isinstance(i_joints, torch.Tensor):
            i_joints = i_joints.cpu().numpy()

        # convert to guofeats, first, make sure to revert the axis, as guofeats have gravity axis in Y
        x, y, z = i_joints.T
        i_joints = np.stack((x, z, -y), axis=0).T
        i_guofeats = joints_to_guofeats(i_joints)

        if is_tesnor:
            i_guofeats = torch.from_numpy(i_guofeats).to(device)

        guofeats.append(i_guofeats)

    return guofeats
