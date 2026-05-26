import torch
import torch.nn.functional as F

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def batch_euler2matzxy(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat_individual = torch.stack([zmat, xmat, ymat], dim=1)
    rotMat = ymat @ xmat @ zmat
    return rotMat, rotMat_individual


def rotmat2eulerzxy(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, x, y axes, respectively
    '''
    N = rotmat.size()[0]
    cx_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N,3).type(torch.float32).to(rotmat.device)

    r11, r12, r13 = rotmat[:,0,0],rotmat[:,0,1],rotmat[:,0,2]
    r21, r22, r23 = rotmat[:,1,0],rotmat[:,1,1],rotmat[:,1,2]
    r31, r32, r33 = rotmat[:,2,0],rotmat[:,2,1],rotmat[:,2,2]

    cx = torch.sqrt(torch.pow(r21,2) + torch.pow(r22,2))
    eulerangle[:,0] = torch.atan2( -r23,  cx)

    eulerangle[cx>cx_thresh,2] = torch.atan2(r21[cx>cx_thresh], r22[cx>cx_thresh])
    eulerangle[cx>cx_thresh,1] = torch.atan2(r13[cx>cx_thresh], r33[cx>cx_thresh])

    eulerangle[cx<=cx_thresh,2] = torch.atan2(-r12[cx<=cx_thresh], r11[cx<=cx_thresh])

    return eulerangle
