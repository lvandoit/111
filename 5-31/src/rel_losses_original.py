import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
from operator import mul

def relative_loss(pos_3d, rel_inds, rel_gt, distance_multiplier):
    # rel_inds are the indices of the depth values  Bx2 matrix
    # rel_gt is the relative label -1, 1, 0         Bx1 vector

    d1 = pos_3d[list(range(pos_3d.size(0))), rel_inds[:, 0]]
    d2 = pos_3d[list(range(pos_3d.size(0))), rel_inds[:, 1]]
    gt = rel_gt[:,0]
    mask = torch.abs(gt)

    dis = (d1-d2) / torch.abs(d1-d2).mean()

    # cr = torch.log(1.0+torch.exp(-gt*dis))
    # NOTE: cap the maximum value to 85 so that the exp doesn't overflow
    capped_dist = torch.min(85*Variable(torch.ones(dis.shape).cuda()),
                            -gt*dis*distance_multiplier)
    # capped_dist = -gt*dis*distance_multiplier
    cr = torch.log1p(torch.exp(capped_dist))
    ce = dis*dis
    ranking_loss = mask*cr+(1.0-mask)*ce

    return ranking_loss.mean()

def reproj_loss_scaled_orthographic(pos_3d, img_targ_coords, scale_params):
    batch_size = pos_3d.shape[0]
    num_keypoints = int(img_targ_coords.shape[1] / 2)

    pred_x = pos_3d[:,0::3]
    pred_y = pos_3d[:,1::3]
     
    sqr_dist = (pred_x * scale_params - img_targ_coords[:,0::2])**2 + \
               (pred_y * scale_params - img_targ_coords[:,1::2])**2
    reproj_loss = torch.sqrt(sqr_dist + 1e-8).mean(dim=1) # mean per frame

    return reproj_loss.mean()

def reproj_loss_estimated_weak_perspective(pos_3d, img_targ_coords, scale_params, f_batch):
    batch_size = pos_3d.shape[0]
    num_keypoints = int(img_targ_coords.shape[1] / 2)

    pred_x = pos_3d[:,0::3]
    pred_y = pos_3d[:,1::3]
    pred_z = pos_3d[:,2::3] + scale_params
    proj_x = (pred_x/pred_z)
    proj_y = (pred_y/pred_z)

    sqr_dist = (proj_x * f_batch[:,0] - img_targ_coords[:,0::2])**2 + \
               (proj_y * f_batch[:,1] - img_targ_coords[:,1::2])**2
    reproj_loss = torch.sqrt(sqr_dist + 1e-8).mean(dim=1) # mean per frame

    return reproj_loss.mean()


def camera_coord_3d_loss(misc, pose_3d_norm, pose_3d_unnorm, gt_limb_lens, loss_weights):
    # pose_3d_norm is the predicted 3D keypoints - Bx(Nx3)

    assert pose_3d_norm.size() == pose_3d_unnorm.size()
    assert pose_3d_norm.size(1) == misc.NUM_KEYPOINTS_3D * 3

    use_symm_loss = True if loss_weights['symm'] > 0.0 else False

    ############################################################################
    # Root Loss: constrain root to be at (0,0,0)
    '''
    root_id = misc.ROOT_IDX_3D
    root_loss = torch.sqrt((pose_3d_norm[:, [root_id*3+0, root_id*3+1, root_id*3+2]]**2).sum(1) + 1e-8)
    '''
    
    ############################################################################
    # Ratio Loss: constrain ratio and symmetry in limb lengths
    if use_symm_loss:
        batch_size = pose_3d_unnorm.shape[0]
        pose_3d_unnorm_rs = pose_3d_unnorm.view(batch_size,-1,3)

        joint_1 = [k[0] for k in misc.SKELETON_3D_IDX]
        joint_2 = [k[1] for k in misc.SKELETON_3D_IDX]
        skel_dis = torch.sqrt(((pose_3d_unnorm_rs[:, joint_1, :] - pose_3d_unnorm_rs[:, joint_2, :])**2).sum(2) + 1e-8)

        # normalize the length of the skeleton based on enforcing unit length for a fixed joint
        # this will enforce correct ratio and, by extension, symmetry
        skel_dis_norm = skel_dis / skel_dis[:, misc.UNIT_LENGTH_LIMB_IDX].unsqueeze(1)
        # skel_dis_norm = skel_dis
        gt_limb_lens_norm = gt_limb_lens / gt_limb_lens[:, misc.UNIT_LENGTH_LIMB_IDX].unsqueeze(1)
        # gt_limb_lens_norm = gt_limb_lens

        valid_idxs = torch.nonzero(gt_limb_lens[0]!=-1).view(-1)
        # skel_ratio = torch.sqrt(((skel_dis_norm - gt_limb_lens_norm)**2) + 1e-8).mean(1)
        skel_ratio = torch.sqrt(((skel_dis_norm[:,valid_idxs] - gt_limb_lens_norm[:,valid_idxs])**2) + 1e-8).mean(1)

    # combine all the losses
    #############################################################
    #loss = loss_weights['root'] * root_loss.mean()
    loss=0
    #############################################################
    if use_symm_loss:
        loss += loss_weights['symm'] * skel_ratio.mean()

    return loss
####################################################################
#jia ru heatmap loss 
def heatmap_loss(t,tag):
    loss=nn.MSELoss()
    final_loss=loss(t,tag)
    return final_loss

####################################################################
def _normalized_linspace(length, dtype=None, device=None):
   
    first = -(length - 1) / length
    last = (length - 1) / length
    return torch.linspace(first, last, length, dtype=dtype, device=device)


def make_gauss(means, size, sigma, normalize=True):
    

    dim_range = range(-1, -(len(size) + 1), -1)
    coords_list = [_normalized_linspace(s, dtype=means.dtype, device=means.device)
                   for s in reversed(size)]

    # PDF = exp(-(x - \mu)^2 / (2 \sigma^2))

    # dists <- (x - \mu)^2
    dists = [(x - mean) ** 2 for x, mean in zip(coords_list, means.split(1, -1))]

    # ks <- -1 / (2 \sigma^2)
    stddevs = [2 * sigma / s for s in reversed(size)]
    ks = [-0.5 * (1 / stddev) ** 2 for stddev in stddevs]

    exps = [(dist * k).exp() for k, dist in zip(ks, dists)]

    # Combine dimensions of the Gaussian
    gauss = reduce(mul, [
        reduce(lambda t, d: t.unsqueeze(d), filter(lambda d: d != dim, dim_range), dist)
        for dim, dist in zip(dim_range, exps)
    ])

    if not normalize:
        return gauss

    # Normalize the Gaussians
    val_sum = reduce(lambda t, dim: t.sum(dim, keepdim=True), dim_range, gauss) + 1e-24
    return gauss / val_sum


def _kl(p, q, ndims):
    eps = 1e-24
    unsummed_kl = p * ((p + eps).log() - (q + eps).log())
    kl_values = reduce(lambda t, _: t.sum(-1, keepdim=False), range(ndims), unsummed_kl)
    return kl_values


def _js(p, q, ndims):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, ndims) + 0.5 * _kl(q, m, ndims)


def _divergence_reg_losses(heatmaps, mu_t, sigma_t, divergence):
    ndims = mu_t.size(-1)
    assert heatmaps.dim() == ndims + 2, 'expected heatmaps to be a {}D tensor'.format(ndims + 2)
    assert heatmaps.size()[:-ndims] == mu_t.size()[:-1]

    gauss = make_gauss(mu_t, heatmaps.size()[2:], sigma_t)
    divergences = divergence(heatmaps, gauss, ndims)
    return divergences


def js_reg_losses(heatmaps, mu_t, sigma_t):
    """Calculate Jensen-Shannon divergences between heatmaps and target Gaussians.
    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        mu_t (torch.Tensor): Centers of the target Gaussians (in normalized units)
        sigma_t (float): Standard deviation of the target Gaussians (in pixels)
    Returns:
        Per-location JS divergences.
    """

    return _divergence_reg_losses(heatmaps, mu_t, sigma_t, _js)

#调用js_reg_losses()即可，js_reg_losses(hm,target_coords,sigma=1)
#优化网络的L2约束
'''
def refine_loss(d_true,d_fake):
    loss1=torch.nn.MSELoss()
    d1_true=d_true[:,0:4]
    d2_true=d_true[:,8:16]
    d3_true=d_true[:,20:24]
    d_true=torch.stack((d1_true,d3_true),0).view(8,8)
    d_true=torch.stack((d_true,d2_true),0).view(8,16)
    
    
    d1_fake=d_fake[:,0:4]
    d2_fake=d_fake[:,8:16]
    d3_fake=d_fake[:,20:24]
    d_fake=torch.stack((d1_fake,d3_fake),0).view(8,8)
    d_fake=torch.stack((d_fake,d2_fake),0).view(8,16)
    loss=loss1(d_true,d_fake)
    return loss
'''
def wei_loss(d_true,d_fake):
    loss2=torch.nn.MSELoss()
    loss=loss2(d_true,d_fake)
    return loss
def refine_loss(d_true,d_fake):
    loss1=torch.nn.MSELoss()
    loss=loss1(d_true,d_fake)
    return loss

      
