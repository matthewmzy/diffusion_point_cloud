import os
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.cond_vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *

# distributed training
import torch.multiprocessing as mp
import torch.distributed as dist


# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='flow', choices=['flow', 'gaussian'])
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--sample_num_points', type=int, default=1024)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/IBS_7d')
parser.add_argument('--categories', type=str_list, default=['ibs'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=1100)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--overfit', action='store_true')
parser.add_argument('--point_dim', type=int, default=4)
parser.add_argument('--condition_dim', type=int, default=256)
parser.add_argument('--scene_name', type=str, default='08000')

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=20000)
parser.add_argument('--sched_end_epoch', type=int, default=40000)

# Training
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_gen')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=30000)
parser.add_argument('--test_size', type=int, default=400)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_workers', action = 'store', type = int, default = 8)


args = parser.parse_args()
seed_all(args.seed)

# mp.set_sharing_strategy('file_system')
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
os.environ['NCCL_P2P_DISABLE'] = '1'
dist.init_process_group(backend='nccl', init_method='env://', world_size=WORLD_SIZE, rank=RANK)

# Logging
if RANK == 0:
    log_dir = get_new_log_dir(args.log_root, prefix='GEN_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
    logger.info(args)
else:
    writer = BlackHole()

# Set up device
torch.cuda.set_device(LOCAL_RANK)
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Datasets and loaders
if RANK == 0:
    logger.info('Loading datasets...')

if 'IBS' in args.dataset_path:
    train_dset = IBSDataset(
        path=args.dataset_path,
        split='train',
        overfit=args.overfit,
        point_dim=args.point_dim,
    )
    val_dset = IBSDataset(
        path=args.dataset_path,
        split='val',
        overfit=args.overfit,
        point_dim=args.point_dim,
    )
else:
    train_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='train',
        scale_mode=args.scale_mode,
    )
    val_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='val',
        scale_mode=args.scale_mode,
    )
sampler = torch.utils.data.distributed.DistributedSampler(
    train_dset,
    num_replicas=WORLD_SIZE,
    rank=RANK,
    shuffle=True
)
train_loader = DataLoader(
    train_dset,
    batch_size=args.train_batch_size // WORLD_SIZE,
    num_workers=args.num_workers,
    sampler=sampler
)
test_loader = DataLoader(
    val_dset,
    batch_size=args.val_batch_size,
    num_workers=args.num_workers,
)

# Model
if RANK == 0:
    logger.info('Building model...')
if args.model == 'gaussian':
    model = GaussianVAE(args).to(args.device)
elif args.model == 'flow':
    model = ConditionalFlowVAE(args).to(args.device)
if RANK == 0:
    logger.info(repr(model))
if args.spectral_norm:
    add_spectral_norm(model, logger=logger)

model = nn.parallel.DistributedDataParallel(
    model,
    device_ids=[LOCAL_RANK],
    output_device=LOCAL_RANK,
)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Load checkpoint
it = 0
if args.ckpt is not None:
    if RANK == 0:
        logger.info('Loading checkpoint from %s...' % args.ckpt)
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['others']['optimizer'])
    scheduler.load_state_dict(ckpt['others']['scheduler'])
    it = int(args.ckpt.split('_')[-1].split('.')[0])
    if RANK == 0:
        logger.info('Loaded checkpoint from %s, iteration %d' % (args.ckpt, it))

def train_dist(it):
    # Load data
    model.train()
    if args.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)
    while it < args.max_iters:
        sampler.set_epoch(it)
        optimizer.zero_grad()
        num_steps = len(train_loader)
        pbar = tqdm(train_loader) if RANK == 0 else train_loader
        avg_loss = 0

        for data in pbar:
            x = data['pointcloud'].to(args.device)
            c = data['scene_pc'].to(args.device)
            # loss = model.get_loss(x, c, kl_weight=args.kl_weight, writer=writer, it=it)
            loss = model.module.get_loss(x, c, kl_weight=args.kl_weight, writer=writer, it=it)
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            avg_loss += loss.item()

        avg_loss = avg_loss / num_steps

        dist.barrier()
        dist.all_reduce(torch.tensor([avg_loss], dtype = torch.float64, device = args.device), op = torch.distributed.ReduceOp.AVG)

        if RANK==0:
            logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
                it, loss.item(), orig_grad_norm, args.kl_weight
            ))
            writer.add_scalar('train/loss', loss, it)
            writer.add_scalar('train/kl_weight', args.kl_weight, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad_norm', orig_grad_norm, it)
            writer.flush()
            if it % args.val_freq == 0 or it % args.test_freq == 0 or it == args.max_iters:
                # t = test_loader[0] # not subscriptable
                t = next(iter(test_loader))
                ref_ibs = t['pointcloud'].to(args.device)
                test_scene_pc = t['scene_pc'].to(args.device)
                print(ref_ibs.shape, test_scene_pc.shape)
                if it % args.val_freq == 0:
                    validate_inspect(it, ref_ibs, test_scene_pc)
                    opt_states = {
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }
                    ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
                if it % args.test_freq == 0:
                    test(it, ref_ibs, test_scene_pc)
        it += 1

def validate_inspect(it, r, c):
    z = torch.randn([args.num_samples, args.latent_dim]).to(args.device)
    c = c[:args.num_samples]
    r = r[:args.num_samples]
    x = model.module.sample(z, c, args.sample_num_points, flexibility=args.flexibility, truncate_std=args.truncate_std)
    if RANK == 0:
        x_color = torch.zeros((x.shape[0], x.shape[1], 3),device=x.device)
        x_interpolate = x[:,:,3]
        x_interpolate[x_interpolate<0.01] = 0
        mask = torch.logical_and(x_interpolate > 0.01, x_interpolate < 0.11)
        x_interpolate[mask] = (x_interpolate[mask] - 0.01) / 0.1
        x_interpolate[x_interpolate>0.11] = 1
        x_color[:,:,0] = x_interpolate
        x_color[:,:,1] = 1-x_interpolate
        r_color = torch.zeros((r.shape[0], r.shape[1], 3),device=r.device)
        r_interpolate = r[:,:,3]
        r_interpolate[r_interpolate<0.01] = 0
        mask = torch.logical_and(r_interpolate > 0.01, r_interpolate < 0.11)
        r_interpolate[mask] = (r_interpolate[mask] - 0.01) / 0.1
        r_interpolate[r_interpolate>0.11] = 1
        r_color[:,:,0] = r_interpolate
        r_color[:,:,1] = 1-r_interpolate
        scene_pc_color = torch.zeros_like(c).to(x.device)
        x_c_color = torch.cat([x_color, scene_pc_color], dim=1)
        x_c = torch.cat([x[:,:,:3], c], dim=1)
        r_c_color = torch.cat([r_color, scene_pc_color], dim=1)
        r_c = torch.cat([r[:,:,:3], c], dim=1)
        # print(f"x_c_shape: {x_c.shape}, r_c_shape: {r_c.shape}")
        # print(f"x_c_color_shape: {x_c_color.shape}, r_c_color_shape: {r_c_color.shape}")
        # print(f"x_c_color_type: {x_c_color.dtype}, r_c_color_type: {r_c_color.dtype}")
        # print(f"x_c_color_max: {x_c_color.max()}, r_c_color_max: {r_c_color.max()}")
        # print(f"x_c_color_min: {x_c_color.min()}, r_c_color_min: {r_c_color.min()}")
        writer.add_mesh('val/ibs_only', vertices=x[:,:,:3], global_step=it, colors=x_c_color)
        writer.add_mesh('val/pointcloud', vertices=x_c, global_step=it, colors=x_c_color)
        writer.add_mesh('val/ref_ibs', vertices=r_c, global_step=it, colors=r_c_color)
        writer.flush()
        logger.info('[Inspect] Generating samples...')

def test(it, r, c):
    ref_pcs = []
    for i, data in enumerate(val_dset):
        if i >= args.test_size:
            break
        ref_pcs.append(data['pointcloud'].unsqueeze(0))
    ref_pcs = torch.cat(ref_pcs, dim=0)
    gen_pcs = []
    for i in tqdm(range(0, math.ceil(args.test_size / args.val_batch_size)), 'Generate'):
        with torch.no_grad():
            z = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
            x = model.module.sample(z, c, args.sample_num_points, flexibility=args.flexibility)
            gen_pcs.append(x.detach().cpu())
    gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_size]

    # Denormalize point clouds, all shapes have zero mean.
# [WARNING]: Do NOT denormalize!
    # ref_pcs *= val_dset.stats['std']
    # gen_pcs *= val_dset.stats['std']

    with torch.no_grad():
        print(f"gen_pcs.shape: {gen_pcs.shape}, ref_pcs.shape: {ref_pcs.shape}")
        results = compute_all_metrics(gen_pcs[:,:,:3].to(args.device), ref_pcs[:,:,:3].to(args.device), args.val_batch_size)
        results = {k:v.item() for k, v in results.items()}
        jsd = jsd_between_point_cloud_sets(gen_pcs[:,:,:3].cpu().numpy(), ref_pcs[:,:,:3].cpu().numpy())
        results['jsd'] = jsd
    if RANK == 0:
        # CD related metrics
        writer.add_scalar('test/Coverage_CD', results['lgan_cov-CD'], global_step=it)
        writer.add_scalar('test/MMD_CD', results['lgan_mmd-CD'], global_step=it)
        writer.add_scalar('test/1NN_CD', results['1-NN-CD-acc'], global_step=it)
        # EMD related metrics
        # writer.add_scalar('test/Coverage_EMD', results['lgan_cov-EMD'], global_step=it)
        # writer.add_scalar('test/MMD_EMD', results['lgan_mmd-EMD'], global_step=it)
        # writer.add_scalar('test/1NN_EMD', results['1-NN-EMD-acc'], global_step=it)
        # JSD
        writer.add_scalar('test/JSD', results['jsd'], global_step=it)

        # logger.info('[Test] Coverage  | CD %.6f | EMD %.6f' % (results['lgan_cov-CD'], results['lgan_cov-EMD']))
        # logger.info('[Test] MinMatDis | CD %.6f | EMD %.6f' % (results['lgan_mmd-CD'], results['lgan_mmd-EMD']))
        # logger.info('[Test] 1NN-Accur | CD %.6f | EMD %.6f' % (results['1-NN-CD-acc'], results['1-NN-EMD-acc']))
        logger.info('[Test] Coverage  | CD %.6f | EMD n/a' % (results['lgan_cov-CD'], ))
        logger.info('[Test] MinMatDis | CD %.6f | EMD n/a' % (results['lgan_mmd-CD'], ))
        logger.info('[Test] 1NN-Accur | CD %.6f | EMD n/a' % (results['1-NN-CD-acc'], ))
        logger.info('[Test] JsnShnDis | %.6f ' % (results['jsd']))

# Main loop
if RANK == 0:
    logger.info('Start training...')
try:
    scene_pc = np.load(os.path.join(args.dataset_path, "scene_pc", f"scene_{args.scene_name}.npy"))
    train_dist(it)

except KeyboardInterrupt:
    if RANK == 0:
        logger.info('Terminating...') 