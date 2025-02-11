import torch
from torch import nn
import torch.nn.functional as F
from .common import *
from .encoders import *
from .diffusion import *
from .flow import *



class ConditionalFlowVAE(nn.Module):
    """
    VAE model with Conditional Flow (Normalizing Flow) on the latent space.
    """
    def __init__(self, args):
        super(ConditionalFlowVAE, self).__init__()
        self.args = args
        # Initialize the encoder with conditional information
        self.encoder = PointNetEncoder(args.latent_dim, input_dim=args.point_dim)
        self.scene_encoder = PointNetEncoder(args.condition_dim, input_dim=3)
        # Build conditional flow layers
        self.flow = build_latent_flow(args)
        # Initialize the diffusion model
        self.diffusion = DiffusionPoint(
            net=PointwiseNet(point_dim=args.point_dim, context_dim=args.latent_dim+args.condition_dim, residual=args.residual),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def get_loss(self, x, c, kl_weight, writer=None, it=None):
        """
        Calculate the VAE loss: reconstruction + KL divergence
        Args:
            x: Input point clouds (batch_size, N, d)
            c: Condition (e.g., label, text, etc.)
            kl_weight: Weight for KL divergence term
        """
        batch_size, _, _ = x.size()
        
        # Step 1: Get latent variables z_mu and z_sigma (encoder output)
        z_mu, z_sigma = self.encoder(x)
        z_prime_mu, z_prime_sigma = self.scene_encoder(c)
        
        # Step 2: Reparameterization to sample latent variable z
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
        z_prime = reparameterize_gaussian(mean=z_prime_mu, logvar=z_prime_sigma)
        
        # Step 3: Apply normalizing flow to the latent variable z
        # Here, w is the transformed latent variable in the prior space
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)  # Reverse=False for forward pass
        
        # Log probability under standard normal prior
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)

        # Step 4: Calculate the entropy term (H[Q(z|X)])
        entropy = gaussian_entropy(logvar=z_sigma)  # Entropy of the variational distribution
        
        # Step 5: Calculate negative ELBO of P(X|z) using diffusion model
        neg_elbo = self.diffusion.get_loss(x, torch.cat([z, z_prime], dim=1))  # Diffusion loss
        
        # Step 6: Compute total loss
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = neg_elbo
        
        loss = kl_weight * (loss_entropy + loss_prior) + neg_elbo
        
        # Optionally log the losses and other metrics for visualization
        if writer is not None:
            writer.add_scalar('train/loss_entropy', loss_entropy, it)
            writer.add_scalar('train/loss_prior', loss_prior, it)
            writer.add_scalar('train/loss_recons', loss_recons, it)
            writer.add_scalar('train/z_mean', z_mu.mean(), it)
            writer.add_scalar('train/z_mag', z_mu.abs().max(), it)
            writer.add_scalar('train/z_var', (0.5 * z_sigma).exp().mean(), it)
        
        return loss

    def sample(self, w, c, num_points, flexibility, truncate_std=None):
        """
        Sampling new point clouds from the model.
        Args:
            w: Latent variable samples (batch_size, latent_dim)
            num_points: Number of points to generate per sample
            flexibility: Parameter controlling the variability of the generated sample
            truncate_std: Optionally truncate the standard deviation for sampling
        """
        batch_size, _ = w.size()
        c_mu, c_sigma = self.scene_encoder(c)
        c = reparameterize_gaussian(mean=c_mu, logvar=c_sigma)
        
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        
        # Reverse pass: Use the flow to transform latent variables back to the original space
        z = self.flow(w, reverse=True).view(batch_size, -1)
        
        # Generate the samples using the diffusion model conditioned on z
        samples = self.diffusion.sample(num_points, context=torch.cat([z, c], dim=1), flexibility=flexibility, point_dim=self.args.point_dim)
        
        return samples
