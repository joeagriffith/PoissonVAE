import torch
from torch.utils.data import DataLoader
from functional import cosine_schedule
from tqdm import tqdm
from evaluate import knn_analysis


def train(
        model,
        train_dataset,
        val_dataset,
        n_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 2e-3,
        wd: float = 3e-4,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        beta_portion: float = 0.6,
        temp_start: float = 1.0,
        temp_end: float = 0.05,
        temp_portion: float = 0.5,
        grad_clip: float = None,
):

    device = next(model.parameters()).device
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    lr_scheduler = None
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs/5)

    # Set beta and temp schedules if required
    betas, temps = None, None
    if hasattr(model, 'set_beta'):
        betas = torch.ones(n_epochs) * beta_end
        betas[:int(n_epochs * beta_portion)] = cosine_schedule(beta_start, beta_end, int(n_epochs * beta_portion))
    if hasattr(model, 'set_temp'):
        temps = torch.ones(n_epochs) * temp_end
        temps[:int(n_epochs * temp_portion)] = torch.linspace(temp_start, temp_end, int(n_epochs * temp_portion))

    # For logging epochs
    logs = {
        'train_losses': [],
        'train_recon_losses': [],
        'train_kl_losses': [],
        'val_losses': [],
        'val_recon_losses': [],
        'val_kl_losses': [],
        'knn_acc': [],
    }

    # Training epoch loop
    for epoch in range(n_epochs):
        model.train()

        # Update beta and temp if required
        if betas is not None:
            model.set_beta(betas[epoch])
        if temps is not None:
            model.set_temp(temps[epoch])

        # Init logging for training epoch
        loss_total = 0.0
        recon_loss_total = 0.0
        kl_loss_total = 0.0 if hasattr(model, 'beta') else None

        # Training batch loop
        loop = tqdm(train_loader, total=len(train_loader), leave=False)
        for x, _ in loop:
            if epoch > 0:
                loop.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
                postfix = {
                    'loss': (round(logs['train_losses'][-1], 3), round(logs['val_losses'][-1], 3)),
                    'recon_loss': (round(logs['train_recon_losses'][-1], 3), round(logs['val_recon_losses'][-1], 3)),
                }
                if hasattr(model, 'beta'):
                    postfix['kl_loss'] = (round(logs['train_kl_losses'][-1], 3), round(logs['val_kl_losses'][-1], 3))
                postfix['knn_acc'] = round(logs['knn_acc'][-1], 3)
                loop.set_postfix(**postfix)

            # Inference
            x = x.to(device)
            out = model(x)
            loss, recon_loss, kl_loss = model.loss(x, out)

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                max_norm = grad_clip * 3
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            # Accumulate training batch for logging
            loss_total += loss.item()
            recon_loss_total += recon_loss.item()
            if hasattr(model, 'beta'):
                kl_loss_total += kl_loss.item()

        if lr_scheduler:
            lr_scheduler.step()

        # Log training epoch
        logs['train_losses'].append(loss_total / len(train_loader))
        logs['train_recon_losses'].append(recon_loss_total / len(train_loader))
        if hasattr(model, 'beta'):
            logs['train_kl_losses'].append(kl_loss_total / len(train_loader))
        
        # Init logging for validation epoch
        val_loss_total = 0.0
        val_recon_loss_total = 0.0
        val_kl_loss_total = 0.0 if hasattr(model, 'beta') else None
        model.eval()

        # Validation Batch Loop
        for x, _ in val_loader:

            # Inference
            with torch.no_grad():
                x = x.to(device)
                out = model(x)
                loss, recon_loss, kl_loss = model.loss(x, out)

            # Accumulate validation batch for logging
            val_loss_total += loss.item()
            val_recon_loss_total += recon_loss.item()
            if hasattr(model, 'beta'):
                val_kl_loss_total += kl_loss.item()
        
        # Log validation epoch
        knn_acc = knn_analysis(model, val_dataset)
        logs['knn_acc'].append(knn_acc)
        logs['val_losses'].append(val_loss_total / len(val_loader))
        logs['val_recon_losses'].append(val_recon_loss_total / len(val_loader))
        if hasattr(model, 'beta'):
            logs['val_kl_losses'].append(val_kl_loss_total / len(val_loader))
        
    return logs
    