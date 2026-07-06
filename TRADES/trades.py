import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def _apply_preprocess(x, preprocess):
    if preprocess is None:
        return x
    return preprocess(x)


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def _build_bce_targets(logits, target, off_label=None):
    num_classes = logits.shape[-1]
    if target.shape != logits.shape:
        target = F.one_hot(target, num_classes=num_classes).to(dtype=logits.dtype)
    else:
        target = target.to(dtype=logits.dtype)
    if off_label is not None:
        target = target.clamp(min=off_label)
    return target


def _robust_consistency_loss(
    logits_adv,
    logits_nat,
    natural_loss,
    criterion_kl,
):
    if natural_loss in {"bce", "bce_uniform"}:
        # BCE-style consistency: match per-class Bernoulli targets from clean logits.
        soft_targets = torch.sigmoid(logits_nat).detach()
        return F.binary_cross_entropy_with_logits(logits_adv, soft_targets)
    return criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf',
                natural_loss='ce',
                bce_off_label=None,
                use_robust_loss=True,
                preprocess=None,
                clip_min=0.0,
                clip_max=1.0):
    criterion_kl = nn.KLDivLoss(reduction='sum')
    batch_size = len(x_natural)
    device = x_natural.device

    if use_robust_loss:
        model.eval()
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=device).detach()
        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    x_adv_model = _apply_preprocess(x_adv, preprocess)
                    x_nat_model = _apply_preprocess(x_natural, preprocess)
                    logits_adv = model(x_adv_model)
                    logits_nat = model(x_nat_model)
                    loss_robust_inner = _robust_consistency_loss(
                        logits_adv=logits_adv,
                        logits_nat=logits_nat,
                        natural_loss=natural_loss,
                        criterion_kl=criterion_kl,
                    )
                grad = torch.autograd.grad(loss_robust_inner, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, clip_min, clip_max)
        elif distance == 'l_2':
            delta = 0.001 * torch.randn(x_natural.shape, device=device).detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    adv_model = _apply_preprocess(adv, preprocess)
                    x_nat_model = _apply_preprocess(x_natural, preprocess)
                    logits_adv = model(adv_model)
                    logits_nat = model(x_nat_model)
                    loss = (-1) * _robust_consistency_loss(
                        logits_adv=logits_adv,
                        logits_nat=logits_nat,
                        natural_loss=natural_loss,
                        criterion_kl=criterion_kl,
                    )
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

        model.train()
        x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    else:
        x_adv = None
        model.train()

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    x_nat_model = _apply_preprocess(x_natural, preprocess)
    logits = model(x_nat_model)
    if natural_loss in {'bce', 'bce_uniform'}:
        off_label = bce_off_label if natural_loss == 'bce_uniform' else None
        bce_target = _build_bce_targets(logits, y, off_label=off_label)
        loss_natural = F.binary_cross_entropy_with_logits(logits, bce_target)
    else:
        loss_natural = F.cross_entropy(logits, y)

    if use_robust_loss:
        x_adv_model = _apply_preprocess(x_adv, preprocess)
        logits_adv = model(x_adv_model)
        logits_nat = model(x_nat_model)
        loss_robust = _robust_consistency_loss(
            logits_adv=logits_adv,
            logits_nat=logits_nat,
            natural_loss=natural_loss,
            criterion_kl=criterion_kl,
        )
        if natural_loss not in {"bce", "bce_uniform"}:
            loss_robust = (1.0 / batch_size) * loss_robust
        loss = loss_natural + beta * loss_robust
    else:
        loss = loss_natural
    return loss
