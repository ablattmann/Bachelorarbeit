from pytorch_lightning.metrics import functional as PF


def ssim(real, fake):
    # if normalize_range:
    #     real = (real + 1.) /2.
    #     fake = (fake + 1.) / 2.

    ssim_batch = PF.ssim(fake,real).cpu().numpy()

    return ssim_batch


def psnr(real, fake):

    # if normalize_range:
    #     real = (real + 1.) / 2.
    #     fake = (fake + 1.) / 2.

    psnr_batch = PF.psnr(fake, real).cpu().numpy()

    return psnr_batch