import numpy as np
import gpytorch
from gpytorch.kernels import PeriodicKernel, AdditiveKernel, ProductKernel, ScaleKernel


def weibull_noise(k=2, length=1, median=1):
    """
    Function to generate weibull noise with a fixed median
    """
    # we set lambda so that median is a given value
    lamda = median / (np.log(2) ** (1 / k))
    return lamda * np.random.weibull(k, length)


def shift_axis(days, shift):
    if shift is None:
        return days
    return days - shift * days[-1]


def get_random_walk_series(length, movements=[-1, 1]):
    """
    Function to generate a random walk series with a specified length
    """
    random_walk = list()
    random_walk.append(np.random.choice(movements))
    for i in range(1, length):
        movement = np.random.choice(movements)
        value = random_walk[i - 1] + movement
        random_walk.append(value)

    return np.array(random_walk)


def sample_scale(low_ratio = 0.6, moderate_ratio = 0.3):
    """
    Function to sample scale such that it follows 60-30-10 distribution
    i.e. 60% of the times it is very low, 30% of the times it is moderate and
    the rest 10% of the times it is high
    """
    rand = np.random.rand()
    # very low noise
    if rand <= low_ratio:
        return np.random.uniform(0, 0.1)
    # moderate noise
    elif rand <= (low_ratio + moderate_ratio):
        return np.random.uniform(0.2, 0.5)
    # high noise
    else:
        return np.random.uniform(0.7, 0.9)


def get_transition_coefficients(context_length):
    """
    Transition series refers to the linear combination of 2 series
    S1 and S2 such that the series S represents S1 for a period and S2
    for the remaining period. We model S as S = (1 - f) * S1 + f * S2
    Here f = 1 / (1 + e^{-k (x-m)}) where m = (a + b) / 2 and k is chosen
    such that f(a) = 0.1 (and hence f(b) = 0.9). a and b refer to
    0.2 * CONTEXT_LENGTH and 0.8 * CONTEXT_LENGTH
    """
    # a and b are chosen with 0.2 and 0.8 parameters
    a, b = 0.2 * context_length, 0.8 * context_length

    # fixed to this value
    f_a = 0.1

    m = (a + b) / 2
    k = 1 / (a - m) * np.log(f_a / (1 - f_a))

    coeff = 1 / (1 + np.exp(-k * (np.arange(1, context_length+1) - m)))
    return coeff


def random_binary_map(a: gpytorch.kernels.Kernel, b: gpytorch.kernels.Kernel):
    """
    Applies a random binary operator (+ or *) with equal probability
    on kernels ``a`` and ``b``.

    Parameters
    ----------
    a
        A GP kernel.
    b
        A GP kernel.

    Returns
    -------
        The composite kernel `a + b` or `a * b`.
    """
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return np.random.choice(binary_maps)(a, b)

def custom_gaussian_sample(max_period_length, kernel_periods=None, gaussian_sample=True, allow_extension=True):

    means = np.array(kernel_periods) if kernel_periods is not None else np.array([3, 5, 7, 14, 20, 21, 24, 30, 60, 90, 120])
    
    if allow_extension:
        if max_period_length > 200:
            st = max_period_length//2 if max(means) < max_period_length//2 else max(means) + 100
            means = np.append(means, np.arange(st, max_period_length, 100))
        else:
            if max(means) < max_period_length/2:
                means = np.append(means, np.array([max_period_length//2, max_period_length]))
            elif max(means) < max_period_length:
                means = np.append(means, max_period_length)
                
    means = means[means <= max_period_length]
    selected_mean = np.random.choice(means)

    if gaussian_sample:
        # Define corresponding standard deviations using np.sqrt(means) * 2
        std_devs = np.sqrt(means) ** 1.2 # / (means *0.008)
        selected_std = std_devs[np.where(means == selected_mean)][0]    
        sample = np.random.normal(selected_mean, selected_std)
    else:
        sample = selected_mean
    
    if sample < 1:
        sample = np.ceil(np.abs(sample))
    
    return int(sample)

def create_kernel(kernel: str, seq_len: int, max_period_length: int = 365, max_degree: int = 5, gaussians_periodic: bool = False, kernel_periods=None,
                  kernel_counter=None, freq=None, exact_freqs=False, gaussian_sample=True, subfreq=''):
    scale_kernel = np.random.choice([True, False])
    lengthscale = np.random.uniform(0.1, 5.0)
    if kernel == "linear_kernel":
        sigma_prior = gpytorch.priors.GammaPrior(np.random.uniform(1,6), np.random.uniform(0.1, 1))
        kernel = gpytorch.kernels.LinearKernel(variance_prior=sigma_prior)
    elif kernel == "rbf_kernel":
        kernel = gpytorch.kernels.RBFKernel()
        kernel.lengthscale = lengthscale
    elif kernel == "periodic_kernel":
        if gaussians_periodic:
            if exact_freqs and freq != "Y" and kernel_counter is not None:
                period_length = custom_gaussian_sample(max_period_length, 
                                                    kernel_periods=kernel_periods[:-3] if (kernel_counter["periodic_kernel"] <= 2) and (subfreq == '') else kernel_periods,
                                                    gaussian_sample=gaussian_sample, allow_extension=(kernel_counter["periodic_kernel"] > 2))
                kernel_counter["periodic_kernel"] -= 1
            else:
                period_length = custom_gaussian_sample(max_period_length, kernel_periods, gaussian_sample=True)
        else:
            period_length = np.random.randint(1, max_period_length)
        kernel = gpytorch.kernels.PeriodicKernel()
        kernel.period_length = period_length/seq_len
        kernel.lengthscale = lengthscale
    elif kernel == "polynomial_kernel":
        offset_prior = gpytorch.priors.GammaPrior(np.random.uniform(1,4), np.random.uniform(0.1, 1))
        degree = np.random.randint(1, max_degree)
        kernel = gpytorch.kernels.PolynomialKernel(offset_prior=offset_prior, power=degree)
    elif kernel == "matern_kernel":
        nu = np.random.choice([0.5, 1.5, 2.5])  # Roughness parameter
        kernel = gpytorch.kernels.MaternKernel(nu=nu)
        kernel.lengthscale = lengthscale
    elif kernel == "rational_quadratic_kernel":
        alpha = np.random.uniform(0.1, 10.0)  # Scale mixture parameter
        kernel = gpytorch.kernels.RQKernel(alpha=alpha)
        kernel.lengthscale = lengthscale
    elif kernel == "spectral_mixture_kernel":
        num_mixtures = np.random.randint(2, 6)  # Number of spectral mixture components
        kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    if scale_kernel:
        kernel = gpytorch.kernels.ScaleKernel(kernel)
    return kernel


def extract_periodicities(kernel, seq_len):
    periodicities = []
    
    # Base case: if the kernel is a PeriodicKernel, extract its period_length
    if isinstance(kernel, PeriodicKernel):
        periodicities.append(kernel.period_length.item()*seq_len)
    
    # If the kernel is a composite kernel (Additive, Product, Scale), recursively extract periodicities
    elif isinstance(kernel, (AdditiveKernel, ProductKernel)):
        for sub_kernel in kernel.kernels:
            periodicities.extend(extract_periodicities(sub_kernel, seq_len))
    
    elif isinstance(kernel, ScaleKernel):
        periodicities.extend(extract_periodicities(kernel.base_kernel, seq_len))
    
    return periodicities