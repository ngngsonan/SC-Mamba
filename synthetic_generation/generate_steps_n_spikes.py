import numpy as np
import torch

def generate_damping(input_size: int, p=[0.4, 0.5, 0.1]) -> torch.Tensor:

    spacing = np.random.choice(["equal", "regular", "random"], p=p)
    t = np.arange(0, input_size, 1).astype(float)
    
    if spacing == "random":
        num_steps = np.random.randint(1,3)  # Number of damping steps
        damping_intervals = np.sort(np.random.choice(t[:-int(input_size*0.1)], num_steps, replace=False))
        damping_factors = np.random.uniform(0.1, 2, num_steps+1)
    elif spacing == "equal":
        # Generate random damping factors and intervals
        num_steps = np.random.randint(3, 7)  # Number of damping steps
        damping_intervals = np.linspace(0, input_size, num_steps+2)[1:-1]  # Random time intervals
        damping_factors = np.array([np.random.uniform(0.4,0.8) if (i%2) == 0 else np.random.uniform(1,2) for i in range(num_steps+1)])
    else:
        custom_lengths = np.random.randint(1, input_size//2, 2)
        damping_intervals = []
        current_time = 0
        while current_time < input_size:
            for length in custom_lengths:
                current_time += length
                if current_time <= input_size:
                    damping_intervals.append(current_time)
                else:
                    break
        damping_intervals = np.array(damping_intervals)
        num_steps = len(damping_intervals)
        damping_factors = np.array([np.random.uniform(0.4,0.8) if (i%2) == 0 else np.random.uniform(1,2) for i in range(num_steps+1)])
    # Create a piecewise damping function based on the random intervals and factors
    damping = np.piecewise(t, 
                            [t < damping_intervals[0]] + 
                            [(t >= damping_intervals[i]) & (t < damping_intervals[i+1]) for i in range(num_steps-1)] + 
                            [t >= damping_intervals[-1]],
                            damping_factors.tolist())
    
    return torch.Tensor(damping)



def generate_spikes(size, spikes_type="choose_randomly", spike_intervals=None, n_spikes=None, to_keep_rate=0.4):
    spikes = np.zeros(size)
    if size < 120:
        build_up_points = 1
    elif size < 250:
        build_up_points = np.random.choice([2, 1], p=[0.3, 0.7])
    else:
        build_up_points = np.random.choice([3, 2, 1], p=[0.15, 0.45, 0.4])
        
    spike_duration = build_up_points*2
    
    if spikes_type == "choose_randomly":
        spikes_type = np.random.choice(["regular", "patchy", "random"], p=[0.4, 0.5, 0.1])
    
    if spikes_type == "patchy" and size < 64:
        spikes_type = "regular"
    
    # print(spikes_type)
    if spikes_type in ["regular", "patchy"]:
        if spike_intervals is None:
            upper_bound = np.ceil(spike_duration / 0.05)  ## at least 1 spike every 24 periods (120 if 5 spike duration) #np.ceil(spike_duration * size/(size*0.05))
            lower_bound = np.ceil(spike_duration / 0.15) ## at most 3 spikes every 24 periods
            spike_intervals = np.random.randint(lower_bound, upper_bound)
        n_spikes = np.ceil(size/spike_intervals)
        spike_intervals = np.arange(spike_intervals, size, spike_intervals)
        if spikes_type == "patchy":
            patch_size = np.random.randint(2, max(n_spikes*0.7,3))
            to_keep = np.random.randint(np.ceil(patch_size*to_keep_rate), patch_size)
    else:
        n_spikes = n_spikes if n_spikes is not None else np.random.randint(4, min(max(size//(spike_duration*3),6), 20))
        spike_intervals = np.sort(np.random.choice(np.arange(spike_duration, size), size=n_spikes, replace=False))
        
    constant_build_rate = False
    if spikes_type in ["regular", "patchy"]:
        random_ = np.random.random()
        constant_build_rate = True

    patch_count = 0
    spike_intervals -= 1
    for interval in spike_intervals:    
        interval = np.round(interval).astype(int)
        if spikes_type == "patchy":
            if patch_count >= patch_size:
                patch_count = 0
            if patch_count < to_keep:
                patch_count += 1
            else:                
                patch_count += 1
                continue         
        if not constant_build_rate:
            random_ = np.random.random()
        build_up_rate = np.random.uniform(0.5, 2) if  random_ < 0.7 else np.random.uniform(2.5, 5) 

        # if patchy, i want to 
        # Build-up phase
        spike_start = interval - build_up_points + 1
        for i in range(build_up_points):
            if 0 <= spike_start + i < len(spikes):
                spikes[spike_start + i] = build_up_rate * (i + 1)
 
        for i in range(1, build_up_points):
            if (interval + i) < len(spikes):
                spikes[interval + i] = spikes[interval - i] 
                
    # randomly make it positive or negative
    spikes += 1
    spikes = spikes * np.random.choice([1, -1], 1, p=[0.7, 0.3])

    return torch.Tensor(spikes) 


def generate_peak_spikes(ts_size, peak_period, spikes_type="regular"):
    return generate_spikes(ts_size, spikes_type=spikes_type, spike_intervals=peak_period)