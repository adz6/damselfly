import h5py
import numpy as np
import torch

def _find_inds_h5(
    data_energy,
    data_pitch,
    data_radius,
    energy_range,
    pitch_range,
    radius_range,
    ):

    print(energy_range, pitch_range, radius_range)
    target_inds = []
    for i, pair in enumerate(zip(data_energy, data_pitch, data_radius)):

        #print(pair)

        if (energy_range[0]<=pair[0]<=energy_range[1])\
        and (pitch_range[0]<=pair[1]<=pitch_range[1])\
        and (radius_range[0]<=pair[2]<=radius_range[1]):
            target_inds.append(i)

    target_inds = np.sort(np.array(target_inds, dtype=np.int32))
    print(target_inds.size)

    return target_inds

def _random_phase(signals):

    rng = np.random.default_rng()

    phase_shifts = (2 * rng.random(signals.shape[0]) - 1) * np.pi 

    return signals * np.exp(1j * phase_shifts)[:, np.newaxis]

def LoadH5ParamRange(
    path=None,
    target_energy_range=None,
    target_pitch_range=None,
    target_radius_range=None,
    val_split=True,
    val_ratio=0.2,
    randomize_phase=False,
    copies=1,
    samples=8192):

    if path is not None:
        file = h5py.File(path, 'r')
    else:
        raise ValueError('Path cannot be None')
    rng = np.random.default_rng()
    try:

        signal_energy = file['meta']['energy'][:]
        signal_pitch = file['meta']['theta_min'][:]
        signal_radius = file['meta']['x_min'][:]

        all_inds = np.arange(0, signal_energy.size, 1)

        if (target_energy_range is not None)\
         and (target_pitch_range is not None)\
         and (target_radius_range is not None):

            target_data_inds = _find_inds_h5(
                signal_energy,
                signal_pitch,
                signal_radius,
                target_energy_range,
                target_pitch_range,
                target_radius_range,
                )
        else:
            raise ValueError('Target parameter ranges cannot be None.')

        if val_split:

            # randomly sample/split the train signals
            random_choice_target_inds = rng.choice(
                target_data_inds,
                size=target_data_inds.size,
                replace=False,
                )

            n_train = int(random_choice_target_inds.size * (1 - val_ratio))

            train_inds = np.sort(random_choice_target_inds[0:n_train])
            val_inds = np.sort(random_choice_target_inds[n_train:])
            n_val = val_inds.size
            n_train = train_inds.size

        else:
            n_val=0
            val_inds = np.array([], dtype=np.int32)
            train_inds = target_data_inds
            n_train = train_inds.size
            
            
        print(f'The number of unique training signals is {n_train}')
        print(f'The number of unique validation signals is {n_val}')


        train_data = []
        val_data = []

        for i in range(copies):
            temp_train = file['x'][train_inds, 0:samples]
            temp_val = file['x'][val_inds, 0:samples]
            if randomize_phase:
                temp_train = _random_phase(temp_train)
                temp_val = _random_phase(temp_val)

            train_data.append(temp_train)
            val_data.append(temp_val)

        train_data = np.concatenate((*train_data,),)
        val_data = np.concatenate((*val_data,),)

        n_train = train_data.shape[0]
        print(f'The number of training signals is {n_train}')
        n_val = val_data.shape[0]
        print(f'The number of validation signals is {n_val}')

    finally:
        file.close()

    print(f'The train data shape is {train_data.shape}')
    print(f'The val data shape is {val_data.shape}')

    return torch.tensor(train_data, dtype=torch.cfloat), torch.tensor(val_data, dtype=torch.cfloat), train_inds, val_inds