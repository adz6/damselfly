import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import torch
import damselfly as df
import scipy.integrate as integrate
import seaborn as sns

def Curriculum(info, save_path, name, epochs_per_xtick=1):

	loss_curves = []
	acc_curves = []
	val = []
	for temp in info['temps']:
		for i, ep in enumerate(info[temp]['loss']):
			loss_curves.extend(info[temp]['loss'][ep])
			acc_curves.extend(info[temp]['acc'][ep])
			#if i % 10 == 9:
			#print(len(info[temp]['val'][ep]), len(info[temp]['acc'][ep]))
			val.append(torch.mean(info[temp]['val'][ep]))


	#print(len(val), len(loss_curves))
	ticks_per_temp = len(loss_curves) / len(info['temps'])
	epochs_per_temp = len(list(info[temp]['loss'].keys()))

	ticks_per_epoch = ticks_per_temp / epochs_per_temp

	kernel_size = 32
	smooth_kernel = np.ones(kernel_size) / kernel_size

	smooth_acc = np.convolve(acc_curves, smooth_kernel)
	smooth_acc = smooth_acc[0:smooth_acc.size - kernel_size + 1]

	xticklabel = info['temps']
	fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7,5))

	ax1.plot(loss_curves, color='tab:blue', label='Train Loss')
	ax1.set_xticks( np.arange(1 , len(xticklabel) + 1, 1) * ticks_per_temp)
	ax1.set_xticklabels(xticklabel)
	ax1.set_xlim(0, len(loss_curves))
	ax1.set_xlabel('Temperature (K)')
	ax1.set_ylabel('Loss')
	ax1.set_title('Curriculum Training Loss and Accuracy vs Data Noise Temp')
	#print(np.arange(1, len(val) + 1, 1) * ticks_per_epoch )
	

	ax2 = ax1.twinx()
	ax2.plot(smooth_acc, color='tab:orange', label='Train Accuracy')
	ax2.set_ylabel('Accuracy')

	ax2.plot(np.arange(1 , len(val) + 1, 1) * ticks_per_epoch , val, color='tab:green', label='Val. Accuracy')

	ax1.set_ylim(0.0, 0.8)
	ax2.set_ylim(0.0, 1.1)

	fig.legend(loc=(0.65,0.4))

	plt.savefig(os.path.join(save_path, name))

def CompareTrainLoss(info_list, temps, save_path, name, epochs_per_xtick=1):

	loss_curves = []
	for info in info_list:
		epochs = list(info['loss'].keys())
		n_epochs = len(epochs)

		loss_all_epochs = []
		for ep in epochs:
			loss_all_epochs.extend(info['loss'][ep])

		n_iter = len(loss_all_epochs)
		n_iter_per_epoch = n_iter // n_epochs
		loss_curves.append(loss_all_epochs)

	xticklabel = []
	for n in range(n_epochs // epochs_per_xtick):
		xticklabel.append(str((n + 1) * epochs_per_xtick))

	fig, ax = plt.subplots(figsize=(7,5))
	for i,loss_curve in enumerate(loss_curves):
		ax.plot(loss_curve, label = str(temps[i]) + ' K')
		
	ax.set_ylabel('Training Loss')
	ax.set_xlabel('Epoch')
	ax.set_xticks( np.arange(1 * epochs_per_xtick, n_epochs + 1, 1 * epochs_per_xtick) * n_iter_per_epoch)
	ax.set_xlim(0, n_iter)
	ax.set_ylim(0, 0.8)
	ax.set_xticklabels(xticklabel)
	ax.set_title('Training Loss Curves for Different Noise Temperatures')

	fig.legend(loc='upper right')

	plt.savefig(os.path.join(save_path, name))

def CompareValAccuracy(info_list, temps, save_path, name, epochs_per_xtick=1):

	accuracy_curves = []
	for info in info_list:
		epochs = list(info['loss'].keys())
		n_epochs = len(epochs)

		loss_all_epochs = []
		val = []
		for ep in epochs:
			loss_all_epochs.extend(info['loss'][ep])
			val.append(torch.mean(torch.as_tensor(info['val'][ep])))
		
		n_iter = len(loss_all_epochs)
		n_iter_per_epoch = n_iter // n_epochs

		accuracy_curves.append(val)

	xticklabel = []
	for n in range(n_epochs // epochs_per_xtick):
		xticklabel.append(str((n + 1) * epochs_per_xtick))

	fig, ax = plt.subplots(figsize=(7,5))
	for i,val in enumerate(accuracy_curves):
		ax.plot((np.arange(len(val))+1) * n_iter_per_epoch, val, label = str(temps[i]) + ' K')
		
	ax.set_ylabel('Validation Accuracy')
	ax.set_xlabel('Epoch')
	ax.set_xticks( np.arange(1 * epochs_per_xtick, n_epochs + 1, 1 * epochs_per_xtick) * n_iter_per_epoch)
	ax.set_xlim(0, n_iter)
	ax.set_ylim(0.4, 1.1)
	ax.set_xticklabels(xticklabel)
	ax.set_title('Validation Accuracy for Different Noise Temperatures')

	fig.legend(loc='upper right')

	plt.savefig(os.path.join(save_path, name))

def CompareTrainAccuracy(info_list, temps, save_path, name, epochs_per_xtick=1):

	accuracy_curves = []
	for info in info_list:
		epochs = list(info['loss'].keys())
		n_epochs = len(epochs)

		loss_all_epochs = []
		acc_all_epochs = []
		val = []
		for ep in epochs:
			loss_all_epochs.extend(info['loss'][ep])
			acc_all_epochs.extend(info['acc'][ep])
			val.append(torch.mean(torch.as_tensor(info['val'][ep])))
		
		n_iter = len(loss_all_epochs)
		n_iter_per_epoch = n_iter // n_epochs

		kernel_size = 12
		smooth_kernel = np.ones(kernel_size) / kernel_size

		smooth_acc = np.convolve(torch.Tensor.cpu(torch.as_tensor(acc_all_epochs)), smooth_kernel)
		smooth_acc = smooth_acc[0:smooth_acc.size - kernel_size + 1]

		accuracy_curves.append(smooth_acc)

	xticklabel = []
	for n in range(n_epochs // epochs_per_xtick):
		xticklabel.append(str((n + 1) * epochs_per_xtick))

	fig, ax = plt.subplots(figsize=(7,5))
	for i,acc in enumerate(accuracy_curves):
		ax.plot(acc, label = str(temps[i]) + ' K')
		
	ax.set_ylabel('Training Accuracy')
	ax.set_xlabel('Epoch')
	ax.set_xticks( np.arange(1 * epochs_per_xtick, n_epochs + 1, 1 * epochs_per_xtick) * n_iter_per_epoch)
	ax.set_xlim(0, n_iter)
	ax.set_ylim(0.4, 1.1)
	ax.set_xticklabels(xticklabel)
	ax.set_title('Training Accuracy for Different Noise Temperatures')

	fig.legend(loc='upper right')

	plt.savefig(os.path.join(save_path, name))

def TrainingInfo(info, save_path, name, epochs_per_xtick=1):

    epochs = list(info['loss'].keys())
    n_epochs = len(epochs)

    loss_all_epochs = []
    acc_all_epochs = []
    val = []
    for ep in epochs:
        loss_all_epochs.extend(info['loss'][ep])
        acc_all_epochs.extend(info['acc'][ep])
        val.append(torch.mean(torch.as_tensor(info['val'][ep])))
    #print(len(val), len(loss_all_epochs))
    n_iter = len(loss_all_epochs)
    n_iter_per_epoch = n_iter // n_epochs

    #kernel_size = 12
    #smooth_kernel = np.ones(kernel_size) / kernel_size

    #smooth_acc = np.convolve(torch.Tensor.cpu(torch.as_tensor(acc_all_epochs)), smooth_kernel)
    #smooth_acc = smooth_acc[0:smooth_acc.size - kernel_size + 1]

    xticklabel = []
    for n in range(n_epochs // epochs_per_xtick):
        xticklabel.append(str((n + 1) * epochs_per_xtick))
    colors = ['tab:blue', 'tab:orange']

    fig1, ax1 = plt.subplots(figsize=(7,5))
    ax1.plot(loss_all_epochs, label = 'Train Loss', color = colors[0])
    ax1.set_ylabel('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_xticks( np.arange(1 * epochs_per_xtick, n_epochs + 1, 1 * epochs_per_xtick) * n_iter_per_epoch)
    ax1.set_xlim(0, n_iter)
    ax1.set_ylim(0, 0.8)
    ax1.set_xticklabels(xticklabel)


    ax2 = ax1.twinx()
    ax2.plot(torch.Tensor.cpu(torch.as_tensor(acc_all_epochs)), label = 'Train Acc.', color = colors[1])
    ax2.plot((np.arange(len(val))+1) * n_iter_per_epoch, val, label = 'Val. Acc.', color = 'tab:green')
    ax2.set_ylim(0, 1.1)

    ax2.set_ylabel('Training Accuracy')
    
    fig1.legend(loc=(.8, 0.85))
    #plt.tight_layout()
    plt.savefig(os.path.join(save_path, name))

def ConfusionMatrix(matrix, font_size=24):

    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(1,1,1)
    
    matrix = np.flip(np.flip(matrix, axis=0), axis=1)
    
    N_matrix = matrix.shape[0]
    
    plot_matrix = np.zeros((N_matrix, N_matrix))
    
    imatrix = np.arange(0, N_matrix + 1, 1)
    
    #plot_matrix[0:N_matrix + 1, 0:N_matrix +1] = matrix
    
    
    
    for irow in imatrix:
        for icol in imatrix:
            if irow < N_matrix and icol <  N_matrix and irow == icol:
                plot_matrix[irow, icol] = np.mean(np.diagonal(matrix))
            elif irow < N_matrix and icol <  N_matrix:
                plot_matrix[irow, icol] = matrix[irow, icol]
    
    
    
    #, axs = plt.subplots(nrows=1, ncols=n_mat, figsize = (4 * n_mat, 4), sharey=True, sharex=True)
    
    cmap = sns.color_palette("light:skyblue", as_cmap=True)
    img = ax.imshow(plot_matrix, cmap = cmap)

    for irow in imatrix:
        for icol in imatrix:
            if irow < N_matrix and icol < N_matrix:
                if irow == icol:
                    color = 'k'
                else:
                    color = 'k'
                ax.text(icol, irow-0.15, int(matrix[irow, icol]), 
                        ha='center', va='center', fontsize=font_size, color=color)
                ax.text(icol, irow+0.15, np.round(matrix[irow, icol] / matrix.sum(), 3), 
                        ha='center', va='center', fontsize=font_size, color=color)
            if irow < N_matrix and icol == N_matrix:
                ax.text(icol, irow-0.15, np.round(matrix[irow, irow] / matrix[irow, :].sum(), 3), 
                        ha='center', va='center', fontsize=font_size, color='tab:green')
                ax.text(icol, irow+0.15, np.round((matrix[irow, :].sum() - matrix[irow, irow])/ matrix[irow, :].sum(), 3), 
                        ha='center', va='center', fontsize=font_size, color='tab:red')
            if icol < N_matrix and irow == N_matrix:
                ax.text(icol, irow-0.15, np.round(matrix[icol, icol] / matrix[:, icol].sum(), 3), 
                        ha='center', va='center', fontsize=font_size, color='tab:green')
                ax.text(icol, irow+0.15, np.round((matrix[:, icol].sum() - matrix[icol, icol])/ matrix[:, icol].sum(), 3), 
                        ha='center', va='center', fontsize=font_size, color='tab:red')
            if icol == N_matrix and irow == N_matrix:
                ax.text(icol, irow-0.15, np.round(np.trace(matrix) / matrix.sum(), 3), 
                        ha='center', va='center', fontsize=font_size, color='tab:green')
                ax.text(icol, irow+0.15, np.round(abs(np.trace(matrix) - matrix.sum()) / matrix.sum(), 3), 
                        ha='center', va='center', fontsize=font_size, color='tab:red')

    class_labels = []
    if N_matrix > 2:
        for i in np.arange(1, N_matrix, 1):
            class_labels.append(f'S{i}')
        class_labels.append('N')
    else:
        class_labels = ['S', 'N']
    ax.tick_params(length=12, width=2)
    ax.tick_params(axis='y', right=True, left=False, labelright=True, labelleft=False)
    ax.set_yticks(np.arange(0, N_matrix, 1))
    ax.set_yticklabels(class_labels, size=24)
    ax.set_ylabel('True Class', size=24, y=np.ceil(N_matrix / 2) / N_matrix)

    ax.set_xticks(np.arange(0, N_matrix, 1))
    ax.set_xticklabels(class_labels, size=24)
    ax.set_xlabel('Predicted Class', size=24, x=1 - np.ceil(N_matrix / 2) / N_matrix)
    ax.yaxis.set_label_position('right')
    
    
    ax.set_xticks(np.arange(-0.5, N_matrix + 1 + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_matrix + 1 + 0.5, 1), minor=True)
    

    plt.grid(which='minor', color='k', linewidth=2)
    return [fig, ax]


def ClassMatrixObservable(train_mats, test_mats, temps, coords, save_path, name):

	n_mat = len(train_mats)

	train = []
	test = []

	for n in range(n_mat):

		train.append(train_mats[n][coords[0], coords[1]] / train_mats[n][coords[0], :].sum())
		test.append(test_mats[n][coords[0], coords[1]] / test_mats[n][coords[0], :].sum())

	fig, axs = plt.subplots(nrows=1, ncols=1, figsize = (7, 5))

	axs.plot(temps, train, 'o', label='train')
	axs.plot(temps, test, 'o', label='test')
	axs.set_xlabel('Noise Temp. K')
	
	if coords == [0, 0]:
		axs.set_ylabel('Detection Efficiency')
	elif coords == [1, 0]:
		axs.set_ylabel('False Alarm Rate')

	fig.legend()
	plt.savefig(os.path.join(save_path, name))

	
def NoiseVsSignals(temps, save_path, name):

	fig, axs = plt.subplots(nrows=len(temps), ncols=1, figsize = (7, 2 * len(temps)), sharex=True)
	time = np.arange(0, 8192, 1) * 5e-9
	for n, temp in enumerate(temps):
		
		signal, noise = df.utils.LoadTemperatureDataset(temp)
		#print(noise.keys())
		signal_ind = np.random.randint(0, 601)
		axs[n].plot(time, noise['train'][signal_ind], label='noise ('+str(temp)+' K)')
		axs[n].plot(time, signal['train_signals'][signal_ind], label='signal')
		axs[n].set_ylim(-1.e-6, 1.e-6)
		axs[n].set_ylabel('Amplitude (V)' )
		axs[n].legend(loc='upper right')
		#axs[n].legend(loc='upper right')
		#print(signal.keys())
	axs[0].set_xlim(time[0], time[256])
	axs[len(temps) - 1].set_xlabel('Time (s)')
	#axs[0].legend(loc=(0.5, 1))
	plt.savefig(os.path.join(save_path, name))

def NoiseVsSignalPlusNoise(temps, save_path, name):

	fig, axs = plt.subplots(nrows=2, ncols=len(temps), figsize = (4 * len(temps), 6 ), sharex=True)
	time = np.arange(0, 8192, 1) * 5e-9
	for n, temp in enumerate(temps):
		signal, noise = df.utils.LoadTemperatureDataset(temp)
		signal_ind = np.random.randint(0, 60100)
		pure_signal_ind = int(np.trunc(signal_ind / 100))
		axs[0][n].plot(time, signal['train'][signal_ind], label = 'signal+noise')
		axs[0][n].plot(time, signal['train_signals'][pure_signal_ind], label = 'signal')
		axs[1][n].plot(time, noise['train'][signal_ind], label = 'noise')
		axs[0][n].set_xlim(time[0], time[128])
		
		axs[0][n].set_ylim(-8 * np.sqrt(temp) * 1e-7, 8 * np.sqrt(temp) * 1e-7)
		axs[1][n].set_ylim(-8 * np.sqrt(temp) * 1e-7, 8 * np.sqrt(temp) * 1e-7)
		axs[1][n].set_xlabel('Time (s)')

		axs[0][n].legend(loc='upper right')
		axs[1][n].legend(loc='upper right')
		axs[0][n].set_title('Noise Temp. = ' + str(temp) + ' K')

	axs[0][0].set_ylabel('Amplitude (V)' )
	axs[1][0].set_ylabel('Amplitude (V)' )	
	plt.savefig(os.path.join(save_path, name))

def PitchAngles(angles, save_path, name):

	fig, axs = plt.subplots(nrows=1, ncols=1, figsize = (7,5))
	time = np.arange(0, 8192, 1) * 5e-9

	signal, noise = df.utils.LoadTemperatureDataset(0.1)

	for n, angle in enumerate(angles):

		#print(np.argwhere(signal['train_pa'] == angle))
		axs.plot(time, signal['train_signals'][np.argwhere(signal['train_pa'] == angle)[0,0]], label=str(angle))

	axs.set_xlim(time[0], time[64])

	axs.set_xlabel('Time (s)')
	axs.set_ylabel('Amplitude (V)')
	axs.legend(loc='upper right')
	plt.savefig(os.path.join(save_path, name))


def SNRvsPitchAngle(save_path, name):

	temp = 10.0
	var = 4 * 1.38e-23 * 100e6 * 10 * 50 / 2
	signal, noise = df.utils.LoadTemperatureDataset(temp)
	#print(signal.keys())
	#print(np.asarray(signal['train_pa']).shape, np.asarray(signal['train']).shape, np.asarray(signal['train_signals']).shape)

	# compute normalized templates numerator -> signal power
	pitch_angles = np.asarray(signal['train_pa'])
	pure_signals = np.asarray(signal['train_signals'])

	signal_energy = np.sum((pure_signals**2), axis=1)

	norm = (1 / np.sqrt(signal_energy * (var / 4))).reshape((signal_energy.shape[0], 1)).repeat(pure_signals.shape[1], axis=1)

	templates = norm * pure_signals

	# compute MF Test Statistic -> signal noise expectation
	# compute MF SNR -> template * signal / template * noise

	T = []
	snr = []

	numerator = np.sum(pure_signals ** 2, axis=1)
	denominator = var 

	snr = numerator / denominator
	T = np.sum(templates * pure_signals, axis = 1)
	#for n in range(templates.shape[0]):

	#	if n % 5 == 4:
	#		print(n + 1)
	#	inds = np.arange(n * 100, (n + 1) * 100, 1, dtype=np.int32)

	#	numerator = abs(np.sum(templates[n, :] * pure_signals[n, :]))**2
		#print(numerator)

	#	template_mat = templates[n, :].reshape((1, len(templates[n, :]))).repeat(100, axis=0)

		#denominator = np.mean(abs(np.sum(template_mat * np.asarray(noise['train'])[inds, :], axis=1))**2)

	#	denominator = var * np.sum(templates[n, :]**2)
		#print(denominator)

	#	T_avg = np.mean(abs(np.sum(template_mat * np.asarray(signal['train'])[inds, :], axis=1)))
		#print(np.mean(abs(np.sum(signal_mat * np.asarray(noise['train'])[noise_inds, :], axis=1))))
	#	T.append(T_avg)	
	#	snr.append(numerator / denominator)	

	#snr = signal_power / np.asarray(signal_noise_expectation)
	#kernel_size = 32
	#smooth_kernel = np.ones(kernel_size) / kernel_size

	#smooth_snr = np.convolve(snr, smooth_kernel)

	fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
	ax1.plot(pitch_angles, T)
	ax1.set_xlabel('Pitch Angle (deg)')
	ax1.set_ylabel('MF Test Statistic for Noise Temperature 10K')
	ax1.set_title('Expected Value for Matched Filter Test Statistic\n using an Ideal Templae')
	#ax1.set_ylim(-5, 5)

	plt.savefig(os.path.join(save_path, 'mf_test_stat_baseline.png'))

	fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
	ax2.plot(pitch_angles, snr)
	ax2.set_xlabel('Pitch Angle (deg)')
	ax2.set_ylabel('SNR for Noise Temperature 10K')
	#ax1.set_ylim(-5, 5)

	plt.savefig(os.path.join(save_path, '2.png'))


def ROC(roc, mean_only = False):

    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    
    #print(roc.keys())
    
    class_ind = np.array(roc['class_ind'])
    
    tpr_array = np.array(roc['tpr_list'])
    fpr_array = np.array(roc['fpr_list'])
    tpr_mean = np.mean(tpr_array, axis = 0)
    fpr_mean = np.mean(fpr_array, axis = 0)
    
    if not mean_only:
        for i in class_ind:
            auc = np.round(-1 * integrate.trapezoid(tpr_array[i, :], x=fpr_array[i, :]), 4)
        
            if i == 0:
                ax.plot(fpr_array[i, :], tpr_array[i, :], label = f'N, auc = {auc}')
            else:
                ax.plot(fpr_array[i, :], tpr_array[i, :], label = f'S{i}, auc = {auc}')
    baseline = np.linspace(0, 1, 100)
    ax.plot(baseline, baseline, '--', color='gray')
    mean_auc = np.round(-1 * integrate.trapezoid(tpr_mean, x=fpr_mean), 4)
    if mean_only:
        ax.plot(fpr_mean, tpr_mean, color='tab:blue', linewidth=2, label = f'auc = {mean_auc}')
    else:
        ax.plot(fpr_mean, tpr_mean, color='k', linewidth=2, label = f'mean, auc = {mean_auc}')
    #for i in range(len(tpr_list)):
    #	ax1.plot(fpr_list[i], tpr_list[i], label = str(temps[i]) + ' K')

    #ax1.legend(loc='right')
    #ax1.set_title('ROC Curves for Different Noise Temperatures')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.grid()

    #ax1.set_xlim(-0.02,1.02)
    #ax1.set_ylim(-0.02, 1.02)
    #plt.savefig(os.path.join(save_path, name))
    return [fig, ax]

def BaselineMFScore(temps, save_path, name, data_path='/home/az396/project/deepfiltering/analysis/baseline_MF_distributions.pkl'):

	with open(data_path, 'rb') as infile:
		data = pkl.load(infile)

	fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7,5))

	for temp in temps:
		axs.hist(data[temp]['signal'], 40, histtype='step', label='Signal (' + str(temp) + ' K)')
	axs.hist(data[temps[2]]['noise'], 40, histtype='step', label='Noise')

	axs.set_xlabel('Matched Filter Score')
	axs.set_ylabel('Counts')
	axs.set_title('Distribution of Matched Filter Scores\nfor Noisy Signal Data and Noise Only Data')

	axs.legend(loc='upper right')

	plt.savefig(os.path.join(save_path, name))

def BaselineROC(temps, save_path, name, zoom=False, data_path='/home/az396/project/deepfiltering/analysis/baseline_MF_distributions.pkl'):

	with open(data_path, 'rb') as infile:
		data = pkl.load(infile)

	thresholds = np.arange(-10, 100, 1)

	tpr_curves = []
	fpr_curves = []
	for temp in temps:
		tpr = []
		fpr = []
		for y in thresholds:
			#print(y)
			tpr.append(len(np.where(data[temp]['signal'] >= y)[0]) / len(data[temp]['signal']))
			fpr.append(len(np.where(data[temp]['noise'] >= y)[0]) / len(data[temp]['noise']))
		#print(tpr, fpr)
		tpr_curves.append(tpr)
		fpr_curves.append(fpr)

	example_threshold = 6.0

	tpr_example = len(np.where(np.asarray(data[temps[-1]]['signal']) >= example_threshold)[0]) / len(data[temps[-1]]['signal'])
	fpr_example = len(np.where(np.asarray(data[temps[-1]]['noise']) >= example_threshold)[0]) / len(data[temps[-1]]['noise'])

	fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7,5))

	for i,temp in enumerate(temps):
		axs.plot(fpr_curves[i], tpr_curves[i], label= str(temp) + ' K')

	if zoom:
		axs.vlines(fpr_example, 0.7, 1.02, color='r')
		axs.set_xlim(-0.0005,0.01)

	axs.set_xlabel('False Positive Rate')
	axs.set_ylabel('True Positive Rate')
	axs.set_title('ROC Curves for a Matched Filter Detector at Select Noise Powers')
	axs.legend(loc=1)
	plt.savefig(os.path.join(save_path, name))

