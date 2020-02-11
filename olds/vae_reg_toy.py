"""
Toy example for VAE regression.

"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

X_DIM = 32
Z_DIM = 32
TRUE_Z_DIM = 4
SHIFT_INDEX = 2
SHIFT = 2.5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ToyDataset():
	"""Centered gaussian noise w/ diagonal covariance and a small shift 20% of the time."""

	def __init__(self, task_prob=0.2):
		self.shift_vec = np.zeros(X_DIM)
		self.shift_vec[SHIFT_INDEX] = SHIFT
		self.scale = np.ones(X_DIM)
		self.scale[:TRUE_Z_DIM] = 5.0
		self.task_values = np.arange(2)
		self.task_probs = np.array([1-task_prob, task_prob])

	def sample(self, n=1024, rnd_seed=None):
		if rnd_seed:
			np.random.seed(rnd_seed)
		batch = np.random.normal(scale=self.scale, size=(n,X_DIM))
		task = np.random.choice(self.task_values, p=self.task_probs, size=n)
		np.random.seed(None)
		batch[task == 1] = batch[task == 1] + self.shift_vec
		batch = torch.from_numpy(batch).type(torch.FloatTensor).to(device)
		task = torch.from_numpy(task).type(torch.FloatTensor).to(device)
		return batch, task

class VAE(torch.nn.Module):
	"""Simple VAE."""

	def __init__(self, gamma=1e3):
		super(VAE, self).__init__()
		self.gamma = gamma
		self.l1 = torch.nn.Linear(X_DIM, 64)
		self.l2 = torch.nn.Linear(64, 64)
		self.l31 = torch.nn.Linear(64, Z_DIM)
		self.l32 = torch.nn.Linear(64, Z_DIM)
		self.l4 = torch.nn.Linear(Z_DIM, 64)
		self.l5 = torch.nn.Linear(64, 64)
		self.l6 = torch.nn.Linear(64, X_DIM)
		task_effect = torch.zeros(X_DIM)
		task_effect[SHIFT_INDEX] = SHIFT
		self.task_effect = torch.nn.Parameter(task_effect)

	def save_state(self, fn='state.tar'):
		torch.save(self.state_dict(), fn)

	def load_state(self, fn='state.tar'):
		self.load_state_dict(torch.load(fn))

	def encode(self, x):
		z = F.relu(self.l1(x))
		z = F.relu(self.l2(z))
		mu = self.l31(z)
		logvar = self.l32(z)
		return mu, logvar

	def decode(self, z):
		z = F.relu(self.l4(z))
		z = F.relu(self.l5(z))
		z = self.l6(z)
		return z

	def forward(self, x, task, capacity=2.0):
		"""
		Capacity is in nats.
		"""
		mu, logvar = self.encode(x)

		kld = torch.sum(torch.exp(logvar), dim=1) - torch.sum(logvar, dim=1)
		kld = kld + torch.sum(torch.pow(mu, 2), dim=1) - Z_DIM
		kld = 0.5 * torch.mean(kld)

		elbo = -self.gamma * torch.abs(kld - capacity) # modified elbo from beta-VAE paper

		z_dist = Normal(mu, torch.sqrt(torch.exp(logvar)))
		z_sample = z_dist.rsample()
		x_rec = self.decode(z_sample)
		x_rec = x_rec + torch.ger(task, self.task_effect) # outer product
		x_dist = Normal(x, 0.2)
		rec_error = torch.mean(torch.sum(x_dist.log_prob(x_rec), dim=1))
		elbo = elbo + rec_error
		return -elbo, rec_error


def capacity_schedule(epoch, burnin=5000, max_val=10.0, ramp_epochs=50000):
	return max(0, (epoch - burnin) / ramp_epochs * max_val)


if __name__ == '__main__':
	model = VAE()
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	dset = ToyDataset()

	epochs, elbos, rec_errors, effect_distance, effect_size, capacities = [], [], [], [], [], []

	for epoch in range(55000):
		optimizer.zero_grad()

		samples, task = dset.sample()

		capacity = capacity_schedule(epoch)
		loss, rec_error = model(samples, task, capacity=capacity)
		loss.backward()

		if epoch<200 or (epoch + 1) % 100 == 0:
			if epoch >= 200:
				print(str(epoch).zfill(4), loss.item())
			epochs.append(epoch)
			capacities.append(capacity)
			elbos.append(-loss.item())
			rec_errors.append(rec_error.item())
			task_effect = model.task_effect.detach().cpu().numpy()
			temp = 100/SHIFT * np.sqrt(np.sum(np.power(dset.shift_vec - task_effect, 2)))
			effect_distance.append(temp)
			effect_size.append(100/SHIFT * np.sqrt(np.sum(np.power(task_effect, 2))))

		optimizer.step()


	model.save_state()

	_, axarr = plt.subplots(2,2, figsize=(8,5), sharex=True)
	axarr[0,0].plot(epochs, rec_errors, c='r')
	axarr[0,0].plot(epochs, elbos, c='b')
	axarr[0,0].set_ylabel('modified ELBO')

	axarr[0,1].plot(epochs, capacities, c='b')
	axarr[0,1].set_ylabel('Channel Capacity (nats)')
	axarr[0,1].axhline(y=0.0, ls='--', alpha=0.5)

	axarr[1,0].plot(epochs, effect_distance)
	axarr[1,0].axhline(y=0.0, ls='--', alpha=0.5)
	axarr[1,0].set_ylabel('Effect Error (%)')
	axarr[1,0].set_xlabel('Epoch')

	axarr[1,1].plot(epochs, effect_size)
	axarr[1,1].axhline(y=0.0, ls='--', alpha=0.5)
	axarr[1,1].set_ylabel('Effect Size (%)')
	axarr[1,1].set_xlabel('Epoch')
	plt.tight_layout()
	plt.savefig('temp.pdf')








###
