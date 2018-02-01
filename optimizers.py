import numpy as np
from scipy.stats import multivariate_normal


class GradientDecent(object):
    def __init__(self, oracle, lr=0.1):
        """
        :param oracle: Oracle function that compute log(value) and a gradient(log(value)) in a given point
        """
        self.oracle = oracle
        self.number_of_steps = 0
        self.lr = lr

    def initialize(self, intialization):
        self.current = intialization

    def update(self):
        value, grad = self.oracle(self.current)
        self.current -= self.lr * grad
        self.number_of_steps += 1
        return value


class MetropolisHastingsMCMC(object):
    def __init__(self, density):
        self.density = density
        self.number_of_steps = 0

    def initialize(self, initialization):
        self.current = initialization

    def transition(self):
        return multivariate_normal(mean = self.current).rvs(size = (1, ))

    def compute_alpha(self, new):
        symmetric_part = self.density(new) / self.density(self.current)
        return min(1, symmetric_part)

    def get_next(self):
        accepted = False


        while not accepted:
            new = self.transition()
            alpha = self.compute_alpha(new)
            th = np.random.uniform(0, 1)
            if alpha > th:
                accepted = True
                self.current = new
            self.number_of_steps += 1

        return self.current

class LangevinMCMC(MetropolisHastingsMCMC):
    def __init__(self, density, log_grad_density, tao):
        super(LangevinMCMC, self).__init__(density)
        self.log_grad_density = log_grad_density
        self.tao = tao

    def transition(self):
        noise = np.random.multivariate_normal(mean=np.zeros_like(self.current), cov=np.identity(len(self.current)))
        return self.current + self.tao * self.log_grad_density(self.current) + np.sqrt(2 * self.tao) * noise


    def proposal_a_given_b(self, a, b):
        val = a - b - self.tao * self.log_grad_density(b)
        return np.exp(-np.sum(val ** 2)/(4 * self.tao))

    def compute_alpha(self, new):
        symmetric_part = self.density(new) / self.density(self.current)
        assymetric_part = self.proposal_a_given_b(self.current, new) / self.proposal_a_given_b(new, self.current)
        return min(1, symmetric_part * assymetric_part)


if __name__ == "__main__":
    density = lambda x: multivariate_normal(mean=[5, 5], cov=[[1, 0], [0, 1]]).pdf(x)
    log_grad_density = lambda x: -np.sum((x - 5))
    mcmc =  LangevinMCMC(density, log_grad_density, 0.001) #MetropolisHastingsMCMC(density)

    mcmc.initialize(np.array([0.5, 0.5]))
    values = []

    for i in range(10000):
        next = mcmc.get_next()
        if i > 1000 and i % 10 == 0:
            values.append(next)

    print (mcmc.number_of_steps)
    import pylab as plt
    plt.subplot(1, 2, 1)
    values = np.array(values)
    print (values.mean(axis=0), values.var(axis=0))
    plt.scatter(values[:, 0], values[:, 1])

    plt.subplot(1, 2, 2)
    vals = multivariate_normal(mean=[5, 5], cov=[[1, 0], [0, 1]]).rvs(size=900)
    print (vals.shape)
    plt.scatter(vals[:, 0], vals[:, 1], color='red')
    plt.show()





