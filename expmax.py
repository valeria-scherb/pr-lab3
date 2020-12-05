import random


class ExpMax:
    def __init__(self, im_dim, clusters=2):
        self.D = im_dim
        self.K = clusters
        self.pi = [1.0/self.K] * self.K
        self.mu = []
        for k in range(0, self.K):
            m, s = [], 0.0
            for i in range(0, self.D):
                m.append(random.random() * 0.5 + 0.25)
                s += m[-1]
            self.mu.append([x / s for x in m])
        self.x, self.z, self.N = [], [], 0

    def load_train(self, data):
        for d in data:
            self.x.append(d)
            self.z.append([0.0] * self.K)
        self.N = len(self.x)

    def run(self, repeat=1):
        for i in range(0, repeat):
            self.expectation()
            self.maximization()

    def expectation(self):
        for n in range(0, self.N):
            s = 0
            for k in range(0, self.K):
                self.z[n][k] = self.expectation_sub(
                    self.pi[k], self.x[n], self.mu[k])
                s += self.z[n][k]
            for k in range(0, self.K):
                if s > 0.0:
                    self.z[n][k] /= s
                else:
                    self.z[n][k] = 1.0 / self.K

    def expectation_sub(self, pi, x, mu):
        z_nk = pi
        for i in range(0, self.D):
            z_nk *= pow(mu[i], x[i]) * pow(1 - mu[i], 1 - x[i])
        return z_nk

    def maximization(self):
        self.pi = [self.n_m(k) / self.N for k in range(0, self.K)]
        for k in range(0, self.K):
            avg_x_k = self.avg_x(k)
            for i in range(0, self.D):
                self.mu[k][i] = avg_x_k[i]

    def avg_x(self, m):
        res = [0.0] * self.D
        for i in range(0, self.D):
            for n in range(0, self.N):
                res[i] += self.z[n][m] * self.x[n][i]
        n_m = self.n_m(m)
        return [r/n_m for r in res]

    def n_m(self, m):
        r = 0.0
        for n in range(0, self.N):
            r += self.z[n][m]
        return r
