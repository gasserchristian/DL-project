from torch.optim import Optimizer
import copy
import numpy as np



class GPOMDP(Optimizer):
    r"""Optimization class for calculating the gradient of one iteration.
            Args:
                params (iterable): iterable of parameters to optimize or dicts defining
                    parameter groups
                lr (float): learning rate
            """
    def __init__(self, model, agent, optimizer, useOptBaseline):
        self.useOptBaseline = useOptBaseline
        self.rewardToCurrentStep = 0
        self.gradientEstimatorCurrentTrial = self.optimizer.grad.clone()
        self.gradientToCurrentStep = self.optimizer.grad.clone()

    def startTrial(self):

        self.rewardToCurrentStep = 0
        self.gradientEstimatorCurrentTrial = self.optimizer.grad.clone()
        self.gradientToCurrentStep = self.optimizer.grad.clone()

    def endTrial(self):
        self.optimizer.grads = 0



class SVRG_optim(Optimizer):
    r"""Optimization class for calculating the gradient of one iteration.
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float): learning rate
        """

    def __init__(self, params, lr, weight_decay=0):
        print("Optimizer: SVRG")
        self.u = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SVRG_optim, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups

    def set_u(self, new_u):
        """Set the mean gradient for the current epoch.
        """
        if self.u is None:
            self.u = copy.deepcopy(new_u)
        for u_group, new_group in zip(self.u, new_u):
            for u, new_u in zip(u_group['params'], new_group['params']):
                u.grad = new_u.grad.clone()

    def step(self, params):
        """Performs a single optimization step.
        """
        for group, new_group, u_group in zip(self.param_groups, params, self.u):
            weight_decay = group['weight_decay']

            for p, q, u in zip(group['params'], new_group['params'], u_group['params']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue
                # core SVRG gradient update
                new_d = p.grad.data - q.grad.data + u.grad.data
                if weight_decay != 0:
                    new_d.add_(weight_decay, p.data)
                p.data.add_(-group['lr'], new_d)


class SVRG_Snapshot(Optimizer):
    r"""Optimization class for calculating the mean gradient (snapshot) of all samples.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """

    def __init__(self, params):
        defaults = dict()
        super(SVRG_Snapshot, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups

    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer.
        """
        for group, new_group in zip(self.param_groups, new_params):
            for p, q in zip(group['params'], new_group['params']):
                p.data[:] = q.data[:]



class SARAH_optim(Optimizer):
    r"""Optimization Class for calculating the gradient of one iteration.
            Args:
                params (iterable): iterable of parameters to optimize or dicts defining
                    parameter groups
                lr (float): learning rate
            """

    def __init__(self, params, lr, weight_decay=0):
        print("Optimizer: SARAH")
        self.v = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        self.param_list = []
        super(SARAH_optim, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups

    def set_v(self, new_v):
        """Initialize the first unbiased gradient estimator (same as SVRG) every epoch
        """
        if self.v is None:
            self.v = copy.deepcopy(new_v)


        for v_group, new_group in zip(self.v, new_v):
            for v, new_v in zip(v_group['params'], new_group['params']):
                v.grad = new_v.grad.clone()

    def step(self, prev_params):
        """Performs a single optimization step.
        """
        for group, prev_group, v_group, d_group in zip(self.param_groups, prev_params, self.v, self.prev_param_groups):
            weight_decay = group['weight_decay']

            for p, q, v, d in zip(group['params'], prev_group['params'], v_group['params'], d_group['params']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue

                # core SVRG gradient update
                new_v = p.grad.data - q.grad.data + v.grad.data

                d.data[:] = p.data[:]
                if weight_decay != 0:
                    new_v.add_(weight_decay, p.data)
                p.data.add_(-group['lr'], new_v)

                v.grad = new_v.clone()

class SARAH_Snapshot(Optimizer):
    r"""Optimization class for calculating the mean gradient (snapshot) of all samples.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """

    def __init__(self, params):
        defaults = dict()
        super(SARAH_Snapshot, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups

    def set_param_groups(self, new_params):
        """Copies the parameters from the other optimizer.
        """
        for group, new_group in zip(self.param_groups, new_params):
            for p, q in zip(group['params'], new_group['params']):
                p.data[:] = q.data[:]



class STORM_optim(Optimizer):
    r"""Optimization Class for calculating the gradient of one iteration.
            Args:
                params (iterable): iterable of parameters to optimize or dicts defining
                    parameter groups
                lr (float): learning rate
            """

    def __init__(self, params, lr, alpha = 0.5,weight_decay=0):
        print("Optimizer: STORM-PG")
        self.v = None
        self.prev_param_groups = None
        self.alpha = alpha
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(STORM_optim, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups

    def set_v(self, new_v):
        """Initialize the first unbiased gradient estimator (same as SVRG) every epoch
        """
        if self.v is None:
            self.v = copy.deepcopy(new_v)


        for v_group, new_group in zip(self.v, new_v):
            for v, new_v in zip(v_group['params'], new_group['params']):
                v.grad = new_v.grad.clone()

    def step(self):
        """Performs a single optimization step.
        """
        for group, new_group, v_group in zip(self.prev_param_groups, self.param_groups, self.v):
            weight_decay = group['weight_decay']
            for p, q, v in zip(group['params'], new_group['params'], v_group['params']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue

                # main STORM gradient update
                new_v = q.grad.data + (1-self.alpha) * (v.grad.data - p.grad.data)

                if weight_decay != 0:
                    new_v.add_(weight_decay, q.data)
                q.data.add_(-group['lr'], new_v)

                # update v
                v.grad = new_v.clone()

                # update prev gradients
                p.grad.data[:] = q.grad.data[:]


class PAGEPG_optim(Optimizer):
    r"""Optimization Class for calculating the gradient of one iteration.
            Args:
                params (iterable): iterable of parameters to optimize or dicts defining
                    parameter groups
                lr (float): learning rate
            """

    def __init__(self, params, lr, weight_decay=0):
        print("Optimizer: PAGE-PG")
        self.v = None
        self.prev_param_groups = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(PAGEPG_optim, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups

    def set_v(self, new_v):
        """Initialize the first unbiased gradient estimator (same as SVRG) every epoch
        """
        if self.v is None:
            self.v = copy.deepcopy(new_v)


        for v_group, new_group in zip(self.v, new_v):
            for v, new_v in zip(v_group['params'], new_group['params']):
                v.grad = new_v.grad.clone()

    def step(self):
        """Performs a single optimization step.
        """
        for group, new_group, v_group in zip(self.prev_param_groups, self.param_groups, self.v):
            weight_decay = group['weight_decay']
            for p, q, v in zip(group['params'], new_group['params'], v_group['params']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue

                # main PAGE-PG gradient update
                # define the respective probabilities
                action = np.random.choice(np.arange(2),p = np.array([0.5,0.5]))

                # define the respective probabilities
                if action == 0:
                    # normal SGD
                    new_v = q.grad.data
                else:
                    # SARAH
                    new_v = q.grad.data - p.grad.data + v.grad.data

                if weight_decay != 0:
                    new_v.add_(weight_decay, q.data)
                q.data.add_(-group['lr'], new_v)

                # update v
                v.grad.data = new_v.clone()

                # update prev gradients
                p.grad.data[:] = q.grad.data[:]


class PAGE_STORM_PG_optim(Optimizer):
    r"""Optimization Class for calculating the gradient of one iteration.
            Args:
                params (iterable): iterable of parameters to optimize or dicts defining
                    parameter groups
                lr (float): learning rate
            """

    def __init__(self, params, lr, alpha = 0.5, weight_decay=0):
        print("Optimizer: PAGE-STORM-PG")
        self.v = None
        self.prev_param_groups = None
        self.alpha = alpha
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(PAGE_STORM_PG_optim, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups

    def set_v(self, new_v):
        """Initialize the first unbiased gradient estimator (same as SVRG) every epoch
        """
        if self.v is None:
            self.v = copy.deepcopy(new_v)


        for v_group, new_group in zip(self.v, new_v):
            for v, new_v in zip(v_group['params'], new_group['params']):
                v.grad = new_v.grad.clone()

    def step(self):
        """Performs a single optimization step.
        """
        for group, new_group, v_group in zip(self.prev_param_groups, self.param_groups, self.v):
            weight_decay = group['weight_decay']
            for p, q, v in zip(group['params'], new_group['params'], v_group['params']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue

                # main PAGE-PG gradient update
                # define the respective probabilities
                action = np.random.choice(np.arange(2),p = np.array([0.5,0.5]))

                # define the respective probabilities
                if action == 0:
                    # normal SGD
                    new_v = q.grad.data
                else:
                    # STORM
                    new_v = q.grad.data + (1 - self.alpha) * (v.grad.data - p.grad.data)

                # update prev timepoint
                p.grad.data[:] = q.grad.data[:]

                if weight_decay != 0:
                    new_v.add_(weight_decay, q.data)
                q.data.add_(-group['lr'], new_v)

                # update v
                v.grad.data = new_v.clone()




