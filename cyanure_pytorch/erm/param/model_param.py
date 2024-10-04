class ModelParameters:

    max_iter: int
    tol: float
    duality_gap_interval: int
    verbose: bool
    max_iter_backtracking: int
    minibatch: int
    threads: int
    non_uniform_sampling: bool
    l_memory: int
    freq_restart: int
    solver: str

    def __init__(self, max_iter: int = 500, tol: float = 1e-3, duality_gap_interval: int = 10, max_iter_backtracking: float = 500,
                 minibatch: int = 1, threads: int = -1, l_memory: int = 20, freq_restart: int = 50, verbose: bool = False,
                 non_uniform_sampling: bool = True, solver: str = "QNING-ISTA"):
        self.max_iter = max_iter
        self.tol = tol
        self.duality_gap_interval = duality_gap_interval
        self.max_iter_backtracking = max_iter_backtracking
        self.minibatch = minibatch
        self.threads = threads
        self.l_memory = l_memory
        self.freq_restart = freq_restart
        self.verbose = verbose
        self.non_uniform_sampling = non_uniform_sampling
        self.solver = solver
