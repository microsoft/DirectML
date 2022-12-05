from typing import Optional, List

class E2EBenchmarkModel():
    """
    A base class for adding models for all e2e models.
    """
    def __init__(self, test: str, batch_size: Optional[int]=None, extra_args: List[str]=[]):
        self.test = test
        assert self.test == "train" or self.test == "eval", f"Test must be 'train' or 'eval', but get {self.test}. Please submit a bug report."
        self.batch_size = batch_size
        if not self.batch_size:
            self.batch_size = self.DEFAULT_TRAIN_BSIZE if test == "train" else self.DEFAULT_EVAL_BSIZE
            # If the model doesn't implement test or eval test
            # its DEFAULT_TRAIN_BSIZE or DEFAULT_EVAL_BSIZE will still be None
            if not self.batch_size:
                raise NotImplementedError(f"Test {test} is not implemented.")
        self.extra_args = extra_args
    
    def next_batch(self):
        raise NotImplementedError("Every E2EModel should implement this")
    
    def run_forward(self, input):
        raise NotImplementedError("Every E2EModel should implement this")

    def run_backward(self, loss):
        raise NotImplementedError("Every E2EModel should implement this")

    def run_optimizer_step(self):
        raise NotImplementedError("Every E2EModel should implement this")
