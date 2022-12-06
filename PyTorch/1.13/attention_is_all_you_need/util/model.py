import os
import torch
from contextlib import contextmanager, ExitStack
import warnings
import inspect
from typing import ContextManager, Optional, List, Tuple, Generator
from torchbenchmark.util.extra_args import check_correctness_p, parse_opt_args, apply_opt_args, \
                                           parse_decoration_args, apply_decoration_args
from torchbenchmark.util.env_check import set_random_seed, correctness_check, stableness_check

class PostInitProcessor(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post__init__()
        return obj

@contextmanager
def no_grad(val):
    """Some meta-learning models (e.g. maml) may need to train a target(another) model
    in inference runs
    """
    old_state = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(not val)
        yield
    finally:
        torch.set_grad_enabled(old_state)

@contextmanager
def nested(*contexts):
    """
    Chain and apply a list of contexts
    """
    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield contexts

class BenchmarkModel(metaclass=PostInitProcessor):
    DEFAULT_TRAIN_BSIZE: Optional[int] = None
    DEFAULT_EVAL_BSIZE: Optional[int] = None
    # by default, deepcopy the model when checking correctness
    # because some models are stateful (such as moco)
    DEEPCOPY: bool = True

    test: str
    device: str
    jit: bool
    batch_size: int
    extra_args: List[str]
    run_contexts: List[ContextManager]

    """
    A base class for adding models to torch benchmark.
    See [Adding Models](#../models/ADDING_MODELS.md)
    """
    def __init__(self, test: str, device: str, jit: bool=False, batch_size: Optional[int]=None, extra_args: List[str]=[]):
        self.test = test
        assert self.test == "train" or self.test == "eval", f"Test must be 'train' or 'eval', but get {self.test}. Please submit a bug report."
        self.device = device
        self.jit = jit
        self.batch_size = batch_size
        if not self.batch_size:
            self.batch_size = self.DEFAULT_TRAIN_BSIZE if test == "train" else self.DEFAULT_EVAL_BSIZE
            # If the model doesn't implement test or eval test
            # its DEFAULT_TRAIN_BSIZE or DEFAULT_EVAL_BSIZE will still be None
            if not self.batch_size:
                raise NotImplementedError(f"Test {test} is not implemented.")
        # Check if customizing batch size is supported
        if hasattr(self, "ALLOW_CUSTOMIZE_BSIZE") and (not getattr(self, "ALLOW_CUSTOMIZE_BSIZE")):
            if test == "train" and (not self.batch_size == self.DEFAULT_TRAIN_BSIZE):
                raise NotImplementedError("Model doesn't support customizing batch size.")
            elif test == "eval" and (not self.batch_size == self.DEFAULT_EVAL_BSIZE):
                raise NotImplementedError("Model doesn't support customizing batch size.")
        self.extra_args = extra_args
        # contexts to run in the test function
        self.run_contexts = []
        set_random_seed()

    # Run the post processing for model acceleration
    def __post__init__(self):
        # sanity checks of the options
        assert self.test == "train" or self.test == "eval", f"Test must be 'train' or 'eval', but provided {self.test}."
        self.dargs, opt_args = parse_decoration_args(self, self.extra_args)
        # if the args contain "--torchdynamo", parse torchdynamo args
        if "--torchdynamo" in opt_args:
            self.dynamo = True
            from torchbenchmark.util.backends.torchdynamo import parse_torchdynamo_args
            self.opt_args = parse_torchdynamo_args(self, opt_args)
        else:
            self.dynamo = False
            self.opt_args = parse_opt_args(self, opt_args)
        self.need_correctness_check = check_correctness_p(self, self.opt_args)
        # currently, only check correctness under CUDA+inference, and `need_correctness_check` is True
        if self.need_correctness_check:
            self.eager_output = stableness_check(self, cos_sim=False, deepcopy=self.DEEPCOPY)
        # apply decoration args
        apply_decoration_args(self, self.dargs)
        # apply optimization args
        if self.dynamo:
            from torchbenchmark.util.backends.torchdynamo import apply_torchdynamo_args
            apply_torchdynamo_args(self, self.opt_args, self.dargs.precision)
        else:
            apply_opt_args(self, self.opt_args)
        # if test is eval, check correctness
        if self.need_correctness_check:
            # tensorrt or fp16 is known to generate less-accurate results
            # in this case, use more relaxed cosine similarity instead of torch.allclose
            # for correctness testing
            # see: https://github.com/pytorch/torchdynamo/pull/438
            if self.dargs.precision == "fp16" or (self.dynamo and self.opt_args.torchdynamo == "fx2trt") or (not self.dynamo and self.opt_args.fx2trt):
                self.correctness = correctness_check(self, cos_sim=True, deepcopy=self.DEEPCOPY)
            else:
                self.correctness = correctness_check(self, cos_sim=False, deepcopy=self.DEEPCOPY)
            torch.cuda.empty_cache()

    def add_context(self, context_fn):
        ctx = context_fn()
        assert isinstance(ctx, ContextManager), f"Expected adding a ContextManager, get {type(ctx)}. Please report a bug."
        self.run_contexts.append(context_fn)

    # Default implementation for replacing the model
    def set_module(self, new_model):
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            self.model = new_model
        else:
            raise NotImplementedError("The instance variable 'model' does not exist or is not type 'torch.nn.Module', implement your own `set_module()` function.")

    def gen_inputs(self, num_batches: int=1) -> Tuple[Generator, Optional[int]]:
        """Generate a tuple of (iterator of model input, the size of the iterator).
           If size is None, the input is randomly generated and has infinite size."""
        raise NotImplementedError("Default input generation function is not implemented. "
                                  "Please submit an issue if you need input iterator implementation for the model.")

    def invoke(self) -> Optional[Tuple[torch.Tensor]]:
        out = None
        with nested(*self.run_contexts):
            if self.test == "train":
                self.train()
            elif self.test == "eval":
                out = self.eval()
        return out

    def eval_in_nograd(self):
        return True

    def check_opt_vs_noopt_jit(self):
        if not self.jit:
            return

        model_name = inspect.getfile(self.__class__).split(os.sep)[-2]
        print(f"model_name={model_name} , {inspect.getfile(self.__class__)}")
        model_blacklist = [
            'demucs', # set up issue
            'yolov3', # set up issue
            'BERT_pytorch', # set up issue
            'moco', # set up issue
            'Super_SloMo', # results don't match, might be due to the way TE CUDA handles rand?
            'attention_is_all_you_need_pytorch', # results don't match, might be due to the way TE CUDA handles rand?
        ]

        if model_name in model_blacklist:
            warnings.warn(UserWarning(f"{model_name}.get_module() doesn't support `check_results` yet!"))
            return

        # if a model doesn't support `get_module`
        # we should let it throw and then
        # override `check_results` for that model
        try:
            model, inputs = self.get_module()
        except NotImplementedError:
            warnings.warn(UserWarning(f"{model_name}.get_module() doesn't support `check_results` yet!"))
            return

        def bench_allclose(a, b):
            if isinstance(a, torch.Tensor):
                assert(isinstance(b, torch.Tensor))
                assert(a.allclose(b))
            elif isinstance(a, tuple) or isinstance (b, list):
                assert(type(a) == type(b))
                assert(len(a) == len(b))
                for i in range(len(a)):
                    bench_allclose(a[i], b[i])
            else:
                raise RuntimeError("Encountered an supported type.\n" +
                    "Please add the type or override `bench_allclose`")

       
        try:
            opt = model(*inputs)
        except Exception as e:
            print(e)
            warnings.warn(UserWarning(f"{model_name}.eval() doesn't support `check_results` yet!"))
            return
        
        # disable optimizations and force a recompilation
        # to a baseline version
        fwd = model._c._get_method("forward")
        fwd._debug_flush_compilation_cache()
        torch._C._set_graph_executor_optimize(False)
        base = model(*inputs)
        torch._C._set_graph_executor_optimize(True)

        bench_allclose(base, opt)
