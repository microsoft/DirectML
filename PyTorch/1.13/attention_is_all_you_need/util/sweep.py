import argparse
import os
import subprocess
from datetime import date, timedelta
from pathlib import Path
from torch_nightly import get_n_prior_nightly_wheels

def run_step(cmd, cwd=None, conda_env=None, verbose=True):
    if verbose:
        print(f"  # running step: {cmd}")
    if conda_env:
        cmd = f'conda run --prefix {conda_env} {cmd}'
    return subprocess.check_output(
        cmd,
        cwd=cwd,
        stderr=subprocess.STDOUT,
        shell=True)

def create_env(env, benchmark, wheelset, py_ver=3.7, verbose=True):
    run_step(f"conda create -y -q -p {env} python={py_ver}")
    run_step(f"pip install -q {wheelset['torch']['wheel']} {wheelset['torchvision']['wheel']}", conda_env=env)
    run_step(f"pip install -q --no-deps {wheelset['torchtext']['wheel']}", conda_env=env)
    run_step(f"python install.py", conda_env=env, cwd=benchmark)

def check_env(env):
    torch_ver = run_step(f'python -c "import torch; print(torch.__version__)"', conda_env=env)
    torchvision_ver = run_step(f'python -c "import torchvision; print(torchvision.__version__)"', conda_env=env)
    torchtext_ver = run_step(f'python -c "import torchtext; print(torchtext.__version__)"', conda_env=env)
    print(torch_ver, torchvision_ver, torchtext_ver)
    return True

def prepare_envs(num_prior, env_root, benchmark):
    wheelsets = get_n_prior_nightly_wheels(['torch', 'torchvision', 'torchtext'], num_prior)
    for wheelset in wheelsets:
        version = wheelset['torch']['version']
        env = Path(env_root) / f"torch-{version}-env"
        print(f"### Creating env for {version} with py{py_ver}")
        try:
            create_env(env, benchmark, wheelset)
            check_env(env)
        except Exception as e:
            print(f"### Failed creating env for {version}: {e}")
            continue

def run_benchmark(conda_env_path, benchmark_repo, output_file, coreset="4-47", filter=None, min_rounds=20):
    
    cmd = [
        f'conda run --prefix {conda_env_path}',
        f'taskset -c "{coreset}"',
        f'pytest test_bench.py -k "{filter}"',
        f'--benchmark-min-rounds {min_rounds}',
        f'--benchmark-json {output_file}'
    ]
    prepared_env = os.environ
    prepared_env['GOMP_CPU_AFFINITY'] = f"{coreset}" 
    output = subprocess.run(
        cmd,
        env=prepared_env,
        cwd=benchmark_repo,
        shell=True,
    )
    print(output)

if __name__ == "__main__":
            #   ' --benchmark-json {output_file}/$(date +"%Y%m%d_%H%M%S")_${c}.json'
    # prepare_envs(num_prior=60, env_root="/home/ec2-user/sweep_conda_envs")
    # prepare_envs(num_prior=60, env_root="/home/ec2-user/sweep_conda_envs", benchmark="/home/ec2-user/benchmark")
    run_benchmark('/home/ec2-user/sweep_conda_envs/torch-1.8.0.dev20201219-env',
        '/home/ec2-user/benchmark',
        '/home/ec2-user/test_benchmark.json',
        filter="(bert and cpu and eval and eager)")