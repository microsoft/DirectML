"""Utilities for tuning the machine for better benchmark stability.

Written for Amazon linux and Intel CPU, Nvidia GPU althogh many utilities will overlap.
"""
import argparse
import cpuinfo
import distro
import enum
import os
import platform
import psutil
import subprocess
import re
import sys
import typing
from pathlib import Path

def read_sys_file(sysfile: Path):
    with open(sysfile, 'r') as f:
        return f.read()

def write_sys_file(sysfile: Path, content: str):
    print(f"Write {content} to {sysfile}")
    with open(sysfile, 'w') as f:
        f.write(content)

def check_intel_no_turbo_state(turbo_file='/sys/devices/system/cpu/intel_pstate/no_turbo'):
    return int(read_sys_file(turbo_file))

def set_intel_no_turbo_state(state: int, turbo_file='/sys/devices/system/cpu/intel_pstate/no_turbo'):
    assert state in [0, 1]
    write_sys_file(turbo_file, str(state))

def parse_lscpu_cpu_core_list():
    coreinfo = subprocess.check_output("lscpu --all --parse=CPU,CORE,ONLINE", shell=True).strip().decode().split('\n')
    matched_cpus = 0
    cpu_core = []
    for line in coreinfo[2:]:
        if line[0] == '#':
            continue
        cpu, core, online = line.split(',')
        cpu = int(cpu)
        online = online == "Y"
        core = int(core) if online else None
        if cpu == core:
            matched_cpus += 1
        cpu_core.append((cpu, core, online))
    assert matched_cpus > 0, "Failed to parse lscpu output"
    return cpu_core


def hyper_threading_enabled():
    for cpu, core, online in parse_lscpu_cpu_core_list():
        if cpu != core and online:
            return True
    return False

def set_hyper_threading(enabled=False):
    for cpu, core, online in parse_lscpu_cpu_core_list():
        if cpu != core:
            if not online and not enabled:
                continue
            if online and enabled:
                continue
            virtual_cpu_online_file = f"/sys/devices/system/cpu/cpu{cpu}/online"
            value = "1" if enabled else "0"
            write_sys_file(virtual_cpu_online_file, value)

def get_intel_max_cstate():
    kernel_args = read_sys_file('/proc/cmdline').split()
    for arg in kernel_args:
        if arg.find('intel_idle.max_cstate') == 0:
            return int(arg.split('=')[1])
    return None

def get_isolated_cpus():
    """
    Returns a list of cpus marked as isolated from the kernel scheduler for regular tasks.
    Only tasks scheduled via taskset command can use these cpus, e.g. benchmarking workload.
    """
    kernel_args = read_sys_file('/proc/cmdline').split()
    isolcpus = set()
    for arg in kernel_args:
        if arg.find('isolcpus') == 0:
            arg = arg.split('=')[1]
            chunks = arg.split(',')
            for chunk in chunks:
                if '-' in chunk:
                    start, end = chunk.split('-')
                    for cpu in range(int(start), int(end) + 1):
                        isolcpus.add(cpu)
                else:
                    isolcpus.add(int(chunk))
    return list(isolcpus)

def get_process_cpu_affinity():
    p = psutil.Process()
    return p.cpu_affinity()

def nvidia_smi_query(query: str, device_ids: typing.List[int] = None):
    if device_ids:
        device_ids = [str(id) for id in device_ids]
        device_ids = ",".join(device_ids)
    id_selector = f"-i {device_ids}" if device_ids else ""
    values = subprocess.check_output(f'nvidia-smi --query-gpu="{query}" {id_selector} --format=csv,noheader,nounits',
                                       shell=True).strip().decode().split("\n")
    return values

def has_nvidia_smi():
    try:
        subprocess.check_output('nvidia-smi', shell=True)
        return True
    except:
        return False

def get_nvidia_gpu_clocks(device_ids: typing.List[int] = None):
    clocks = nvidia_smi_query("clocks.applications.graphics", device_ids)
    return [int(clock) for clock in clocks]

def get_nvidia_gpu_temps(device_ids: typing.List[int] = None):
    temps = {}
    raw_temps = nvidia_smi_query("temperature.gpu,temperature.memory", device_ids)
    temps['gpu'] = [temp.split(',')[0] for temp in raw_temps]
    temps['memory'] = [temp.split(',')[1] for temp in raw_temps]
    return temps

def set_nvidia_graphics_clock(device_id=0, clock=900):
    if has_nvidia_smi():
        return subprocess.check_call(['nvidia-smi', '-ac', '5001,900'])
    return False

def get_nvidia_throttle_reasons(device_ids: typing.List[int] = None):
    """ See 'nvidia-smi --help-query-gpu for explanation of throttle reasons
    """
    queries = ['gpu_idle', 'applications_clocks_setting', 'sw_power_cap', 
        'hw_slowdown', 'hw_thermal_slowdown',
        'hw_power_brake_slowdown', 'sw_thermal_slowdown', 'sync_boost']
    query_str = ','.join(["clocks_throttle_reasons." + q for q in queries])
    raw = nvidia_smi_query(query_str, device_ids)
    throttle_reasons = []
    for line in raw:
        gpu_reasons = [q for q, v in zip(queries, line.split(',')) if 'Active' == v]
        throttle_reasons.append(gpu_reasons)
    return throttle_reasons

MACHINE = enum.Enum('MACHINE', ['AMAZON_LINUX', 'UNKNOWN'])
def get_machine_type():
    # It's tricky to write platform setup code that works on different OS/configs.
    # initially, just intend to identify a known environment and for any other 
    # environment revert to no-op.  Expand functionality over time as needed.
    if platform.system() == 'Linux':
        if distro.name() == "Amazon Linux":
            return MACHINE.AMAZON_LINUX

    return MACHINE.UNKNOWN

def get_cpu_temp():
    temps = {}
    if MACHINE.AMAZON_LINUX == get_machine_type():
        thermal_path = Path('/sys/class/thermal/')
        for zone in filter(lambda x: "thermal_zone" in x, os.listdir(thermal_path)):
            temps[zone] = int(read_sys_file(thermal_path / zone / "temp")) / 1000.
    return temps

def is_using_isolated_cpus():
    isolated_cpus = get_isolated_cpus()
    using_cpus = get_process_cpu_affinity()
    omp_using_cpus = get_omp_affinity()
    lscpu = parse_lscpu_cpu_core_list()
    assert len(lscpu) > 0, "unable to parse current CPUs"
    for cpu, core, active in lscpu:

        # check that all used cpus are isolated ones (more critical)
        if (cpu in using_cpus or cpu in omp_using_cpus) and cpu not in isolated_cpus:
            return False

        # check all isolated cpus are used (less critical)
        elif active and cpu in isolated_cpus:
            if cpu not in using_cpus:
                # currently after importing torch, process cpu affinity mask changes from e.g. 4-47 to 4.
                # since we can't assert that all intended cores are being used, we can at least assert that
                # the first core in the range of isolated cores is used.
                # see https://github.com/pytorch/pytorch/issues/49971
                # return False
                pass
            if cpu not in omp_using_cpus:
                return False
    return True

def get_omp_affinity():
    if 'GOMP_CPU_AFFINITY' not in os.environ:
        return []
    raw = os.environ['GOMP_CPU_AFFINITY']
    affinity = []

    def parse_block(block):
        if '-' in block:
            start, end = block.split('-')
            return list(range(int(start), int(end) + 1))
        return [int(block)]

    if ' ' in raw:
        for block in raw.split(' '):
            affinity.extend(parse_block(block)) 
    else:
        affinity.extend(parse_block(raw))
    return affinity

def get_pstate_frequency():
    CPU_FREQ_BASE_DIR = '/sys/devices/system/cpu'
    CPU_FREQ_FILES = ["scaling_min_freq", "scaling_max_freq", "scaling_cur_freq"]
    cpu_dirs = ["cpu" + str(cpu[0]) for cpu in parse_lscpu_cpu_core_list() if cpu[2]]
    output = dict()
    for cpu_dir in cpu_dirs:
        full_path = os.path.join(CPU_FREQ_BASE_DIR, cpu_dir, "cpufreq")
        freq_paths = [os.path.join(full_path, x) for x in CPU_FREQ_FILES]
        all_exist = True
        for path in freq_paths:
            all_exist = all_exist and os.path.exists(path)
        if all_exist:
            output[cpu_dir] = dict()
            for i, path in enumerate(freq_paths):
                output[cpu_dir][CPU_FREQ_FILES[i]] = int(read_sys_file(path)) / 1000
    return output

def set_pstate_frequency(min_freq = 2500, max_freq = 2500):
    CPU_FREQ_BASE_DIR = '/sys/devices/system/cpu'
    CPU_FREQ_FILES = ["scaling_min_freq", "scaling_max_freq", "scaling_cur_freq"]
    cpu_dirs = ["cpu" + str(cpu[0]) for cpu in parse_lscpu_cpu_core_list() if cpu[2]]
    for cpu_dir in cpu_dirs:
        full_path = os.path.join(CPU_FREQ_BASE_DIR, cpu_dir, "cpufreq")
        freq_paths = [os.path.join(full_path, x) for x in CPU_FREQ_FILES]
        all_exist = True
        for path in freq_paths:
            all_exist = all_exist and os.path.exists(path)
        if all_exist:
            write_sys_file(freq_paths[0], str(min_freq * 1000))
            write_sys_file(freq_paths[1], str(max_freq * 1000))

def check_pstate_frequency_pin(pin_freq = 2500):
    FREQ_THRESHOLD = 15  # Allow 15 MHz difference maximum
    all_freq = get_pstate_frequency()
    for cpuid in all_freq:
        for attr in all_freq[cpuid]:
            freq = all_freq[cpuid][attr]
            difference = abs(freq - pin_freq)
            if difference > FREQ_THRESHOLD:
                print(f"Specify frequency {pin_freq} Mhz, find setting {cpuid} {attr}: {freq}.")
                return False
    return True

def get_machine_config():
    config = {}
    machine_type = get_machine_type()
    config['machine_type'] = machine_type
    config['cpu_brand'] = cpuinfo.get_cpu_info()['brand_raw']
    if MACHINE.AMAZON_LINUX == machine_type:
        config['linux_distribution'] = distro.linux_distribution()
        config['intel_turbo_disabled'] = check_intel_no_turbo_state()
        config['intel_hyper_threading_enabled'] = hyper_threading_enabled()
        config['intel_max_cstate'] = get_intel_max_cstate()
        config['isolated_cpus'] = get_isolated_cpus()
        config['process_cpu_affinity'] = get_process_cpu_affinity()
        config['is_using_isolated_cpus'] = is_using_isolated_cpus()
        config['cpu_pstate_frequency'] = get_pstate_frequency()
    return config

def check_machine_configured(check_process_affinity=True):
    check_environment()
    if MACHINE.AMAZON_LINUX == get_machine_type():
        assert 1 == check_intel_no_turbo_state(), "Turbo Boost is not disabled"
        assert False == hyper_threading_enabled(), "HyperThreading is not disabled"
        assert 1 == get_intel_max_cstate(), "Intel max C-State isn't set to 1, which avoids power-saving modes."
        assert len(get_isolated_cpus()) > 0, "No cpus are isolated for benchmarking with isolcpus"
        assert 900 == get_nvidia_gpu_clocks()[0], "Nvidia gpu clock isn't limited, to increase consistency by reducing throttling"
        assert is_using_isolated_cpus(), "taskset or GOMP_CPU_AFFINITY not specified or not matching kernel isolated cpus"
        assert check_pstate_frequency_pin(), "Must pin CPU frequency to a fixed number in MHz"
    else:
        raise RuntimeError(f"Unsupported machine type {get_machine_type()}")

def get_machine_state():
    state = {}
    machine_type = get_machine_type()
    state['machine_type'] = machine_type
    if MACHINE.AMAZON_LINUX == machine_type:
        state['cpu_temps'] = get_cpu_temp()
        if has_nvidia_smi():
            state['nvidia_gpu_temps'] = get_nvidia_gpu_temps()
            state['nvidia_gpu_clocks'] = get_nvidia_gpu_clocks()
            state['nvidia_gpu_throttle_reasons'] = get_nvidia_throttle_reasons()
            state['process_cpu_affinity'] = get_process_cpu_affinity()
    return state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable_ht", action="store_true", help="Enable HyperThreading")
    parser.add_argument("--configure", action="store_true", help="Apply benchmark tuning to this machine")
    parser.add_argument("--no_verify", action="store_true", help="Skip verifying machine is configured for benchmarking")
    args = parser.parse_args()

    machine_type = get_machine_type()
    if not MACHINE.AMAZON_LINUX == machine_type:
        raise RuntimeError(f"Unsupported machine type {machine_type}")

    if args.enable_ht:
        set_hyper_threading(True)

    if args.configure:
        set_intel_no_turbo_state(1)
        set_hyper_threading(False)
        set_nvidia_graphics_clock()
        set_pstate_frequency()

    if not args.no_verify:
        assert 1 == check_intel_no_turbo_state(), "Turbo Boost is not disabled"
        assert False == hyper_threading_enabled(), "HyperThreading is not disabled"
        assert 1 == get_intel_max_cstate(), "Intel max C-State isn't set to 1, which avoids power-saving modes."
        assert len(get_isolated_cpus()) > 0, "No cpus are isolated for benchmarking with isolcpus"
        assert 900 == get_nvidia_gpu_clocks()[0], "Nvidia gpu clock isn't limited, to increase consistency by reducing throttling"
        assert check_pstate_frequency_pin(), "CPU frequency is not correctly pinned, which is required to minimize noise."
        # doesn't make too much sense to ask the user to run this configure script with the isolated cpu cores
        # that check is more important to be done at runtime of benchmark, and is checked by conftest.py
        #assert is_using_isolated_cpus(), "Not using isolated CPUs for this process"

def check_environment():
    checks = [
        # VAR_NAME, blacklist
        ("DEBUG", None),
        ("MKLDNN_VERBOSE", None),
        ("PYTORCH_JIT_LOG_LEVEL", None)
    ]

    for check in checks:
        if check[0] in os.environ and (check[1] == None or os.environ[check[0]] in check[1]):
            raise RuntimeError(f"{check[0]} is set")
