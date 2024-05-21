"""gitutils.py

Utils for getting git-related information.
"""

import os
import subprocess
from datetime import datetime
from typing import Optional, List

def clean_git_repo(repo: str) -> bool:
    try:
        command = f"git clean -xdf"
        subprocess.check_call(command, cwd=repo, shell=True)
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to cleanup git repo {repo}")
        return None

def update_git_repo_branch(repo: str, branch: str) -> bool:
    try:
        command = f"git pull origin {branch}"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except subprocess.CalledProcessError:
        print(f"Failed to update git repo {repo}, branch {branch}")
        return None

def get_git_commit_on_date(repo: str, date: datetime) -> Optional[str]:
    try:
        # Get the first commit since date
        formatted_date = date.strftime("%Y-%m-%d")
        command = f"git log --until={formatted_date} -1 --oneline | cut -d ' ' -f 1"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except subprocess.CalledProcessError:
        print(f"Failed to get the last commit on date {formatted_date} in repo {repo}")
        return None
 
def check_git_exist_local_branch(repo: str, branch: str) -> bool:
    command = f"git rev-parse --verify {branch} &> /dev/null "
    retcode = subprocess.call(command, cwd=repo, shell=True)
    return (retcode == 0)

def get_git_commit_date(repo: str, commit: str) -> str:
    try:
        command = f"git show -s --format=%ci {commit}"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except subprocess.CalledProcessError:
        print(f"Failed to get date of commit {commit} in repo {repo}")
        return None
    
def checkout_git_branch(repo: str, branch: str) -> bool:
    try:
        if check_git_exist_local_branch(repo, branch):
            command = f"git checkout {branch} &> /dev/null "
        else:
            command = f"git checkout --track origin/{branch} &> /dev/null"
        retcode = subprocess.call(command, cwd=repo, shell=True)
        return (retcode == 0)
    except subprocess.CalledProcessError:
        print(f"Failed to checkout git repo {repo}, branch {branch}")
        return None

def get_current_branch(repo: str) -> Optional[str]:
    try:
        command = "git branch --show-current"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except subprocess.CalledProcessError:
        print(f"Failed to get current branch name for repo {repo}")
        return None

def get_git_origin(repo: str) -> Optional[str]:
    try:
        command = "git remote get-url origin"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except:
        print(f"git command {command} returns non-zero status in repo {repo}")
        return None

def get_git_commits(repo: str, start: str, end: str) -> Optional[List[str]]:
    try:
        command = f"git log --reverse --oneline --ancestry-path {start}^..{end} | cut -d \" \" -f 1"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip().split("\n")
        if out == ['']:
            out = None
        return out
    except subprocess.CalledProcessError:
        print(f"git command {command} returns non-zero status in repo {repo}")
        return None

def get_current_commit(repo: str) -> Optional[str]:
    try:
        command = f"git log --reverse --oneline -1 | cut -d \" \" -f 1"
        out = subprocess.check_output(command, cwd=repo, shell=True).decode().strip()
        return out
    except subprocess.CalledProcessError:
        print(f"Failed to get the current commit in repo {repo}")
        return None

def checkout_git_commit(repo: str, commit: str) -> bool:
    try:
        assert len(commit) != 0
        command = f"git checkout {commit}"
        subprocess.check_call(command, cwd=repo, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        command = "git submodule sync"
        subprocess.check_call(command, cwd=repo, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        command = "git submodule update --init --recursive"
        subprocess.check_call(command, cwd=repo, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to checkout commit {commit} in repo {repo}")
        return False

def update_git_repo(repo: str, branch: str="main") -> bool:
    try:
        assert len(branch) != 0
        command = f"git checkout {branch}"
        subprocess.check_call(command, cwd=repo, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        command = "git submodule sync"
        subprocess.check_call(command, cwd=repo, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        command = "git submodule update --init --recursive"
        subprocess.check_call(command, cwd=repo, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to update git repo {repo}")
        return False
