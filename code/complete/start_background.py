"""Start on the REMOTE server in the currently activated Python environment."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
from common import ROOT

if __name__=='__main__':
    log=ROOT/'background.log'
    args=sys.argv[1:] or ['--stage','all']
    cmd=[sys.executable,'-u',str(ROOT/'run_all.py')]+args
    with open(log,'ab',buffering=0) as f:
        p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                           cwd=str(ROOT),start_new_session=True,close_fds=True)
    (ROOT/'background.pid').write_text(str(p.pid)+'\n')
    print('Remote process started, PID:',p.pid)
    print('Python:',sys.executable)
    print('Log:',log)
    print('Monitor: tail -f '+shlex.quote(str(log)))
    print('Disconnecting SSH or closing the local laptop does not normally terminate this detached remote process.')
    print('Server restart, scheduler limits or administrator cleanup can stop it. Rerun this same command to resume saved epochs.')
