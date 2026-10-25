"""Run on the remote GPU server using the currently activated Python environment."""
import subprocess
import sys
import shlex
from common import ROOT

if __name__ == '__main__':
    args = sys.argv[1:] or ['--stage', 'all']
    cmd = [sys.executable, '-u', str(ROOT/'run_uniform.py')] + args
    with open(ROOT/'background.log', 'ab', buffering=0) as log:
        process = subprocess.Popen(cmd, cwd=str(ROOT), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    (ROOT/'background.pid').write_text(str(process.pid) + '\n')
    print('Remote process launched, PID:', process.pid)
    print('Python:', sys.executable)
    print('Check startup: python status.py')
    print('Follow log: tail -f ' + shlex.quote(str(ROOT/'background.log')))
    print('The remote task normally survives SSH disconnect/local laptop closure.')
    print('After server restart or termination, reconnect and repeat this command to resume.')
