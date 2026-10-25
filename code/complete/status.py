from pathlib import Path
import argparse
from common import ROOT,read_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'results'));a=p.parse_args();root=Path(a.out)
    results=list(root.glob('*/result.json'));print('Complete:',len(results))
    if (root/'progress.json').exists():print(read_json(root/'progress.json'))
    files=list(root.glob('*/status.json'))
    if files:
        latest=max(files,key=lambda p:p.stat().st_mtime);print('Latest job:',latest.parent.name,read_json(latest))
    if (root/'last_error.json').exists():print('Recorded error (may predate a successful retry):',read_json(root/'last_error.json')['error'])
