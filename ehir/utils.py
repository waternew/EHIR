from multiprocessing import Pool
from tqdm import tqdm

def run_parallel(worker, tsks, jobs, debug):
    assert len(tsks) > 0
    if debug:
        for tsk in tsks:
            worker(tsk)
    else:
        with Pool(min(len(tsks), jobs)) as p:
            max_ = len(tsks)
            with tqdm(total=max_) as pbar:
                for i, _ in enumerate(p.imap_unordered(worker, tsks)):
                    pbar.update()
