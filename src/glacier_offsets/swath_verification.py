"""Bounded read-only checksum prefetch for calibration restarts.

Only warm the revision-aware cache. Stage-local validation remains authoritative:
failed reads are retried there so unavailable links retain their usual handling.
Keep this scheduling code outside numerical implementation fingerprints.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path

from .stack_config import verification_workers


def verification_paths(experiment, stages, pilot_only=False):
    """Collect registered products; never crawl raw SAFE or SLC directories."""
    stages = set(stages)
    if not stages.intersection(('measure', 'physics', 'fit', 'previews')):
        return
    yield from experiment.frame.get('products', {})
    days = experiment.dates[:3] if pilot_only else experiment.dates
    selected_days = set(days)
    manifests = [experiment.stack/'acquisitions'/day/'manifest.json' for day in days]
    reports = []
    directories = ['measurements']
    if stages.intersection(('physics', 'fit', 'previews')):
        directories.append('physics')
    if stages.intersection(('fit', 'previews')):
        reports.append(experiment.output/'fit.json')
    if 'previews' in stages:
        directories.append('previews')
    pairs = {link['pair'] for link in experiment.links
             if not pilot_only or {link['earlier'], link['later']} <= selected_days}
    for directory in directories:
        for pair in sorted(pairs):
            reports.extend((experiment.output/directory/pair).glob('**/report.json'))
    for path in [*manifests, *reports]:
        try:
            record = json.loads(path.read_text())
            products = record.get('alignment_products' if path in manifests else 'products', {})
            if not isinstance(products, dict):
                continue
        except (OSError, ValueError, AttributeError):
            # The responsible stage supplies the normal missing/stale diagnostic.
            continue
        yield from products


def prefetch_verification(experiment, stages, *, workers=None, pilot_only=False):
    """Hash distinct files concurrently, joining readers before releasing locks."""
    count = verification_workers(experiment.cfg) if workers is None else workers
    if type(count) is not int or count < 1:
        raise ValueError('Verification workers must be a positive integer')
    registered = dict.fromkeys(str(Path(p).resolve())
                               for p in verification_paths(experiment, stages, pilot_only))
    paths = iter(registered)
    logging.info('Verifying %d calibration inputs and cached products with %d worker(s)', len(registered), count)
    pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix='swath-check')
    pending = deque()
    failures, completed = 0, 0

    def submit():
        path = next(paths, None)
        if path is not None:
            pending.append((path, pool.submit(experiment.inputs.hashes, path)))

    try:
        for _ in range(count):
            submit()
        while pending:
            path, future = pending.popleft()
            try:
                future.result()
            except (OSError, ValueError) as exc:
                failures += 1
                logging.debug('Calibration verification deferred to owning stage: %s: %s', path, exc)
            completed += 1
            if completed % 25 == 0:
                logging.info('Calibration checksum prefetch: %d/%d files', completed, len(registered))
            submit()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    logging.info('Calibration checksum prefetch complete; %d read failure(s) deferred to stage validation', failures)
