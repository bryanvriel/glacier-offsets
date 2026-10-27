"""Bounded, read-only acquisition verification ahead of sequential processing."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging
from pathlib import Path

from .products import identity
from .stack_config import verification_workers
from .stack_storage import alignment_verified, date_specification, read_manifest, verified


def check_acquisition(cfg, acquisition, frame, hashes, checksums, number, total):
    """Warm revision-aware digests; the workflow rechecks revisions before reuse.

    Never run native geometry, write manifests, invalidate products or prepare
    an acquisition here. In particular a failed digest is handled by the main
    workflow for this date, rather than aborting other dates' checks.
    """
    logging.info('Checking stack acquisition %s (%d/%d)', acquisition.date, number, total)
    specification = date_specification(acquisition, frame, hashes)
    previous = read_manifest(Path(cfg['output']['directory'])/'acquisitions'/acquisition.date)
    if alignment_verified(previous, identity(specification), checksums=checksums):
        if previous.get('assembly_identity'):
            verified(previous, 'assembly_identity', previous['assembly_identity'],
                     allow_released=True, checksums=checksums)
    return specification


@contextmanager
def acquisition_checks(cfg, acquisitions, frame, hashes, checksums):
    """Yield dates/futures in inventory order, with at most N checks in flight.

    Exceptions stay attached to their date. Cancellation stops queued checks;
    active read-only checks finish before the stack lock can be released.
    """
    count = min(verification_workers(cfg), len(acquisitions))
    logging.info('Verifying stack acquisitions with %d worker(s)', count)
    pool = ThreadPoolExecutor(max_workers=max(1, count), thread_name_prefix='stack-check')
    inventory = iter(enumerate(acquisitions, 1))
    pending = deque()

    def submit():
        item = next(inventory, None)
        if item is not None:
            number, acquisition = item
            pending.append((acquisition, pool.submit(check_acquisition, cfg, acquisition,
                frame, hashes, checksums, number, len(acquisitions))))

    def ordered():
        while pending:
            yield pending.popleft()
            submit()

    try:
        for _ in range(count):
            submit()
        yield ordered()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
