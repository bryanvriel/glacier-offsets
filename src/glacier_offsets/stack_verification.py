"""Bounded, read-only acquisition verification ahead of sequential processing."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging
from pathlib import Path

from .products import ChecksumCache, identity
from .stack_config import verification_workers
from .stack_storage import alignment_verified, date_specification, read_manifest, verified


def trusted_restart_caches(root, acquisitions):
    """Seed explicitly trusted digests from an existing stack's manifests."""
    root = Path(root)
    frame = read_manifest(root/'geometry')
    if frame.get('status') != 'complete' or not frame.get('products'):
        raise ValueError('--no-check requires an existing complete reference; run without it first')
    products, sources = ChecksumCache(), ChecksumCache()
    products.trust_existing(frame['products'])
    records = [frame['specification'].get('inputs', {})]
    for acquisition in acquisitions:
        saved = read_manifest(root/'acquisitions'/acquisition.date)
        for key in ('alignment_products', 'products'):
            products.trust_existing(saved.get(key, {}))
        records.append(saved.get('specification', {}))
    for record in records:
        for value in record.values():
            if isinstance(value, dict) and 'path' in value and 'sha256' in value:
                sources.trust_existing({value['path']: value['sha256']})
    return products, sources


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
    if checksums.trusted_mode:
        logging.info('Checking stack acquisition metadata with %d worker(s); recorded hashes trusted', count)
    else:
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
