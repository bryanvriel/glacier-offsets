"""Invocation-local digest reuse must never bypass manifest or file checks."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glacier_offsets.products import ChecksumCache, checksum
from glacier_offsets.storage import products_valid, release_slcs


class ChecksumCacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.folder).resolve()
        self.path = self.root/'slc.tif'
        self.path.write_bytes(b'original')

    def test_unchanged_file_and_symlink_alias_hash_once_but_new_cache_hashes_again(self):
        alias = self.root/'alias.tif'; alias.symlink_to(self.path)
        expected = checksum(self.path)
        cache = ChecksumCache()
        with patch('glacier_offsets.products.checksum', wraps=checksum) as read:
            self.assertEqual(cache(self.path), expected)
            self.assertEqual(cache(alias), expected)
            self.assertEqual(cache(self.path), expected)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(ChecksumCache()(self.path), expected)
            self.assertEqual(read.call_count, 2)
        self.assertEqual(cache.hits, 2)
        self.assertEqual(cache.files_hashed, 1)
        self.assertEqual(cache.bytes_hashed, len(b'original'))

    def test_same_size_rewrite_with_restored_mtime_and_atomic_replacement_rehash(self):
        cache = ChecksumCache()
        original = cache(self.path)
        before = self.path.stat()
        self.path.write_bytes(b'modified')
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertNotEqual(cache(self.path), original)
        rewritten = cache(self.path)
        replacement = self.root/'replacement.tif'; replacement.write_bytes(b'replaced')
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(self.path)
        self.assertNotEqual(cache(self.path), rewritten)
        self.assertEqual(cache.files_hashed, 3)

    def test_missing_recreated_and_changed_manifest_digest_cannot_reuse_success(self):
        cache = ChecksumCache()
        products = {str(self.path):checksum(self.path)}
        manifest = {'identity':'a'}
        verify = lambda **kwargs: products_valid(products, manifest, 'identity', 'a', checksums=cache, **kwargs)
        self.assertTrue(verify())
        products[str(self.path)] = 'incorrect-digest'
        self.assertFalse(verify())
        products[str(self.path)] = checksum(self.path)
        self.path.unlink()
        self.assertFalse(verify())
        self.path.write_bytes(b'replaced')
        self.assertFalse(verify())
        self.assertFalse(products_valid(products, manifest, 'identity', 'changed', checksums=cache))

    def test_changed_during_read_is_rejected_and_not_cached(self):
        cache = ChecksumCache()
        def changing(path):
            digest = checksum(path)
            self.path.write_bytes(b'changed during hashing')
            return digest
        with patch('glacier_offsets.products.checksum', side_effect=changing):
            with self.assertRaisesRegex(ValueError, 'changed while computing checksum'):
                cache(self.path)
        self.assertEqual(cache(self.path), checksum(self.path))
        self.assertEqual(cache.hits, 0)

    def test_directory_children_are_not_hidden_by_cached_parent_metadata(self):
        cache = ChecksumCache()
        original = cache(self.root)
        self.path.write_bytes(b'modified')
        self.assertNotEqual(cache(self.root), original)
        self.assertEqual(cache.hits, 0)

    def test_release_still_requires_journal_and_existing_masks_still_verified(self):
        mask = self.root/'valid.tif'; mask.write_bytes(b'mask')
        products = {str(p):checksum(p) for p in (self.path, mask)}
        manifest = {'assembly_identity':'a', 'products':products}
        cache = ChecksumCache()
        self.assertTrue(products_valid(products, manifest, 'assembly_identity', 'a', checksums=cache))
        release_slcs(self.root, manifest, 'assembly_identity', [self.path], products)
        self.assertFalse(products_valid(products, manifest, 'assembly_identity', 'a', checksums=cache))
        self.assertTrue(products_valid(products, manifest, 'assembly_identity', 'a', allow_released=True, checksums=cache))
        mask.write_bytes(b'bad!')
        self.assertFalse(products_valid(products, manifest, 'assembly_identity', 'a', allow_released=True, checksums=cache))


if __name__ == '__main__':
    unittest.main()
