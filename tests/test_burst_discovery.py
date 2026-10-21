from datetime import datetime, timedelta
from types import SimpleNamespace
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import yaml
from glacier_offsets.inputs import Burst, inventory, intersect, select_pair
from glacier_offsets.cli import main


class BurstTests(unittest.TestCase):
    def setUp(self):
        self.t=datetime(2020,1,10)
    def burst(self,key,swath=1,dt=0):
        return Burst(key,swath,self.t+timedelta(days=dt))
    def test_multiple_swaths_geographic_intersection(self):
        a=[self.burst('t1_iw1'),self.burst('t2_iw2',2),self.burst('only_ref')]
        b=[self.burst('t2_iw2',2,12),self.burst('t1_iw1',1,12)]
        rows=intersect(a,b)
        self.assertEqual([x[0].id for x in rows],['t1_iw1','t2_iw2'])
        self.assertEqual(select_pair({'inputs':{'burst_id':'t2_iw2'}},rows)[0].swath,2)
        for key in [None,'not_common']:
            with self.assertRaisesRegex(ValueError,'list-bursts'):
                select_pair({'inputs':{'burst_id':key}},rows)
    def test_ambiguous_reversed_swath_and_empty(self):
        a=self.burst('a')
        for left,right in [([a,a],[self.burst('a',dt=12)]),([a],[self.burst('a',dt=-1)]),([a],[self.burst('a',2,12)])]:
            with self.assertRaises(ValueError):intersect(left,right)
        self.assertEqual(intersect([a],[self.burst('b',dt=12)]),[])
        self.assertEqual(intersect([],[]),[])
    def test_loader_arguments_and_errors(self):
        def loader(s,o,sw,pol,flag_apply_eap):
            self.assertFalse(flag_apply_eap);self.assertEqual(pol,'hh')
            return [SimpleNamespace(burst_id=f'b{sw}',sensing_start=self.t)]
        rows=inventory('safe','orbit','HH',[1,2,3],loader)
        self.assertEqual([x.swath for x in rows],[1,2,3])
        with self.assertRaisesRegex(ValueError,'Cannot parse'):
            inventory('missing','orbit','HH',[1],lambda *a,**k:(_ for _ in ()).throw(RuntimeError('bad XML')))
    def test_cli_without_burst_dem_aoi(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'local';p.touch()
            cfg={'inputs':{k:str(p) for k in ['reference_safe','secondary_safe','reference_orbit','secondary_orbit']}}
            f=Path(d)/'cfg.yaml';f.write_text(yaml.safe_dump(cfg))
            rows=[(self.burst('z_iw1'),self.burst('z_iw1',dt=12)),
                  (self.burst('a_iw3',3),self.burst('a_iw3',3,12)),
                  (self.burst('a_iw2',2),self.burst('a_iw2',2,12)),
                  (self.burst('a_iw1'),self.burst('a_iw1',dt=12))]
            with patch('glacier_offsets.cli.discover',return_value=rows),contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(['list-bursts',str(f)]),0)
                text=out.getvalue()
                self.assertIn('4 common bursts',text)
                self.assertLess(text.index('a_iw1'),text.index('z_iw1'))
                self.assertLess(text.index('z_iw1'),text.index('a_iw2'))
                self.assertLess(text.index('a_iw2'),text.index('a_iw3'))
            with patch('glacier_offsets.cli.discover',return_value=[]),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['list-bursts',str(f)]),2)
