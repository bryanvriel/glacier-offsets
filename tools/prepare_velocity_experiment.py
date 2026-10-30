"""Freeze verified baseline netCDFs and prepare a pinned two-year stack YAML.

This command never starts processing or widens a correction calibration.
Run in the ice environment with PYTHONPATH=src.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import h5netcdf
import yaml
from glacier_offsets.velocity_io import digest, revision, verify_hashes, write_json, writer_lock


def estimate_extension_capacity(stack_root, plan_path):
    """Conservative retained-alignment estimate from existing product byte sizes."""
    root=Path(stack_root).resolve()
    plan=json.loads(Path(plan_path).read_text())
    sizes={}
    for manifest in sorted((root/'acquisitions').glob('*/manifest.json')):
        if manifest.parent.name==plan['reference_date']:
            # Reference dates have no secondary resampling-offset products.
            continue
        record=json.loads(manifest.read_text())
        products=record.get('alignment_products',{})
        burst_sizes={}
        for filename in products:
            path=Path(filename)
            parts=path.parts
            if 'bursts' not in parts or not path.is_file():
                continue
            burst=parts[parts.index('bursts')+1]
            burst_sizes[burst]=burst_sizes.get(burst,0)+path.stat().st_size
        for burst,size in burst_sizes.items():
            sizes[burst]=max(sizes.get(burst,0),size)
        if set(plan['selected_burst_ids']) <= sizes.keys():
            # All aligned products use the same pinned reference burst shapes.
            # A complete representative date supplies the required byte budget.
            break
    required=0
    new_dates=[]
    unknown=[]
    for acquisition in plan['inventory']:
        if not acquisition.get('eligible'):
            continue
        date=acquisition['date']
        existing=root/'acquisitions'/date/'manifest.json'
        if existing.exists():
            continue
        new_dates.append(date)
        for burst in acquisition['available_burst_ids']:
            if burst not in sizes:
                unknown.append(burst)
            else:
                required+=sizes[burst]
    free=shutil.disk_usage(root).free
    return {'schema':'glacier-offsets-extension-capacity-v1','stack_root':str(root),
            'extended_plan_sha256':digest(plan_path),'new_dates':new_dates,'per_burst_alignment_bytes':sizes,
            'estimated_new_alignment_bytes':required,'available_bytes':free,
            'minimum_extra_alignment_bytes':max(required-free,0),'unestimated_bursts':sorted(set(unknown)),
            'enough_for_alignments_alone':not unknown and required<=free,
            'excludes':['new pair products','temporary merged SLCs','range-2x scratch','new netCDF and fit products'],
            'processing_started':False}


def freeze(paths, directory):
    directory = Path(directory).resolve()
    with writer_lock(directory):
        ledger = {'schema':'glacier-offsets-frozen-inputs-v1','inputs':[]}
        for path in map(Path,paths):
            source = path.resolve()
            before = revision(source)
            logging.info('Verifying baseline %s',source)
            with h5netcdf.File(source) as h:
                verify_hashes(h)
                metadata = {k:str(h.attrs[k]) for k in ('geometry_identity','layout_identity','package_identity')}
                pairs = h['pair_id'][:].astype(str).tolist()
            sha = digest(source)
            destination = directory/source.name
            if destination==source:
                raise ValueError('Freeze destination must differ from source')
            if destination.exists():
                if digest(destination)!=sha:
                    raise ValueError('Frozen baseline differs; choose a new snapshot directory')
            else:
                fd,temp = tempfile.mkstemp(dir=directory,suffix='.nc')
                os.close(fd)
                try:
                    shutil.copyfile(source,temp)
                    if digest(temp)!=sha or revision(source)!=before:
                        raise ValueError('Baseline changed during checksum/copy verification')
                    os.chmod(temp,0o444)
                    os.replace(temp,destination)
                finally:
                    if os.path.exists(temp):
                        os.unlink(temp)
            if revision(source)!=before:
                raise ValueError('Baseline source changed during freezing')
            ledger['inputs'].append({'source':str(source),'frozen':str(destination),'sha256':sha,
                                     'pair_ids':pairs,'metadata':metadata})
            write_json(directory/'frozen.json',ledger)
    return ledger


def extended_config(parent,output):
    parent,output = Path(parent).resolve(),Path(output).resolve()
    if parent==output or output.exists():
        raise ValueError('Extended configuration must be a new path')
    record = {'extends':str(parent),'stack':{'start_date':'2017-01-01','end_date':'2018-12-31',
              'reference_date':'2017-01-01','swath_correction':{'enabled':False}}}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        yaml.safe_dump(record,stream,sort_keys=False)
    return {'path':str(output),'notes':'Raw processing only. Reference, grid and matching inherited; calibration remains disabled.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+')
    parser.add_argument('--freeze-directory',required=True)
    parser.add_argument('--stack-config')
    parser.add_argument('--extended-config')
    parser.add_argument('--capacity-plan',help='Optional saved extended stack plan for a disk-capacity report')
    parser.add_argument('--stack-root',help='Existing stack whose retained alignment sizes are the capacity baseline')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    if bool(args.stack_config)!=bool(args.extended_config):
        parser.error('--stack-config and --extended-config must be supplied together')
    if bool(args.capacity_plan)!=bool(args.stack_root):
        parser.error('--capacity-plan and --stack-root must be supplied together')
    result = freeze(args.inputs,args.freeze_directory)
    if args.stack_config:
        result['extended_config'] = extended_config(args.stack_config,args.extended_config)
    if args.capacity_plan:
        result['capacity']=estimate_extension_capacity(args.stack_root,args.capacity_plan)
        write_json(Path(args.freeze_directory)/'extension_capacity.json',result['capacity'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
