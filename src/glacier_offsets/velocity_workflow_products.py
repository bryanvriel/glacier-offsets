"""Name-based exports, verification and reports for pixel-inversion results.

Exports consume stored parameter names and year labels rather than fixed
coefficient positions, and take CRS labels from the inputs. The same exporter
serves version-1 prepared directories and version-2 workflow results.
"""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'glacier-offsets-matplotlib'))
import h5py
import h5netcdf
import numpy as np

from .velocity_inspect import M2_PERIOD, O1_PERIOD
from .velocity_io import digest, write_json
from .velocity_pixel_config import display_masks, parameters_from_recipe
from .velocity_pixel_general import Layout, load_field, screen, solve, predict, MODELS
from .velocity_pixel_runtime import module_fingerprints


EXPORT_MODULES = ('velocity_workflow_products', 'velocity_pixel_maps', 'velocity_pixel_plot', 'velocity_pixel_series')


class RunView:
    """Arrays and settings needed to export one result directory."""

    @classmethod
    def open(cls, out, regions=None):
        """Version-2 result directory (run manifest) or version-1 prepared directory."""
        from .velocity_workflow import is_v2_output
        return cls.v2(out, None, regions) if is_v2_output(out) else cls.legacy(out, regions)

    @classmethod
    def v2(cls, out, config=None, regions=None):
        from .velocity_workflow import load_run
        view = cls()
        view.out = Path(out)
        view.version = 2
        manifest, paths = load_run(out)
        config = manifest['config'] if config is None else config
        view.manifest = manifest
        view.obs_path, view.set_path, view.noise_path = paths['observations'], paths['solid_earth'], paths['noise']
        view.fits_path = view.out/'source_coefficients.h5'
        view.recipe = json.loads((view.out/'recipe.json').read_text())
        view.parameters = {**view.recipe['parameters'], 'display': config['display']}
        view.regions_file = Path(regions) if regions is not None else (
            None if config['diagnostics']['regions'] is None else Path(config['diagnostics']['regions']))
        with h5py.File(view.fits_path, 'r') as h:
            if h.attrs.get('status') != 'complete':
                raise ValueError('Source fits are incomplete')
            view.years = [int(y) for y in h.attrs['years']]
            layout = json.loads(h.attrs['layout_json'])
            view.layout = Layout(view.years, layout['order_years'], layout.get('order_spacecraft'))
            view.x, view.y = h['x'][:], h['y'][:]
            view.index = h['map_source_index'][:]
            view.source_id = h['source_id'][:]
            view.known_mixed = h['known_mixed_source'][:]
            view.epsg = int(h.attrs['EPSG'])
            view.crs_attrs = dict(h['crs'].attrs)
        with h5py.File(view.obs_path, 'r') as h:
            year_index = h['year_index'][:]
        view.pairs_by_year = [int(np.sum(year_index == k)) for k in range(len(view.years))]
        view.check_fit()
        return view

    @classmethod
    def legacy(cls, out, regions=None):
        from .velocity_pixel_runtime import auxiliary_path
        view = cls()
        view.out = Path(out)
        view.version = 1
        view.regions_file = auxiliary_path(out, 'regions', regions)
        view.obs_path = view.out/'observations.h5'
        view.set_path = view.out/'solid_earth_tides.h5'
        view.fits_path = view.out/'source_coefficients.h5'
        view.recipe = json.loads((view.out/'recipe.json').read_text())
        view.parameters = parameters_from_recipe(view.recipe)
        meta = json.loads((view.out/'inputs.json').read_text())
        view.years = [int(y) for y in meta.get('years', [2017, 2018])]
        view.layout = Layout(view.years)
        with h5py.File(view.obs_path, 'r') as h:
            view.x, view.y = h['x'][:], h['y'][:]
            view.index = h['map_source_index'][:]
            view.source_id = h['source_id'][:]
            view.known_mixed = h['known_mixed_source'][:] if 'known_mixed_source' in h else np.zeros(len(view.source_id), bool)
            view.epsg = int(np.asarray(h.attrs.get('EPSG', 3031)).ravel()[0])
            year_index = h['year_index'][:]
        view.pairs_by_year = [int(np.sum(year_index == k)) for k in range(len(view.years))]
        view.crs_attrs = None
        if meta.get('inputs') and Path(meta['inputs'][0]['path']).is_file():
            with h5py.File(meta['inputs'][0]['path'], 'r') as original:
                view.crs_attrs = dict(original['crs'].attrs)
        if view.crs_attrs is None:
            from pyproj import CRS
            crs = CRS.from_epsg(view.epsg)
            view.crs_attrs = {**crs.to_cf(), 'spatial_ref': crs.to_wkt()}
        view.check_fit()
        return view

    def check_fit(self):
        from .velocity_cache import identity_digest
        completion = json.loads((self.out/'fit-completion.json').read_text())
        recipe = digest(self.out/'recipe.json') if self.version == 1 else identity_digest(self.recipe)
        if recipe != completion['recipe_sha256']:
            raise ValueError('Stored fit recipe changed')
        if digest(self.fits_path) != completion['output_sha256']:
            raise ValueError('Changed source fits')

    def field(self, index):
        """Corrected observations for sorted source indices, with the run's mixed-window policy."""
        remove = self.parameters['screening']['remove_mixed_windows']
        with h5py.File(self.obs_path, 'r') as h:
            body = None if self.set_path is None else h5py.File(self.set_path, 'r')
            try:
                if self.version == 2:
                    return load_field(h, body, index, remove)
                from .velocity_pixel_inversion import get_field
                return get_field(h, body, index, remove)
            finally:
                if body is not None:
                    body.close()

    def tidal_response(self):
        """Per-source CATS response factor (NaN where unavailable)."""
        with h5py.File(self.obs_path, 'r') as h:
            if 'tidal_response_factor' in h:
                return h['tidal_response_factor'][:]
            rows, cols = h['row'][:], h['col'][:]
        meta = json.loads((self.out/'inputs.json').read_text())
        with h5py.File(meta['inputs'][0]['path'], 'r') as original:
            return original['tidal_response_factor'][:][rows, cols]

    @property
    def crs_wkt(self):
        value = self.crs_attrs.get('crs_wkt', self.crs_attrs.get('spatial_ref', f'EPSG:{self.epsg}'))
        return value.decode() if isinstance(value, bytes) else str(value)

    @property
    def axis_label(self):
        return f'EPSG:{self.epsg}'

    @property
    def period(self):
        return '+'.join(str(y) for y in self.years)

    def regions(self, role=None):
        if self.regions_file is None:
            return []
        regions = json.loads(Path(self.regions_file).read_text())['regions']
        return [r for r in regions if role is None or r.get('role') == role]


def _names(group):
    return json.loads(group.attrs['parameter_names'])


def export_maps(view):
    """Native-grid NetCDF, GeoTIFFs, figures and summary from stored source fits."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    out = view.out
    display = view.parameters['display']
    x, y, index = view.x, view.y, view.index
    footprint = index >= 0
    safe = np.maximum(index, 0)
    shape = index.shape
    map_chunks = (min(128, len(y)), min(128, len(x)))

    def mapped(value):
        v = np.asarray(value, dtype=float)[safe].copy()
        v[~footprint] = np.nan
        return v
    files = []
    with h5py.File(view.fits_path, 'r') as fits:
        joint, az = fits['residual_vertical'], fits['azimuth_direct']
        names, az_names = _names(joint), _names(az)
        col = {n: j for j, n in enumerate(names)}
        standard = fits['standard_coverage'][:]
        support = fits['fit_support'][:]
        speed = fits['mean_flow_speed'][:]
        angle = fits['year_direction_change_deg'][:]
        projection = fits['azimuth_flow_projection_by_year'][:]
        estimable, azestimable = joint['estimable'][:], az['estimable'][:]
        a, sigma = joint['amplitude'][:], joint['amplitude_sigma'][:]
        masks = display_masks(estimable, azestimable, standard, speed, angle, projection, a, sigma,
                              joint['phase_sigma_deg'][:], display)
        coef, azcoef = joint['coefficients'][:], az['coefficients'][:]
        counts = fits['clean_pairs_by_year'][:]
        fields = {
            'msf_along_C': (coef[:, col['Msf_C']], 'm day-1'), 'msf_along_S': (coef[:, col['Msf_S']], 'm day-1'),
            'msf_along_amplitude': (a, 'm day-1'), 'msf_along_phase_deg': (joint['phase_deg'][:], 'degree'),
            'msf_along_amplitude_diagnostic_se': (sigma, 'm day-1'),
            'msf_along_phase_diagnostic_se_deg': (joint['phase_sigma_deg'][:], 'degree'),
            'msf_azimuth_C': (azcoef[:, az_names.index('Msf_C')], 'm day-1'),
            'msf_azimuth_S': (azcoef[:, az_names.index('Msf_S')], 'm day-1'),
            'msf_azimuth_amplitude': (az['amplitude'][:], 'm day-1'), 'msf_azimuth_phase_deg': (az['phase_deg'][:], 'degree'),
            'msf_azimuth_amplitude_diagnostic_se': (az['amplitude_sigma'][:], 'm day-1'),
            'msf_azimuth_phase_diagnostic_se_deg': (az['phase_sigma_deg'][:], 'degree'),
            'mean_flow_speed': (speed, 'm day-1'), 'year_direction_change_deg': (angle, 'degree'),
            'flow_axis_east': (fits['flow_axis_EN'][:, 0], '1'), 'flow_axis_north': (fits['flow_axis_EN'][:, 1], '1'),
            'clean_pairs_total': (counts.sum(1), '1'),
            'span_days': (fits['span_days'][:], 'day'), 'msf_phase_bins_occupied': (fits['phase_bins'][:], '1'),
            'joint_target_condition': (joint['target_condition'][:], '1'),
            'azimuth_target_condition': (az['target_condition'][:], '1'),
            'range_residual_rms': (joint['rms_m'][:, 0], 'm'), 'azimuth_residual_rms': (joint['rms_m'][:, 1], 'm'),
            'Msf_partial_R2': (joint['partial_r2'][:], '1'), 'calendar_clusters': (joint['calendar_clusters'][:], '1')}
        for k, year in enumerate(view.years):
            fields[f'azimuth_flow_projection_{year}'] = (projection[:, k], '1')
            fields[f'clean_pairs_{year}'] = (counts[:, k], '1')
            fields[f'east_velocity_{year}'] = (coef[:, col[f'east_velocity_{year}']], 'm day-1')
            fields[f'north_velocity_{year}'] = (coef[:, col[f'north_velocity_{year}']], 'm day-1')
            if f'order_{year}' in col:
                fields[f'order_displacement_{year}'] = (coef[:, col[f'order_{year}']], 'm')
        for name in ('M2', 'O1'):
            c, s = coef[:, col[f'vertical_{name}_C']], coef[:, col[f'vertical_{name}_S']]
            fields[f'residual_vertical_{name}_C'] = (c, 'm')
            fields[f'residual_vertical_{name}_S'] = (s, 'm')
            fields[f'residual_vertical_{name}_amplitude'] = (np.hypot(c, s), 'm')
            fields[f'residual_vertical_{name}_phase_deg'] = (np.degrees(np.arctan2(s, c)), 'degree')
        mask_fields = {'fit_estimable': estimable, 'azimuth_fit_estimable': azestimable, 'fit_support_mask': support,
                       'standard_coverage_mask': standard, 'along_display_mask': masks['along'],
                       'along_flow_proxy_mask': masks['proxy'], 'phase_quality_display_mask': masks['phase_quality'],
                       'known_mixed_source_mask': view.known_mixed}
        nc = out/'coefficient_maps.nc'
        temporary = out/'coefficient_maps.partial.nc'
        with h5netcdf.File(temporary, 'w') as dst:
            dst.dimensions = {'y': len(y), 'x': len(x), 'parameter': len(names), 'azimuth_parameter': len(az_names),
                              'target': 2, 'target2': 2, 'year': len(view.years)}
            dst.attrs.update(Conventions='CF-1.8', status='complete', EPSG=view.epsg, years=list(view.years),
                             title=f'Exploratory {view.period} local Msf and residual vertical tide coefficients',
                             phase_origin='2000-01-01T00:00:00Z',
                             phase_convention='v=C cos(omega*tau)+S sin(omega*tau); phase=atan2(S,C) degrees',
                             horizontal_Msf_period_days=14.765294, vertical_M2_period_days=M2_PERIOD,
                             vertical_O1_period_days=O1_PERIOD, order_sign=json.dumps({str(y): view.layout.order_sign(y) for y in view.layout.order_years}, sort_keys=True),
                             secular_vertical_velocity_m_day=0.,
                             vertical_basis='height endpoint differences, M2 and O1; coefficients in metres',
                             uncertainty=view.recipe['diagnostic_uncertainty'],
                             display_settings_json=json.dumps(display, sort_keys=True),
                             spatial_fit='independent unique radar sources, copied native cells share source fits; no spatial smoothing',
                             recipe_json=json.dumps(view.recipe, sort_keys=True))
            for name, v, dimension in [('x', x, 'x'), ('y', y, 'y')]:
                variable = dst.create_variable(name, (dimension,), dtype='f8')
                variable[:] = v
                variable.attrs.update(units='m', axis=name.upper(),
                                      standard_name='projection_x_coordinate' if name == 'x' else 'projection_y_coordinate')
            variable = dst.create_variable('year', ('year',), dtype='i4')
            variable[:] = view.years
            crs = dst.create_variable('crs', (), dtype='i4')
            crs[()] = 0
            crs.attrs.update({k: v for k, v in view.crs_attrs.items()})
            for name, (value, unit) in fields.items():
                variable = dst.create_variable(name, ('y', 'x'), dtype='f4', fillvalue=np.nan, compression='gzip',
                                               compression_opts=2, chunks=map_chunks)
                variable[:] = mapped(value)
                variable.attrs.update(units=unit, grid_mapping='crs')
            for name, value in mask_fields.items():
                variable = dst.create_variable(name, ('y', 'x'), dtype='u1', compression='gzip', compression_opts=2)
                variable[:] = np.where(footprint, value[safe], False).astype('u1')
                variable.attrs.update(units='1', grid_mapping='crs')
            variable = dst.create_variable('source_observation_index', ('y', 'x'), dtype='i8', fillvalue=-1)
            variable[:] = np.where(footprint, view.source_id[safe], -1)
            for group, dimension in [('residual_vertical', 'parameter'), ('azimuth_direct', 'azimuth_parameter')]:
                group_names = _names(fits[group])
                v = dst.create_variable(group+'_parameter_name', (dimension,), dtype=h5py.string_dtype())
                v[:] = group_names
                unit_var = dst.create_variable(group+'_parameter_units', (dimension,), dtype=h5py.string_dtype())
                unit_var[:] = ['m' if n.startswith(('order_', 'vertical_')) else 'm day-1' for n in group_names]
                v = dst.create_variable(group+'_coefficients', (dimension, 'y', 'x'), dtype='f8', fillvalue=np.nan,
                                        compression='gzip', compression_opts=2, chunks=(1, *map_chunks))
                v[:] = np.moveaxis(mapped(fits[group]['coefficients'][:]), -1, 0)
                v = dst.create_variable(group+'_Msf_covariance', ('target', 'target2', 'y', 'x'), dtype='f8',
                                        fillvalue=np.nan, compression='gzip', compression_opts=2, chunks=(1, 1, *map_chunks))
                v[:] = np.moveaxis(mapped(fits[group]['coefficient_covariance'][:]), (2, 3), (0, 1))
        temporary.replace(nc)
        files.append(nc.name)
        # Explicit center-to-edge transform, checked by rasterio readback.
        dx, dy = x[1]-x[0], y[1]-y[0]
        transform = from_origin(x[0]-dx/2, y[0]-dy/2, dx, -dy)
        tiff_crs = CRS.from_wkt(view.crs_wkt) if not view.crs_wkt.startswith('EPSG:') else CRS.from_epsg(view.epsg)
        exports = {'msf_along_amplitude': masks['along'], 'msf_along_phase_deg': masks['along'],
                   'msf_azimuth_amplitude': masks['azimuth'], 'msf_azimuth_phase_deg': masks['azimuth'],
                   'msf_along_phase_diagnostic_se_deg': masks['along'], 'mean_flow_speed': standard,
                   'residual_vertical_M2_amplitude': masks['along'], 'residual_vertical_M2_phase_deg': masks['along'],
                   'residual_vertical_O1_amplitude': masks['along'], 'residual_vertical_O1_phase_deg': masks['along'],
                   'range_residual_rms': estimable, 'azimuth_residual_rms': estimable,
                   'clean_pairs_total': np.ones_like(estimable)}
        tiffs = []
        for name, mask in exports.items():
            image = mapped(np.where(mask, fields[name][0], np.nan)).astype('f4')
            path = out/(name+'.tif')
            with rasterio.open(path, 'w', driver='GTiff', height=len(y), width=len(x), count=1, dtype='float32',
                               crs=tiff_crs, transform=transform, nodata=np.nan, compress='deflate', tiled=True) as dst:
                dst.write(image, 1)
                dst.update_tags(units=fields[name][1], mask='see coefficient_maps.nc and report.md', phase_origin='2000-01-01 UTC')
            with rasterio.open(path) as check:
                if not np.allclose(check.xy(0, 0), (x[0], y[0])):
                    raise ValueError('GeoTIFF pixel-center shift')
                if not np.array_equal(check.read(1), image, equal_nan=True):
                    raise ValueError('GeoTIFF values changed')
                if check.crs.to_epsg() not in (None, view.epsg):
                    raise ValueError('GeoTIFF CRS differs from the inputs')
            tiffs.append(path.name)
        files += tiffs
        try:
            import iceutils as ice
            check = ice.Raster(rasterfile=str(out/'msf_along_amplitude.tif'))
            if (check.hdr.epsg != view.epsg or not np.allclose(check.hdr.xcoords+check.hdr.dx/2, x)
                    or not np.allclose(check.hdr.ycoords+check.hdr.dy/2, y)):
                raise ValueError('iceutils GeoTIFF coordinate mismatch')
            ice_check = 'default corner grid converted to pixel centers with dx/2,dy/2'
        except ImportError:
            ice_check = 'not run; iceutils unavailable'
        extent = [(x[0]-dx/2)/1000, (x[-1]+dx/2)/1000, (y[-1]+dy/2)/1000, (y[0]-dy/2)/1000]
        regions = view.regions()
        label = view.axis_label

        def panel(ax, values, mask, title, cmap, limits, unit):
            ax.imshow(np.where(footprint, 1., np.nan), extent=extent, origin='upper', cmap='gray', vmin=0, vmax=1.15)
            im = ax.imshow(mapped(np.where(mask, values, np.nan)), extent=extent, origin='upper', cmap=cmap,
                           vmin=limits[0], vmax=limits[1], interpolation='nearest')
            ax.set(title=title, xlabel=f'{label} x (km)', ylabel=f'{label} y (km)')
            ax.set_aspect('equal')
            for r in regions:
                w, s, e, n = np.array(r['bounds'])/1000
                ax.add_patch(Rectangle((w, s), e-w, n-s, fill=False, edgecolor='k', lw=.6))
                ax.text(w, n, r['id'].replace('target_', 'T').replace('control_', 'C'), fontsize=6, va='bottom', color='k',
                        clip_on=True, bbox=dict(facecolor='white', alpha=.6, edgecolor='none', pad=.2))
            plt.colorbar(im, ax=ax, shrink=.85, label=unit)
            return im

        def save(fig, stem):
            for extension in ('png', 'pdf'):
                fig.savefig(out/f'{stem}.{extension}', dpi=180)
                files.append(f'{stem}.{extension}')
            plt.close(fig)
        phase = joint['phase_deg'][:]
        fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), layout='constrained')
        panel(axes[0, 0], a, masks['along'], 'Along-flow Msf amplitude: coverage + moving ice', 'viridis', (0, .2), 'm/day')
        panel(axes[0, 1], phase, masks['along'], 'Along-flow Msf phase: same support', 'twilight_shifted', (-180, 180), 'degrees from 2000-01-01')
        panel(axes[1, 0], a, masks['phase_quality'], 'Amplitude: additional phase/direction diagnostic screen', 'viridis', (0, .2), 'm/day')
        panel(axes[1, 1], phase, masks['phase_quality'],
              f"Phase: diagnostic SE ≤{display['max_phase_sigma_deg']:g}°, amplitude/SE ≥{display['min_amplitude_over_sigma']:g}",
              'twilight_shifted', (-180, 180), 'degrees from 2000-01-01')
        fig.suptitle(f'{view.period} independent-source Msf + residual vertical M2/O1\n'
                     'Annual secular motion and spacecraft order; no spatial smoothing', fontsize=13)
        save(fig, 'msf_amplitude_phase')
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
        panel(axes[0], a, masks['along'], 'Along-flow Msf velocity amplitude', 'viridis', (0, .2), 'm/day')
        panel(axes[1], phase, masks['along'], 'Along-flow Msf velocity phase', 'twilight_shifted', (-180, 180), 'degrees from 2000-01-01')
        fig.suptitle(f'{view.period} pointwise inversion: residual vertical M2/O1 + annual secular/order terms', fontsize=12)
        save(fig, 'msf_quicklook')
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
        panel(axes[0], az['amplitude'][:], masks['azimuth_moving'], 'Direct satellite-azimuth Msf amplitude', 'viridis', (0, .15), 'm/day in satellite azimuth')
        panel(axes[1], az['phase_deg'][:], masks['azimuth_moving'], 'Direct satellite-azimuth Msf phase', 'twilight_shifted', (-180, 180), 'degrees from 2000-01-01')
        fig.suptitle('Direction-independent comparator; phase flips ~180° when flow projects negatively into azimuth', fontsize=11)
        save(fig, 'azimuth_amplitude_phase')
        diagnostic = [(f'clean_pairs_{year}', f'{year} clean pairs', 'viridis', (0, max(60, view.pairs_by_year[k])), 'pairs')
                      for k, year in enumerate(view.years)]
        diagnostic += [('mean_flow_speed', 'Mean robust secular speed', 'viridis', (0, 1.2), 'm/day'),
                       ('year_direction_change_deg', 'Maximum annual secular direction change', 'magma', (0, 45), 'degrees'),
                       (f'azimuth_flow_projection_{view.years[0]}', 'Azimuth projection of flow direction', 'coolwarm', (-1, 1), 'dimensionless'),
                       ('msf_along_phase_diagnostic_se_deg', 'Calendar-cluster phase diagnostic SE', 'magma', (0, 90), 'degrees')]
        columns = 3
        rows = int(np.ceil(len(diagnostic)/columns))
        fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5*rows), layout='constrained', squeeze=False)
        for ax in axes.ravel()[len(diagnostic):]:
            ax.axis('off')
        for ax, (name, title, cmap, limits, unit) in zip(axes.ravel(), diagnostic):
            panel(ax, fields[name][0], standard, title, cmap, limits, unit)
        fig.suptitle('Support, flow geometry and conditional uncertainty diagnostics', fontsize=13)
        save(fig, 'fit_diagnostics')
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
        residual_panels = [('range_residual_rms', 'Range residual RMS', 'magma', (0, .6), 'm'),
                           ('azimuth_residual_rms', 'Azimuth residual RMS', 'magma', (0, 1.5), 'm'),
                           ('Msf_partial_R2', 'Msf partial R²', 'viridis', (0, .2), 'dimensionless'),
                           ('clean_pairs_total', 'Retained pairs, all years', 'viridis', (0, sum(view.pairs_by_year)), 'pairs'),
                           ('span_days', 'Retained endpoint span', 'viridis', (0, 365.25*len(view.years)), 'days'),
                           ('msf_phase_bins_occupied', 'Occupied Msf phase bins', 'viridis',
                            (0, view.parameters['coverage']['phase_bins']), 'bins')]
        for ax, (name, title, cmap, limits, unit) in zip(axes.ravel(), residual_panels):
            panel(ax, fields[name][0], estimable if 'rms' in name or 'R2' in name else np.ones_like(estimable),
                  title, cmap, limits, unit)
        fig.suptitle('Residual and coverage diagnostics (residuals: estimable fits; coverage: all sources)', fontsize=13)
        save(fig, 'residual_coverage_diagnostics')
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
        for row, name in enumerate(('M2', 'O1')):
            panel(axes[row, 0], fields[f'residual_vertical_{name}_amplitude'][0], masks['along'],
                  f'Residual vertical {name} height amplitude', 'viridis', (0, .3), 'metres')
            panel(axes[row, 1], fields[f'residual_vertical_{name}_phase_deg'][0], masks['along'],
                  f'Residual vertical {name} height phase', 'twilight_shifted', (-180, 180), 'degrees from 2000-01-01')
        fig.suptitle('Residual height coefficients after CATS and full ENU solid-Earth correction\n'
                     'Descriptive nuisance estimates, not total ocean-tide amplitudes', fontsize=12)
        save(fig, 'residual_vertical_amplitude_phase')

        def count(mask):
            return {'unique_sources': int(mask.sum()), 'map_cells': int(np.sum(footprint & mask[safe]))}

        def percentiles(value):
            return np.nanpercentile(value, [10, 25, 50, 75, 90, 95, 99]) if np.isfinite(value).any() else np.full(7, np.nan)

        def median(value):
            return float(np.nanmedian(value)) if np.isfinite(value).any() else None
        along = masks['along']
        summary = {'unique_sources': len(a), 'native_grid_shape': list(shape), 'years': view.years,
                   'pairs_by_year': view.pairs_by_year, 'EPSG': view.epsg,
                   'source_coverage': count(standard & estimable), 'fit_support': count(support),
                   'along_display': count(along), 'phase_quality_display': count(masks['phase_quality']),
                   'fast_stable_flow_proxy': count(masks['proxy']),
                   'along_amplitude_percentiles_m_day': percentiles(a[along]),
                   'azimuth_amplitude_percentiles_m_day': percentiles(az['amplitude'][:][masks['azimuth_moving']]),
                   'median_phase_diagnostic_se_deg': median(joint['phase_sigma_deg'][:][along]),
                   'median_range_rms_m': median(joint['rms_m'][:][along, 0]),
                   'median_azimuth_rms_m': median(joint['rms_m'][:][along, 1]),
                   'GeoTIFFs': tiffs, 'netCDF_sha256': digest(nc), 'GeoTIFF_center_and_data_roundtrip': True,
                   'iceutils_georeference_roundtrip': ice_check, 'parameters': view.parameters,
                   'regions_sha256': None if view.regions_file is None else digest(view.regions_file),
                   'mask_definitions': {
                       'along_display': f"standard coverage, estimable, mean speed>={display['min_speed_m_day']:g}m/day, "
                                        f"minimum annual absolute azimuth-flow projection>={display['min_abs_azimuth_projection']:g}",
                       'phase_quality_display': f"along display plus direction change<={display['max_direction_change_deg']:g}deg, "
                                                f"min projection>={display['diagnostic_min_abs_azimuth_projection']:g}, cluster diagnostic "
                                                f"phase SE<={display['max_phase_sigma_deg']:g}deg, amplitude/SE>={display['min_amplitude_over_sigma']:g}",
                       'fast_stable_flow_proxy': f"along display plus speed>={display['proxy_min_speed_m_day']:g}m/day, direction change"
                                                 f"<={display['max_direction_change_deg']:g}deg, min projection>={display['diagnostic_min_abs_azimuth_projection']:g}"},
                   'uncertainty_warning': 'diagnostic pixelwise cluster covariance, no FDR/significance or calibrated phase-confidence claim'}
    write_json(out/'map-summary.json', summary)
    files.append('map-summary.json')
    logging.info('Exported maps: %s', json.dumps({k: summary[k] for k in ('source_coverage', 'along_display', 'phase_quality_display')}))
    return files


def verify_v2(view):
    """Replay sampled fits and check physical prediction, mapping and CRS."""
    recipe = view.recipe
    from .velocity_workflow import FIT_MODULES
    current = module_fingerprints(FIT_MODULES)
    if current != recipe['code_sha256']:
        changed = sorted(k for k in current if current[k] != recipe['code_sha256'].get(k))
        raise ValueError(f'Numerical implementation changed since fitting: {changed}')
    for item in view.manifest['collection']['inputs']:
        if digest(item['path']) != item['sha256']:
            raise ValueError(f'Input package changed: {item["path"]}')
    parameters = recipe['parameters']
    rng = np.random.default_rng(12345)
    with h5py.File(view.obs_path, 'r') as h, h5py.File(view.fits_path, 'r') as fits:
        body = None if view.set_path is None else h5py.File(view.set_path, 'r')
        try:
            eligible = np.flatnonzero(fits['residual_vertical/estimable'][:])
            if not len(eligible):
                raise ValueError('No estimable sources to verify')
            selected = np.sort(rng.choice(eligible, min(512, len(eligible)), replace=False))
            covariances = fits['covariance_by_year'][:]
            f = load_field(h, body, selected, parameters['screening']['remove_mixed_windows'])
            good, axis, _, _, _ = screen(f, view.layout, parameters['screening'])
            stored_axis = fits['flow_axis_EN'][selected]
            if not np.allclose(axis, stored_axis, rtol=0, atol=1e-12):
                raise ValueError('Stored flow axes do not reproduce')
            errors = {}
            for model in MODELS:
                original = fits[model]['coefficients'][selected]
                repeated = solve(f, view.layout, good, axis, covariances, model,
                                 cluster_days=parameters['uncertainty']['cluster_days'])
                if list(repeated['names']) != _names(fits[model]):
                    raise ValueError(f'{model} parameter names changed')
                errors[model] = float(np.nanmax(abs(repeated['coefficients']-original)))
                if errors[model] > 1e-9:
                    raise ValueError(f'{model} source fits do not reproduce ({errors[model]:.3g})')
            names = _names(fits['residual_vertical'])
            original = fits['residual_vertical/coefficients'][selected]
            residual = np.where(good[..., None], np.nan_to_num(f['data']-predict(f, view.layout, original, names, stored_axis)), 0.)
            rms = np.sqrt(np.sum(residual**2, axis=1)/good.sum(1)[:, None])
            rms_error = float(np.max(abs(rms-fits['residual_vertical/rms_m'][selected])))
            if rms_error > 1e-6:
                raise ValueError('Standalone radar prediction/RMS does not reproduce')
            with h5netcdf.File(view.out/'coefficient_maps.nc', 'r') as maps:
                rr, cc = fits['row'][selected], fits['col'][selected]
                values = maps['residual_vertical_coefficients'][:][:, rr, cc].T
                if not np.array_equal(values, original, equal_nan=True):
                    raise ValueError('Mapped source coefficients changed')
                back = view.index
                expected = fits['residual_vertical/amplitude'][:][np.maximum(back, 0)]
                if not np.array_equal(maps['msf_along_amplitude'][:][back >= 0], expected[back >= 0], equal_nan=True):
                    raise ValueError('Copied map-cell values differ from source fits')
                if int(maps.attrs['EPSG']) != view.epsg or maps['crs'].attrs.get('crs_wkt') != view.crs_attrs.get('crs_wkt'):
                    raise ValueError('Exported CRS differs from the inputs')
                names_nc = [v.decode() if isinstance(v, bytes) else str(v) for v in maps['residual_vertical_parameter_name'][:]]
                if names_nc != names:
                    raise ValueError('Exported parameter names differ from the source fits')
        finally:
            if body is not None:
                body.close()
    record = {'status': 'passed', 'inputs_unchanged': True, 'cache_artifacts_verified': True,
              'sampled_sources': len(selected), 'coefficient_replay_max_error': errors,
              'independent_prediction_rms_max_error_m': rms_error, 'native_copy_mapping_exact': True,
              'crs_from_inputs': True, 'parameters': parameters, 'years': view.years,
              'regional_year_check': 'not run; version-2 results have no frozen regional cohort',
              'verification_code_sha256': module_fingerprints(('velocity_workflow_products',)),
              'checked_utc': datetime.now(timezone.utc).isoformat()}
    write_json(view.out/'verification.json', record)
    logging.info('Verification passed: coefficient replay %s, RMS %.3g m', errors, rms_error)
    return record


def _table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
                     + ['| '+' | '.join(map(str, r))+' |' for r in rows])


def report_v2(view, series=None):
    out = view.out
    summary = json.loads((out/'map-summary.json').read_text())
    noise = json.loads(Path(view.noise_path).read_text())
    verification = json.loads((out/'verification.json').read_text())
    manifest = view.manifest
    recipe = view.recipe
    noise_rows = [[y, f"{d['sigma_range_m']:.4f}", f"{d['sigma_azimuth_m']:.4f}", f"{d['correlation']:.3f}",
                   next((c['samples'] for c in noise.get('calibration', []) if c['year'] == int(y)), 'supplied')]
                  for y, d in noise['diagnostics'].items()]
    solid = manifest['artifacts']['solid_earth']
    lines = [f'# {view.period} pixel inversion (version-2 workflow)', '',
             '![Along-flow Msf amplitude and phase](msf_quicklook.png)', '',
             recipe['model']+'.', recipe['comparator']+'.', '',
             'These are exploratory coefficients. Pixelwise uncertainty and display filters are diagnostics, not calibrated detections.', '',
             '## Inputs and corrections', '',
             _table(['Input', 'SHA-256'], [[i['path'], i['sha256'][:16]+'…'] for i in manifest['collection']['inputs']]), '',
             f"Pairs by calendar-midpoint year: `{json.dumps(manifest['collection']['pairs_by_year'])}`; "
             f"cross-year pairs: {len(manifest['collection']['cross_year_pairs'])}.", '',
             'Spacecraft order offsets: '+(', '.join(f'{y}: {view.layout.order_sign(y).split(";")[0]}' for y in view.layout.order_years)
                                           or 'none (single-spacecraft years)')+'; years not listed have no order term.', '',
             'Ocean tides: existing CATS correction required and verified by provenance. Solid Earth: '
             + (f"PySolid full-ENU correction artifact `{solid['id']}`." if 'id' in solid else 'not applied.'), '',
             f"Observation cache `{manifest['artifacts']['observations']['id']}`; noise `{manifest['artifacts']['noise']['id']}`.", '',
             '## Annual noise', '',
             f"Mode: {noise['mode']}.", '',
             _table(['Year', 'σ range (m)', 'σ azimuth (m)', 'Correlation', 'Samples'], noise_rows), '',
             '## Effective settings', '',
             'Screen: '+recipe['screen']+'.', '',
             'Remove mixed windows: '+str(recipe['parameters']['screening']['remove_mixed_windows']).lower()+'.', '',
             'Fit support: '+recipe['min_fit']+'.', '',
             'Standard map coverage: '+recipe['standard_coverage']+'.', '',
             'Uncertainty: '+recipe['diagnostic_uncertainty']+'.', '',
             'Display filters: `'+json.dumps(view.parameters['display'], sort_keys=True)+'`.', '',
             '## Results', '',
             _table(['Selection', 'Unique sources', 'Native cells'],
                    [[name, summary[name]['unique_sources'], summary[name]['map_cells']]
                     for name in ('fit_support', 'source_coverage', 'along_display', 'phase_quality_display', 'fast_stable_flow_proxy')]), '',
             'Amplitude percentiles (10, 25, 50, 75, 90, 95, 99), m/day: `'
             + json.dumps(summary['along_amplitude_percentiles_m_day'])+'`.', '',
             '## Verification', '',
             f"Coefficient replay: `{json.dumps(verification['coefficient_replay_max_error'])}`; independent radar RMS "
             f"difference {verification['independent_prediction_rms_max_error_m']:.3g} m; native mapping exact; CRS from inputs.", '',
             ('Point time series: '+', '.join(p['label'] for p in series['points'])+' in `point_timeseries/`.'
              if series else 'Point time series: not produced (no reviewed target regions supplied).'), '',
             '## Reproduction', '', '```bash',
             f'glacier-offsets velocity-pixels run "{manifest["config_file"]}"', '```', '',
             'Repeated runs reuse verified cache artifacts and result stages; changed settings rebuild only dependent stages.', '']
    (out/'report.md').write_text('\n'.join(lines))
    return ['report.md']


def products(out, config, run_manifest):
    """Export, maps, optional point series, verification and report stages."""
    from .velocity_workflow import stage_current, record_stage, invalidate
    out = Path(out)
    view = RunView.v2(out, config)
    regions_sha = None if view.regions_file is None else digest(view.regions_file)
    base = {'fit_sha256': digest(view.fits_path), 'display': config['display'], 'regions_sha256': regions_sha,
            'code_sha256': module_fingerprints(EXPORT_MODULES)}
    if not stage_current(out, 'export', base):
        invalidate(out, ('export', 'maps', 'series', 'verify', 'report'))
        record_stage(out, 'export', base, export_maps(view))
    if not stage_current(out, 'maps', base):
        from .velocity_pixel_maps import plot_maps
        records = plot_maps(out, component='all', regions=view.regions_file)
        files = [str(Path(f).relative_to(out)) for r in records for f in r['files']]
        record_stage(out, 'maps', base, files+['amplitude_phase_maps/plot-settings.json'])
    series = None
    targets = [r for r in view.regions('target') if r.get('reviewed') is True]
    if targets:
        if not stage_current(out, 'series', base):
            from .velocity_pixel_plot import run_view
            files = run_view(view, targets)
            record_stage(out, 'series', base, files)
        series = json.loads((out/'point_timeseries/points.json').read_text())
    else:
        invalidate(out, ('series',))
    verify_identity = {**base, 'export': json.loads((out/'stages/export.json').read_text())['identity_sha256']}
    if not stage_current(out, 'verify', verify_identity):
        verify_v2(view)
        record_stage(out, 'verify', verify_identity, ['verification.json'])
    report_identity = {**verify_identity, 'series': series is not None}
    if not stage_current(out, 'report', report_identity):
        record_stage(out, 'report', report_identity, report_v2(view, series))
    return view
