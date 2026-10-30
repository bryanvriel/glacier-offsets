"""Joint EN inversion, empirical covariance, robust weighting and information."""
import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import chi2
from .velocity_basis import design

WEIGHT_MODELS = ('constant', 'constant_robust', 'quality_robust', 'quality_rule')


def quality_rule(snr, peak):
    return ((peak >= .3) & (snr >= 8)) | (snr >= 20)


def _features(snr, peak, with_peak):
    columns = [np.ones_like(snr), np.log(np.maximum(snr, 1e-8))]
    if with_peak:
        columns.append(np.log(np.maximum(peak, 1e-8)))
    return np.stack(columns, -1)


def calibrate_noise(residual_ra, snr, peak, groups, *, minimum=40):
    """Estimate balanced RA covariance and monotone log-linear variance models.

    Groups are spatial blocks/controls; each contributes equal total weight.
    Residuals must be held-out secular predictions on reviewed stable controls.
    """
    residual_ra = np.asarray(residual_ra, float)
    good = np.isfinite(residual_ra).all(-1)
    if good.sum() < minimum:
        raise ValueError('Insufficient held-out control residuals for noise calibration')
    residual_ra, snr, peak, groups = (np.asarray(v)[good] for v in (residual_ra, snr, peak, groups))
    _, ids, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weight = 1/counts[ids]
    weight /= weight.sum()
    center = np.array([np.median(residual_ra[:, j]) for j in range(2)])
    r = residual_ra-center
    # Winsorization prevents a few failed matches defining the baseline noise.
    scale = np.maximum(1.4826*np.median(abs(r), axis=0), 1e-4)
    clipped = np.clip(r, -4*scale, 4*scale)
    covariance = (clipped.T*weight) @ clipped
    sigma = np.maximum(np.sqrt(np.diag(covariance)), 1e-4)
    quality = np.where(np.isfinite(peak) & (peak>0), snr*peak, snr)
    finite_quality = np.isfinite(quality) & (quality>0)
    high = finite_quality & (quality >= np.quantile(quality[finite_quality], .9)) if finite_quality.any() else np.ones(len(r),bool)
    # The observed high-quality residual scatter is the empirical noise floor;
    # a 0.1 mm numerical floor only protects exact synthetic/control zeros.
    high_weight = weight[high]/weight[high].sum()
    floors = np.maximum(np.sqrt(np.sum(clipped[high]**2*high_weight[:,None],axis=0)),1e-4)
    correlation = float(np.clip(covariance[0, 1]/np.prod(sigma), -.95, .95))
    baseline = np.diag(sigma) @ np.array([[1., correlation], [correlation, 1.]]) @ np.diag(sigma)
    models = {}
    for name, with_peak in [('snr', False), ('snr_peak', True)]:
        take = np.isfinite(snr) & (snr > 0)
        if with_peak:
            take &= np.isfinite(peak) & (peak > 0)
        if take.sum() < minimum:
            continue
        x = _features(snr[take], peak[take], with_peak)
        w = np.sqrt(weight[take]/weight[take].mean())
        coeff = []
        for axis in range(2):
            # E log(Z²) for a standard Gaussian is -gamma-log(2).
            y = np.log(np.maximum(r[take, axis]**2, floors[axis]**2))+1.270362845
            fit = lsq_linear(x*w[:, None], y*w, bounds=(np.full(x.shape[1], -np.inf), [np.inf]+[0.]*(x.shape[1]-1)))
            coeff.append(fit.x.tolist())
        models[name] = {'coefficients': coeff, 'domain': [np.min(x[:, 1:], axis=0).tolist(), np.max(x[:, 1:], axis=0).tolist()]}
    return {'covariance_ra_m2': baseline.tolist(), 'floors_m': floors.tolist(), 'correlation': correlation,
            'floor_method': 'spatially balanced residual RMS in the highest quality decile',
            'quality_models': models, 'control_bias_m': center.tolist(), 'samples': len(r),
            'spatial_groups': len(counts), 'weight_convention': 'inverse standard deviation; sqrt Huber factor',
            'uncertainty_scope': 'empirical residual noise; correction sensitivity remains separate'}


def covariance_ra(noise, snr, peak, quality=False):
    baseline = np.asarray(noise['covariance_ra_m2'])
    if baseline.shape != (2, 2) or not np.isfinite(baseline).all() or np.linalg.eigvalsh(baseline).min() <= 0:
        raise ValueError('Control covariance must be finite positive definite')
    result = np.broadcast_to(baseline, (len(snr), 2, 2)).copy()
    if not quality:
        return result
    floors = np.maximum(noise['floors_m'], np.sqrt(np.diag(baseline))/10)
    for i in range(len(snr)):
        name = 'snr_peak' if np.isfinite(peak[i]) and peak[i] > 0 else 'snr'
        model = noise['quality_models'].get(name)
        if model is None or not np.isfinite(snr[i]) or snr[i] <= 0:
            continue
        x = _features(np.array([snr[i]]), np.array([peak[i]]), name=='snr_peak')[0]
        lo, hi = np.asarray(model['domain'])
        if ((x[1:] < lo) | (x[1:] > hi)).any():
            continue
        sig = np.maximum(np.exp(np.clip(np.asarray(model['coefficients']) @ x/2, -50, 50)), floors)
        corr = noise['correlation']
        candidate = np.diag(sig) @ np.array([[1., corr], [corr, 1.]]) @ np.diag(sig)
        # Bound precision in every direction, including correlated combinations.
        minimum = np.linalg.eigvalsh(np.linalg.solve(np.linalg.cholesky(baseline),
                    np.linalg.solve(np.linalg.cholesky(baseline), candidate).T).T).min()
        if minimum < .01:
            candidate *= .01/max(minimum, 1e-12)
        result[i] = candidate
    return result


def whiten(matrix, data, covariance):
    """Whiten each two-component pair by its full EN Cholesky factor."""
    factor = np.linalg.cholesky(covariance)
    return np.linalg.solve(factor, matrix), np.linalg.solve(factor, data[..., None])[..., 0]


def _solve(a, y, penalty_indices, penalty, l1):
    scale = np.sqrt(np.mean(a*a, axis=0))
    scale[scale < 1e-12] = 1.
    x = a/scale
    regularizer = np.zeros((len(penalty_indices), x.shape[1]))
    regularizer[np.arange(len(penalty_indices)), penalty_indices] = np.sqrt(penalty)
    beta = np.linalg.lstsq(np.vstack((x, regularizer)), np.r_[y, np.zeros(len(regularizer))], rcond=1e-10)[0]
    if l1 and len(penalty_indices):
        try:
            from cvxopt import matrix as cvx_matrix, solvers
        except ImportError:
            solvers = None
        if solvers is not None and penalty>0:
            from itertools import product
            # Eliminate the unpenalized secular/Msf terms analytically. The
            # convex solver then sees spline columns only, with exactly the
            # same 0.5*||r||² + penalty*||beta_spline||_1 objective.
            unpenalized=np.setdiff1d(np.arange(x.shape[1]),penalty_indices)
            nuisance=x[:,unpenalized]
            q,s,_=np.linalg.svd(nuisance,full_matrices=False)
            q=q[:,s>s[0]*1e-10]
            z=x[:,penalty_indices]
            z=np.asfortranarray(z-q @ (q.T @ z))
            response=y-q @ (q.T @ y)
            n=z.shape[1]
            hessian=np.zeros((2*n,2*n));hessian[:n,:n]=z.T @ z
            constraints=np.block([[np.eye(n),-np.eye(n)],[-np.eye(n),-np.eye(n)]])
            sparse=np.zeros(n) if np.max(abs(z.T @ response))<=penalty else None
            for response_scale,jitter in product(
                    (max(float(np.std(response)),1e-12),1.,max(float(np.linalg.norm(response)),1e-12)),
                    (0.,1e-12,1e-10,1e-8)):
                if sparse is not None:
                    break
                # Rescaling response, coefficients and lambda together leaves
                # the objective unchanged and improves the redundant QP's
                # numerical conditioning for large held-out control residuals.
                linear=np.r_[-z.T @ (response/response_scale),np.full(n,penalty/response_scale)]
                stabilized=hessian.copy()
                stabilized[np.arange(n),np.arange(n)]+=jitter
                try:
                    solution=solvers.qp(cvx_matrix(stabilized),cvx_matrix(linear),cvx_matrix(constraints),cvx_matrix(np.zeros(2*n)),
                        options={'show_progress':False,'abstol':1e-11,'reltol':1e-11,'feastol':1e-10,'maxiters':200})
                except (ValueError,ArithmeticError):
                    continue
                candidate=np.asarray(solution['x']).ravel()[:n]*response_scale
                residual=response-z @ candidate
                primal=.5*residual @ residual+penalty*np.sum(abs(candidate))
                maximum=np.max(abs(z.T @ residual))
                dual=residual*min(1.,penalty/max(maximum,1e-30))
                gap=primal-(response @ dual-.5*dual @ dual)
                # Any numerical stabilization is accepted only if the original
                # pure L1 objective passes the same dual certificate.
                if np.isfinite(gap) and gap<=1e-8*max(1.,primal):
                    sparse=candidate
            # Redundant spline dictionaries can produce an 'unknown' QP status
            # near convergence. A feasible dual certificate bounds the actual
            # convex objective error without relying on that status alone.
            if sparse is None:
                # The equivalent dual has an identity Hessian, avoiding the
                # singular spline Gram matrix altogether. Its inequality
                # multipliers recover the primal spline coefficients.
                dual_constraints=np.concatenate((z.T,-z.T),axis=0)
                for response_scale in (max(float(np.std(response)),1e-12),1.):
                    try:
                        solution=solvers.qp(cvx_matrix(np.eye(len(response))),cvx_matrix(-response/response_scale),
                            cvx_matrix(dual_constraints),cvx_matrix(np.full(2*n,penalty/response_scale)),
                            options={'show_progress':False,'abstol':1e-11,'reltol':1e-11,'feastol':1e-10,'maxiters':200})
                    except (ValueError,ArithmeticError):
                        continue
                    multipliers=np.asarray(solution['z']).ravel()
                    candidate=(multipliers[:n]-multipliers[n:])*response_scale
                    residual=response-z @ candidate
                    primal=.5*residual @ residual+penalty*np.sum(abs(candidate))
                    maximum=np.max(abs(z.T @ residual))
                    dual=residual*min(1.,penalty/max(maximum,1e-30))
                    gap=primal-(response @ dual-.5*dual @ dual)
                    if np.isfinite(gap) and gap<=1e-8*max(1.,primal):
                        sparse=candidate
                        break
            if sparse is None:
                raise RuntimeError('L1 convex solve failed its duality-gap check')
            beta[penalty_indices]=sparse
            beta[unpenalized]=np.linalg.lstsq(nuisance,y-x[:,penalty_indices] @ sparse,rcond=1e-10)[0]
            return beta/scale,scale
        # One L1 weighting iteration; coordinate descent solves the convex objective.
        penalized = np.zeros(x.shape[1], bool)
        penalized[penalty_indices] = True
        norm = np.sum(x*x, axis=0)
        residual = y-x @ beta
        for _ in range(10000):
            before = beta.copy()
            for j in range(len(beta)):
                if norm[j] < 1e-15:
                    continue
                residual += x[:, j]*beta[j]
                z = x[:, j] @ residual
                beta[j] = np.sign(z)*max(abs(z)-penalty, 0)/norm[j] if penalized[j] else z/norm[j]
                residual -= x[:, j]*beta[j]
            if np.max(abs(beta-before)) < 1e-8*(1+np.max(abs(beta))):
                break
        else:
            raise RuntimeError('L1 coordinate descent did not converge')
    return beta/scale, scale


def _project_nuisance(a, target):
    nuisance = np.setdiff1d(np.arange(a.shape[1]), target)
    n = a[:, nuisance]
    if n.size:
        norms = np.linalg.norm(n,axis=0)
        n = n/np.where(norms>1e-15,norms,1.)
        u, s, _ = np.linalg.svd(n, full_matrices=False)
        u = u[:, s > (s[0]*1e-10 if len(s) else 0)]
        t = a[:, target]-u @ (u.T @ a[:, target])
    else:
        t = a[:, target]
    singular = np.linalg.svd(t, compute_uv=False)
    # Relative to original target scale, not its potentially zero residual.
    tolerance = max(np.linalg.norm(a[:, target], 2)*1e-8, 1e-10)
    return t.T @ t, int(np.sum(singular > tolerance)), singular


def fit(early, late, data, covariance, basis, parameterization='free_en', direction=None,
        penalty=0., robust=False, huber=2.45):
    """Fit simultaneously; covariance is conditional on model and empirical noise."""
    early, late, data, covariance = map(np.asarray, (early, late, data, covariance))
    good = np.isfinite(early+late) & (late > early) & np.isfinite(data).all(-1) & np.isfinite(covariance).all(axis=(-1, -2))
    early, late, data, covariance = (v[good] for v in (early, late, data, covariance))
    if len(data) < 5:
        raise ValueError('At least five valid pairs are required')
    if direction is None:
        direction = flow_direction(early, late, data, covariance)
    direction = np.asarray(direction, float)
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-12:
        raise ValueError('A nonzero local EN flow direction is required')
    direction = direction/np.linalg.norm(direction)
    matrix, names, indices = design(early, late, basis, parameterization, direction)
    aw, yw = whiten(matrix, data, covariance)
    factors = np.ones(len(data))
    for _ in range(50 if robust else 1):
        used_factors=factors.copy()
        a = (aw*np.sqrt(factors)[:, None, None]).reshape(-1, matrix.shape[-1])
        y = (yw*np.sqrt(factors)[:, None]).ravel()
        coefficients, scale = _solve(a, y, indices, penalty, basis['background']=='spline_l1')
        residual_w = yw-np.einsum('ijk,k->ij', aw, coefficients)
        distance = np.linalg.norm(residual_w, axis=1)
        updated = np.minimum(1., huber/np.maximum(distance, 1e-15))
        if not robust or np.max(abs(updated-factors)) < 1e-5:
            break
        factors = updated
    factors=used_factors
    # Rebuild with the factors actually used in the last solve.
    target = np.arange(len(coefficients)-(4 if parameterization=='free_en' else 2), len(coefficients))
    reg = np.zeros((len(indices), len(coefficients)))
    reg[np.arange(len(indices)), indices] = np.sqrt(penalty)*scale[indices]
    if basis['background']=='spline_l1':
        # Conditional active-set covariance; model selection uncertainty is resampled.
        inactive = indices[abs(coefficients[indices]*scale[indices]) < 1e-7]
        reg = np.zeros((len(inactive), len(coefficients)))
        reg[np.arange(len(inactive)), inactive] = 1e8*scale[inactive]
    aug = np.vstack((a, reg))
    # Scale before SVD; never form a badly scaled normal matrix to invert.
    u, singular, vt = np.linalg.svd(aug/scale, full_matrices=False)
    keep = singular > singular[0]*1e-10
    pinv = ((vt[keep].T/singular[keep]) @ u[:, keep].T)/scale[:, None]
    estimator = pinv[:, :len(y)]
    cm = estimator @ estimator.T
    information, rank, target_singular = _project_nuisance(aug, target)
    _, data_rank, _ = _project_nuisance(a, target)
    return {'coefficients': coefficients, 'covariance': cm, 'direction': direction, 'names': names,
            'target_indices': target, 'target_information': information, 'target_rank': rank,
            'target_data_rank': data_rank, 'target_singular_values': target_singular,
            'global_condition': float(singular[0]/singular[-1]) if singular[-1] > 0 else np.inf,
            'global_rank': int(keep.sum()), 'residual': data-np.einsum('ijk,k->ij', matrix, coefficients),
            'robust_factors': factors, 'whitened_rms': float(np.sqrt(np.mean(residual_w**2))),
            'valid_input': good, 'n_pairs': len(data), 'span_days': float(late.max()-early.min()),
            'basis': basis, 'parameterization': parameterization, 'penalty': penalty}


def flow_direction(early, late, data, covariance):
    """Robust secular EN estimate, with no tidal direction selected from amplitude."""
    dt = np.asarray(late)-np.asarray(early)
    direction = np.median(np.asarray(data)/dt[:, None], axis=0)
    for _ in range(30):
        residual = data-dt[:, None]*direction
        distance = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(covariance), residual[..., None])[..., 0], axis=1)
        weights = np.minimum(1., 2.45/np.maximum(distance, 1e-12))
        precision = np.linalg.inv(covariance)*weights[:, None, None]
        normal = np.einsum('n,nij->ij', dt**2, precision)
        direction_new = np.linalg.solve(normal, np.einsum('n,nij,nj->i', dt, precision, data))
        if np.linalg.norm(direction_new-direction) < 1e-9:
            break
        direction = direction_new
    norm = np.linalg.norm(direction_new)
    if norm < 1e-8:
        raise ValueError('Secular flow direction is undefined on this control')
    return direction_new/norm


def diagnostics(result):
    target = result['target_indices']
    coeff = result['coefficients'][target]
    cov = result['covariance'][np.ix_(target, target)]
    direction = result['direction']
    perpendicular = np.array([-direction[1], direction[0]])
    if result['parameterization']=='free_en':
        along = np.kron(direction[None, :], np.eye(2))
        cross = np.kron(perpendicular[None, :], np.eye(2))
        axes,ellipse,_ = np.linalg.svd(coeff.reshape(2, 2))
        orientation = float(np.mod(np.arctan2(axes[1,0],axes[0,0]),np.pi))
        rotation = float(np.sign(np.linalg.det(coeff.reshape(2,2))))
    else:
        along, cross, ellipse = np.eye(2), np.zeros((2, 2)), [np.linalg.norm(coeff), 0.]
        orientation = float(np.mod(np.arctan2(direction[1],direction[0]),np.pi))
        rotation = 0.
    cs = along @ coeff
    cm = along @ cov @ along.T
    amplitude = float(np.linalg.norm(cs))
    grad = cs/max(amplitude, 1e-15)
    phase_grad = np.array([-cs[1], cs[0]])/max(amplitude**2, 1e-30)
    identifiable = result['target_rank']==len(target)
    along_score = float(cs @ np.linalg.pinv(cm, rcond=1e-10) @ cs) if identifiable else 0.
    score = float(coeff @ np.linalg.pinv(cov,rcond=1e-10) @ coeff) if identifiable else 0.
    return {'along_coefficients': cs, 'along_covariance': cm, 'amplitude': amplitude,
            'phase': float(np.arctan2(cs[1], cs[0])),
            'amplitude_std': float(np.sqrt(max(grad @ cm @ grad if amplitude>1e-12 else np.trace(cm)/2, 0))),
            'phase_std': float(np.sqrt(max(phase_grad @ cm @ phase_grad, 0))) if amplitude>1e-12 else np.inf,
            'cross_amplitude': float(np.linalg.norm(cross @ coeff)), 'ellipse_major': float(ellipse[0]),
            'ellipse_minor': float(ellipse[1]), 'ellipse_orientation':orientation,'ellipse_rotation':rotation,
            'p_value': float(chi2.sf(score,len(coeff))), 'along_p_value':float(chi2.sf(along_score,2)), 'identifiable': identifiable}


def blocked_folds(early, late, block_days=60.):
    """Hold out midpoint blocks and purge all training pairs sharing acquisitions."""
    early, late = np.asarray(early), np.asarray(late)
    block = np.floor(((early+late)/2-np.min(early))/block_days).astype(int)
    for label in np.unique(block):
        test = block==label
        endpoints = np.unique(np.r_[early[test], late[test]])
        train = ~test & ~np.isin(early, endpoints) & ~np.isin(late, endpoints)
        yield train, test


def false_discovery(p_values, q=.05):
    """Benjamini-Yekutieli across both models; valid under spatial dependence."""
    p_values = np.asarray(p_values)
    good = np.isfinite(p_values)
    p = p_values[good]
    result = np.zeros(p_values.shape, bool)
    if not len(p):
        return result
    order = np.argsort(p)
    threshold = q*np.arange(1, len(p)+1)/(len(p)*np.sum(1/np.arange(1, len(p)+1)))
    passing = p[order] <= threshold
    if passing.any():
        result[good] = p <= p[order][np.where(passing)[0][-1]]
    return result
