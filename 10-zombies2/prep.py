"""Inputs of the zombie sequel that the first video did not have, all on the 3 km census grid of ../08-zombies/grid.py:

- terrain: how fast a zombie walks out of each cell, from NOAA ETOPO1 (1 arc-minute) and Tobler's hiking function
  (Tobler 1993: speed = 6 exp(-3.5 |slope + 0.05|) km/h). Time adds up, so we average pace (time per km), not speed:
  for every 1' sample the pace is averaged over all walking directions on the local slope, relative to flat ground; a
  cell's pace is the mean over its samples; a step between two cells crosses half of each, so its rate is the paper's
  mu divided by the mean pace of the two. Flat ground = the paper. A second version triples every slope, as an upper
  bound for the steepness that 1.85 km samples smooth away.
- police: 787,565 full-time sworn state and local officers (BJS, Census of State and Local Law Enforcement Agencies
  2018, NCJ 302187) for 327,167,434 people (Census Bureau, 1 July 2018), placed in every cell in that proportion.

Reads grid.npz and grid.json from ZOMBIE_GRID (default ../08-zombies/out, written by ../08-zombies/grid.py) and the
ETOPO1 subset from ZOMBIE_ETOPO (default data/etopo_conus.nc, see README.md).
Writes prep.npz and prep.json to ZOMBIE2_OUT (default out/).
"""
import json
import os
import sys

import numpy as np
from scipy.io import netcdf_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '08-zombies'))
import grid as G      # noqa: E402
import zsim           # noqa: E402
from cities import city_cells  # noqa: E402

SRC = os.getenv('ZOMBIE_GRID', os.path.join(HERE, '..', '08-zombies', 'out'))
OUT = os.getenv('ZOMBIE2_OUT', os.path.join(HERE, 'out'))
ETOPO = os.getenv('ZOMBIE_ETOPO', os.path.join(HERE, 'data', 'etopo_conus.nc'))
POLICE = 787565
POP_2018 = 327167434


def tobler_rel(s):
    """Tobler's hiking speed on a slope s (rise over run), relative to flat ground."""
    return np.exp(-3.5 * (np.abs(s + 0.05) - 0.05))


def pace_factor(slope_mag):
    """Mean relative hiking pace (time per distance) over all directions on a slope of this steepness."""
    th = (np.arange(72) + 0.5) * (2 * np.pi / 72)
    out = np.empty_like(slope_mag, dtype=np.float64)
    step = 200000
    flat = slope_mag.ravel()
    o = out.ravel()
    for i in range(0, len(flat), step):
        s = flat[i:i + step, None] * np.cos(th)[None, :]
        o[i:i + step] = (1.0 / tobler_rel(s)).mean(axis=1)
    return out


def cell_pace(slope, cell, ok, ok2, n, nbr):
    """Mean pace of the 1' samples of every land cell (cells without a sample: mean of their neighbours)."""
    pf = pace_factor(slope)
    cnt = np.bincount(cell, minlength=n).astype(np.float64)
    psum = np.bincount(cell, weights=pf.ravel()[ok][ok2], minlength=n)
    pace = np.where(cnt > 0, psum / np.maximum(cnt, 1), np.nan)
    for _ in range(3):
        miss = np.nonzero(np.isnan(pace))[0]
        if len(miss) == 0:
            break
        for c in miss:
            v = [pace[k] for k in nbr[c] if k >= 0 and not np.isnan(pace[k])]
            if v:
                pace[c] = np.mean(v)
    return np.where(np.isnan(pace), 1.0, pace)


def step_rates(pace, nbr):
    pn = np.where(nbr >= 0, pace[np.maximum(nbr, 0)], 1.0)
    return np.where(nbr >= 0, zsim.MU / (0.5 * (pace[:, None] + pn)), 0.0)


def main():
    g = np.load(os.path.join(SRC, 'grid.npz'))
    meta = json.load(open(os.path.join(SRC, 'grid.json'), encoding='utf-8'))
    pop2d, land = g['pop'], g['land']
    H, W = land.shape
    idx, ys, xs, nbr = zsim.land_index(land)
    pop = pop2d[ys, xs].astype(np.int64)

    # ---- terrain
    f = netcdf_file(ETOPO, 'r', mmap=False)
    lat = f.variables['latitude'][:].astype(np.float64)
    lon = f.variables['longitude'][:].astype(np.float64)
    alt = f.variables['altitude'][:].astype(np.float64)
    f.close()
    alt = np.maximum(alt, 0.0)                       # sea surface for ocean samples
    dlat = np.deg2rad(np.diff(lat).mean()) * 6371000.0
    dlon = np.deg2rad(np.diff(lon).mean()) * 6371000.0 * np.cos(np.deg2rad(lat))[:, None]
    gy = np.gradient(alt, axis=0) / dlat
    gx = np.gradient(alt, axis=1) / dlon
    slope = np.hypot(gx, gy)
    LON, LAT = np.meshgrid(lon, lat)
    px, py = G.albers(LON.ravel(), LAT.ravel())
    j = np.floor((py - meta['y0']) / meta['cell_km']).astype(np.int64)
    i = np.floor((px - meta['x0']) / meta['cell_km']).astype(np.int64)
    ok = (j >= 0) & (j < H) & (i >= 0) & (i < W)
    j, i = j[ok], i[ok]
    ok2 = land[j, i]
    j, i = j[ok2], i[ok2]
    cell = idx[j, i]
    n = len(pop)
    cnt = np.bincount(cell, minlength=n).astype(np.float64)
    nmiss = int((cnt == 0).sum())
    zsum = np.bincount(cell, weights=alt.ravel()[ok][ok2], minlength=n)
    ssum = np.bincount(cell, weights=slope.ravel()[ok][ok2], minlength=n)
    elev = np.where(cnt > 0, zsum / np.maximum(cnt, 1), 0.0)
    smean = np.where(cnt > 0, ssum / np.maximum(cnt, 1), 0.0)
    pace = cell_pace(slope, cell, ok, ok2, n, nbr)
    pace3 = cell_pace(3.0 * slope, cell, ok, ok2, n, nbr)
    walk, walk3 = 1.0 / pace, 1.0 / pace3
    mu_flat = np.where(nbr >= 0, zsim.MU, 0.0)
    mu_terrain = step_rates(pace, nbr)
    mu_terrain3 = step_rates(pace3, nbr)

    # ---- police in every cell, in the national proportion
    fpol = POLICE / POP_2018
    rng = np.random.default_rng(20261027)
    police = rng.binomial(pop, fpol)

    os.makedirs(OUT, exist_ok=True)
    cc = city_cells(pop2d, land, meta)
    names = list(cc)
    cidx = np.array([idx[cc[k]] for k in names], np.int64)
    np.savez_compressed(os.path.join(OUT, 'prep.npz'), pop=pop, nbr=nbr, ys=ys, xs=xs, walk=walk, walk3=walk3, elev=elev,
                        slope=smean, mu_flat=mu_flat, mu_terrain=mu_terrain, mu_terrain3=mu_terrain3, police=police, city_idx=cidx,
                        city_names=np.array(names), county=g['county'][ys, xs])
    info = {
        'land_cells': n, 'cells_without_etopo_sample': nmiss,
        'walk_mean': float(walk.mean()), 'walk_pop_weighted': float((walk * pop).sum() / pop.sum()),
        'walk_max': float(walk.max()), 'walk_p01': float(np.percentile(walk, 1)), 'walk_min': float(walk.min()),
        'walk3_mean': float(walk3.mean()), 'walk3_p01': float(np.percentile(walk3, 1)), 'walk3_min': float(walk3.min()),
        'police_total': int(police.sum()), 'police_per_person': fpol, 'people_per_officer': 1 / fpol,
    }
    for nm in ['New York, NY', 'Pittsburgh, PA', 'Denver, CO', 'Salt Lake City, UT', 'Scranton, PA']:
        c = idx[cc[nm]]
        info[f'walk_{nm}'] = [round(float(walk[c]), 3), round(float(walk3[c]), 3)]
    json.dump(info, open(os.path.join(OUT, 'prep.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
