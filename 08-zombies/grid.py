"""Population grid for the zombie video: every person of the 2020 Census (lower 48 + DC) on 3 km cells.

Same idea as Alemi et al. 2015 (they used 2010 census blocks): each person is placed at a random spot
inside their own area, and the people are counted on a lattice of ~3 km cells.
Here the areas are the 2020 census block groups (population from the Census Bureau's centers-of-population
file, shapes from the 2020 cartographic boundary file), and the lattice is an equal-area projection
(USA Contiguous Albers, spherical earth) with cells of exactly 3 x 3 km.

As in the paper, six iterations of binary closing add one person to vacant land cells between occupied ones.

Run: python grid.py      (writes out/grid.npz and out/grid.json; the census files go in census/, see README.md)
"""
import json
import os
import zipfile

import numba as nb
import numpy as np
from scipy import ndimage

HERE_ = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('ZOMBIE_CENSUS', os.path.join(HERE_, 'census'))
OUT = os.environ.get('ZOMBIE_OUT', os.path.join(HERE_, 'out'))
CELL = 3.0            # km
SUB = 8               # fine subcells per cell side (375 m) used to place people inside each block group
R_EARTH = 6371.0      # km
PHI1, PHI2, PHI0, LAM0 = 29.5, 45.5, 23.0, -96.0     # EPSG:5070 parameters
EXCLUDE = {'02', '15', '60', '66', '69', '72', '78'}  # Alaska, Hawaii, territories
SEED = 2020


def albers(lon, lat):
    """Spherical Albers equal-area conic, km."""
    p1, p2, p0 = np.radians(PHI1), np.radians(PHI2), np.radians(PHI0)
    n = (np.sin(p1) + np.sin(p2)) / 2
    c = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho0 = R_EARTH * np.sqrt(c - 2 * n * np.sin(p0)) / n
    rho = R_EARTH * np.sqrt(c - 2 * n * np.sin(np.radians(lat))) / n
    th = n * np.radians(np.asarray(lon) - LAM0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)


def albers_inv(x, y):
    p1, p2, p0 = np.radians(PHI1), np.radians(PHI2), np.radians(PHI0)
    n = (np.sin(p1) + np.sin(p2)) / 2
    c = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho0 = R_EARTH * np.sqrt(c - 2 * n * np.sin(p0)) / n
    rho = np.hypot(x, rho0 - y)
    th = np.arctan2(x, rho0 - y)
    lat = np.degrees(np.arcsin((c - (rho * n / R_EARTH) ** 2) / (2 * n)))
    lon = LAM0 + np.degrees(th / n)
    return lon, lat


def extract():
    z = zipfile.ZipFile(os.path.join(DATA, 'cb_2020_us_bg_500k.zip'))
    for ext in ('shp', 'shx', 'dbf'):
        f = os.path.join(DATA, f'cb_2020_us_bg_500k.{ext}')
        if not os.path.exists(f):
            z.extract(f'cb_2020_us_bg_500k.{ext}', DATA)


def read_dbf(path, fields):
    raw = np.memmap(path, dtype=np.uint8, mode='r')
    nrec = int(raw[4:8].view('<u4')[0])
    hlen = int(raw[8:10].view('<u2')[0])
    rlen = int(raw[10:12].view('<u2')[0])
    desc, pos, off = {}, 32, 1
    while raw[pos] != 0x0D:
        name = bytes(raw[pos:pos + 11]).split(b'\0')[0].decode()
        ln = int(raw[pos + 16])
        desc[name] = (off, ln)
        off += ln
        pos += 32
    rec = np.asarray(raw[hlen:hlen + nrec * rlen]).reshape(nrec, rlen)
    out = {}
    for f in fields:
        o, ln = desc[f]
        out[f] = np.array([bytes(r).decode('latin-1').strip() for r in rec[:, o:o + ln]])
    return out


def read_shp(path_shp, path_shx):
    """Polygons -> (point x, point y, part starts, polygon -> first part, polygon -> number of parts)."""
    shx = np.fromfile(path_shx, dtype=np.uint8)
    idx = shx[100:].view('>i4').reshape(-1, 2)
    offs = idx[:, 0].astype(np.int64) * 2
    shp = np.memmap(path_shp, dtype=np.uint8, mode='r')
    npoly = len(offs)
    nparts = np.zeros(npoly, np.int64)
    npts = np.zeros(npoly, np.int64)
    for i in range(npoly):
        o = offs[i] + 8
        if int(np.frombuffer(shp, '<i4', 1, o)[0]) == 0:
            continue
        nparts[i], npts[i] = np.frombuffer(shp, '<i4', 2, o + 36)
    pstart = np.zeros(npoly + 1, np.int64)
    pstart[1:] = np.cumsum(nparts)
    qstart = np.zeros(npoly + 1, np.int64)
    qstart[1:] = np.cumsum(npts)
    part_start = np.zeros(pstart[-1] + 1, np.int64)
    lon = np.zeros(qstart[-1])
    lat = np.zeros(qstart[-1])
    for i in range(npoly):
        if nparts[i] == 0:
            continue
        o = offs[i] + 8 + 44
        parts = np.frombuffer(shp, '<i4', nparts[i], o).astype(np.int64)
        pts = np.frombuffer(shp, '<f8', 2 * npts[i], o + 4 * nparts[i]).reshape(-1, 2)
        part_start[pstart[i]:pstart[i + 1]] = qstart[i] + parts
        lon[qstart[i]:qstart[i + 1]] = pts[:, 0]
        lat[qstart[i]:qstart[i + 1]] = pts[:, 1]
    part_start[-1] = qstart[-1]
    # a part ends where the next part starts (the next part of the same polygon or the first of the next one)
    return lon, lat, part_start, pstart, nparts


@nb.njit(cache=True)
def place(px, py, part_start, pstart, nparts, pop, cx, cy, x0, y0, H, W, seed):
    """Rasterize each block group at 375 m and spread its people uniformly over its subcells (multinomial)."""
    np.random.seed(seed)
    rf = 3.0 / 8
    grid = np.zeros(H * W, np.int64)
    fine = np.zeros(H * W, np.int32)           # covered subcells per cell (land mask)
    best = np.zeros(H * W, np.int32)
    owner = -np.ones(H * W, np.int64)          # block group with the most subcells in the cell
    buf = np.empty(4_000_000, np.int64)
    xs = np.empty(20000)
    tiny = 0
    for i in range(len(pop)):
        if nparts[i] == 0:
            continue
        a, b = pstart[i], pstart[i] + nparts[i]
        q0, q1 = part_start[a], part_start[b]
        ymin, ymax = py[q0:q1].min(), py[q0:q1].max()
        j0 = int(np.ceil((ymin - y0) / rf - 0.5))
        j1 = int(np.floor((ymax - y0) / rf - 0.5))
        m = 0
        for j in range(j0, j1 + 1):
            y = y0 + (j + 0.5) * rf
            nx = 0
            for p in range(a, b):
                s, e = part_start[p], part_start[p + 1]
                for k in range(s, e - 1):
                    y1, y2 = py[k], py[k + 1]
                    if (y1 <= y < y2) or (y2 <= y < y1):
                        xs[nx] = px[k] + (y - y1) * (px[k + 1] - px[k]) / (y2 - y1)
                        nx += 1
            if nx < 2:
                continue
            xr = np.sort(xs[:nx])
            for k in range(0, nx - 1, 2):
                i0 = int(np.ceil((xr[k] - x0) / rf - 0.5))
                i1 = int(np.floor((xr[k + 1] - x0) / rf - 0.5))
                for ii in range(i0, i1 + 1):
                    buf[m] = (j // 8) * W + (ii // 8)
                    m += 1
        if m == 0:
            tiny += 1
            c = int((cy[i] - y0) // 3.0) * W + int((cx[i] - x0) // 3.0)
            grid[c] += pop[i]
            continue
        cells = np.sort(buf[:m])
        left, wleft = pop[i], m
        k = 0
        while k < m:
            c = cells[k]
            n = 1
            while k + n < m and cells[k + n] == c:
                n += 1
            fine[c] += n
            if n > best[c]:
                best[c] = n
                owner[c] = i
            if left > 0:
                got = left if wleft == n else np.random.binomial(left, n / wleft)
                grid[c] += got
                left -= got
            wleft -= n
            k += n
    return grid, fine, owner, tiny


def main():
    os.makedirs(OUT, exist_ok=True)
    extract()
    dbf = read_dbf(os.path.join(DATA, 'cb_2020_us_bg_500k.dbf'), ['GEOID', 'STATEFP', 'ALAND'])
    lon, lat, part_start, pstart, nparts = read_shp(os.path.join(DATA, 'cb_2020_us_bg_500k.shp'),
                                                    os.path.join(DATA, 'cb_2020_us_bg_500k.shx'))
    keep = np.array([s not in EXCLUDE for s in dbf['STATEFP']])
    # census population and population center by block group
    cen = np.genfromtxt(os.path.join(DATA, 'CenPop2020_Mean_BG.txt'), delimiter=',', skip_header=1,
                        dtype=None, encoding='utf-8-sig')
    geo = {f'{r[0]:02d}{r[1]:03d}{r[2]:06d}{r[3]}': (int(r[4]), float(r[5]), float(r[6])) for r in cen}
    npoly = len(dbf['GEOID'])
    pop = np.zeros(npoly, np.int64)
    clat = np.zeros(npoly)
    clon = np.zeros(npoly)
    missing = 0
    for i, g in enumerate(dbf['GEOID']):
        if not keep[i]:
            nparts[i] = 0
            continue
        if g in geo:
            pop[i], clat[i], clon[i] = geo[g]
        else:
            missing += 1
    census_total = sum(v[0] for g, v in geo.items() if g[:2] not in EXCLUDE)
    px, py = albers(lon, lat)
    cx, cy = albers(clon, clat)
    sel = np.concatenate([np.arange(part_start[pstart[i]], part_start[pstart[i] + nparts[i]])
                          for i in np.flatnonzero(nparts)])
    x0 = np.floor(px[sel].min() / CELL) * CELL - 2 * CELL
    y0 = np.floor(py[sel].min() / CELL) * CELL - 2 * CELL
    W = int(np.ceil((px[sel].max() - x0) / CELL)) + 2
    H = int(np.ceil((py[sel].max() - y0) / CELL)) + 2
    grid, fine, owner, tiny = place(px, py, part_start, pstart, nparts, pop, cx, cy, x0, y0, H, W, SEED)
    grid = grid.reshape(H, W)
    land = (fine > 0).reshape(H, W)
    occ = grid > 0
    closed = ndimage.binary_closing(occ, iterations=6)
    added = closed & ~occ & land
    grid[added] += 1
    county = np.full(H * W, -1, np.int32)
    has = owner >= 0
    county[has] = np.array([int(g[:5]) for g in dbf['GEOID'][owner[has]]], np.int32)
    county = county.reshape(H, W)
    occ = grid[grid > 0]
    meta = {
        'source': 'US Census Bureau: CenPop2020_Mean_BG.txt + cb_2020_us_bg_500k (lower 48 + DC)',
        'cell_km': CELL, 'H': H, 'W': W, 'x0': x0, 'y0': y0,
        'projection': {'type': 'albers_spherical', 'phi1': PHI1, 'phi2': PHI2, 'phi0': PHI0, 'lam0': LAM0, 'R': R_EARTH},
        'census_total_lower48_dc': int(census_total),
        'placed_people': int(grid.sum() - added.sum()),
        'binary_closing_added': int(added.sum()),
        'total_people': int(grid.sum()),
        'block_groups_missing_population': missing,
        'block_groups_smaller_than_a_subcell': int(tiny),
        'land_cells': int(land.sum()), 'occupied_cells': int((grid > 0).sum()),
        'largest_cell': int(grid.max()), 'mean_occupied': float(occ.mean()), 'median_occupied': float(np.median(occ)),
    }
    j, i = np.unravel_index(np.argmax(grid), grid.shape)
    meta['largest_cell_lonlat'] = [float(v) for v in albers_inv(x0 + (i + 0.5) * CELL, y0 + (j + 0.5) * CELL)]
    np.savez_compressed(os.path.join(OUT, 'grid.npz'), pop=grid.astype(np.int32), land=land, county=county,
                        landfrac=(fine.reshape(H, W) * 255 // 64).clip(0, 255).astype(np.uint8))
    with open(os.path.join(OUT, 'grid.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
