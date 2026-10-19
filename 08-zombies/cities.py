"""Reference places for the narration and the map labels: (name, state, latitude, longitude) of each downtown.
The cell used for a city is its most populated land cell within 6 km of these coordinates."""
import numpy as np

import grid as G

CITIES = [
    ('New York', 'NY', 40.7128, -74.0060), ('Los Angeles', 'CA', 34.0522, -118.2437), ('Chicago', 'IL', 41.8781, -87.6298),
    ('Houston', 'TX', 29.7604, -95.3698), ('Phoenix', 'AZ', 33.4484, -112.0740), ('Philadelphia', 'PA', 39.9526, -75.1652),
    ('San Antonio', 'TX', 29.4241, -98.4936), ('San Diego', 'CA', 32.7157, -117.1611), ('Dallas', 'TX', 32.7767, -96.7970),
    ('San Francisco', 'CA', 37.7749, -122.4194), ('Seattle', 'WA', 47.6062, -122.3321), ('Denver', 'CO', 39.7392, -104.9903),
    ('Washington', 'DC', 38.9072, -77.0369), ('Boston', 'MA', 42.3601, -71.0589), ('Atlanta', 'GA', 33.7490, -84.3880),
    ('Miami', 'FL', 25.7617, -80.1918), ('Minneapolis', 'MN', 44.9778, -93.2650), ('Detroit', 'MI', 42.3314, -83.0458),
    ('Las Vegas', 'NV', 36.1699, -115.1398), ('Salt Lake City', 'UT', 40.7608, -111.8910), ('Portland', 'OR', 45.5152, -122.6784),
    ('New Orleans', 'LA', 29.9511, -90.0715), ('St. Louis', 'MO', 38.6270, -90.1994), ('Kansas City', 'MO', 39.0997, -94.5786),
    ('Nashville', 'TN', 36.1627, -86.7816), ('Charlotte', 'NC', 35.2271, -80.8431), ('Pittsburgh', 'PA', 40.4406, -79.9959),
    ('Cleveland', 'OH', 41.4993, -81.6944), ('Albuquerque', 'NM', 35.0844, -106.6504), ('El Paso', 'TX', 31.7619, -106.4850),
    ('Omaha', 'NE', 41.2565, -95.9345), ('Bakersfield', 'CA', 35.3733, -119.0187), ('Scranton', 'PA', 41.4090, -75.6624),
    ('Wilkes-Barre', 'PA', 41.2459, -75.8813), ('Allentown', 'PA', 40.6023, -75.4714), ('Billings', 'MT', 45.7833, -108.5007),
    ('Missoula', 'MT', 46.8721, -113.9940), ('Boise', 'ID', 43.6150, -116.2023), ('Reno', 'NV', 39.5296, -119.8138),
    ('Elko', 'NV', 40.8324, -115.7631), ('Cheyenne', 'WY', 41.1400, -104.8202), ('Fargo', 'ND', 46.8772, -96.7898),
]


def city_cells(pop2d, land, meta, radius_km=6.0):
    """{name: (row, col)} of the most populated land cell near each downtown."""
    x, y = G.albers(np.array([c[3] for c in CITIES]), np.array([c[2] for c in CITIES]))
    out = {}
    r = int(np.ceil(radius_km / meta['cell_km']))
    for (name, st, _, _), xx, yy in zip(CITIES, x, y):
        j = int((yy - meta['y0']) // meta['cell_km'])
        i = int((xx - meta['x0']) // meta['cell_km'])
        best, bj, bi = -1, j, i
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if dj * dj + di * di > r * r:
                    continue
                jj, ii = j + dj, i + di
                if 0 <= jj < pop2d.shape[0] and 0 <= ii < pop2d.shape[1] and land[jj, ii] and pop2d[jj, ii] > best:
                    best, bj, bi = pop2d[jj, ii], jj, ii
        out[f'{name}, {st}'] = (bj, bi)
    return out
