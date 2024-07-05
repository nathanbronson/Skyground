from tqdm import tqdm
import numpy as np

from opensky_api import OpenSkyApi
api = OpenSkyApi()

## MAKE STATES ONLY FETCH EVERY ONCE IN A WHILE AND INCREASE PATH RESOLUTION

def increase_resolution(path, factor=4):
    return path
    if len(path[0]) < 2:
        return path
    return np.array([np.concatenate([np.linspace(s0, s1, factor) for s0, s1 in zip(p[:-1], p[1:])]) for p in path])

def wrap_try(c):
    try:
        return api.get_track_by_aircraft(c)
    except Exception as e:
        print(type(e), e)

ALT_FAC = .5
def get_in_progress_paths(box=(15, 70, -135, -65), only_first=None, alt_fac=ALT_FAC, factor=4, crafts=None, state=False):
    if only_first is None:
        states = api.get_states(bbox=box)
        crafts = list(sorted(set([s.icao24 for s in states.states])))
    elif crafts is None:
        states = api.get_states(bbox=box)
        crafts = np.random.choice(list(sorted(set([s.icao24 for s in states.states]))), size=only_first)
    elif state:
        states = api.get_states(bbox=box)
        _c = list(sorted(set([s.icao24 for s in states.states])))
        crafts = list(filter(lambda x: x in _c, crafts))
        while len(crafts) < only_first:
            if (_n := np.random.choice(_c)) not in crafts:
                crafts.append(_n)
    tracks = [r for c in tqdm(crafts) if (r := wrap_try(c)) is not None]
    paths = [{
        "icao24": t.icao24,
        "callsign": t.callsign,
        "time": [i[0] for i in t.path],
        "lats": [i[1] for i in t.path],
        "longs": [i[2] for i in t.path],
        "alts": [i[3] * alt_fac for i in t.path]
    } for t in filter(lambda x: x is not None, tracks)]
    return [increase_resolution(np.array([p["lats"], p["longs"], p["alts"]]), factor=factor) for p in paths], crafts

def get_paths_at_time(time, box=(15, 70, -135, -65), only_first=None, alt_fac=ALT_FAC, expire=1e6, factor=4, crafts=None, state=False):
    if only_first is None:
        states = api.get_states(bbox=box)
        crafts = list(sorted(set([s.icao24 for s in states.states])))
    elif crafts is None:
        states = api.get_states(bbox=box)
        crafts = np.random.choice(list(sorted(set([s.icao24 for s in states.states]))), size=only_first)
    elif state:
        states = api.get_states(bbox=box)
        _c = list(sorted(set([s.icao24 for s in states.states])))
        crafts = list(filter(lambda x: x in _c, crafts))
        while len(crafts) < only_first:
            if (_n := np.random.choice(_c)) not in crafts:
                crafts.append(_n)
    tracks = [r for c in tqdm(crafts) if (r := wrap_try(c)) is not None]
    paths = [{
        "icao24": t.icao24,
        "callsign": t.callsign,
        "time": [i[0] for i in t.path],
        "lats": [i[1] for i in t.path],
        "longs": [i[2] for i in t.path],
        "alts": [i[3] * alt_fac for i in t.path]
    } for t in filter(lambda x: x is not None, tracks)]
    t = time
    return [increase_resolution(np.array([np.array(p["lats"])[idx], np.array(p["longs"])[idx], np.array(p["alts"])[idx]]), factor=factor) for p in paths if np.sum((idx := ((np.array(p["time"]) <= t) & ((t - expire) <= np.array(p["time"]))))) > 0], crafts

def get_timed_paths(box=(15, 70, -135, -65), only_first=None, alt_fac=ALT_FAC, expire=1e6, factor=4, crafts=None, state=False):
    if only_first is None:
        states = api.get_states(bbox=box)
        crafts = list(sorted(set([s.icao24 for s in states.states])))
    elif crafts is None:
        states = api.get_states(bbox=box)
        crafts = np.random.choice(list(sorted(set([s.icao24 for s in states.states]))), size=only_first)
    elif state:
        states = api.get_states(bbox=box)
        _c = list(sorted(set([s.icao24 for s in states.states])))
        crafts = list(filter(lambda x: x in _c, crafts))
        while len(crafts) < only_first:
            if (_n := np.random.choice(_c)) not in crafts:
                crafts.append(_n)
    tracks = [r for c in tqdm(crafts) if (r := wrap_try(c)) is not None]
    paths = [{
        "icao24": t.icao24,
        "callsign": t.callsign,
        "time": [i[0] for i in t.path],
        "lats": [i[1] for i in t.path],
        "longs": [i[2] for i in t.path],
        "alts": [i[3] * alt_fac for i in t.path]
    } for t in filter(lambda x: x is not None, tracks)]
    times = np.unique(np.concatenate([i["time"] for i in paths]))
    interval = (np.min(times), np.max(times))
    times = np.linspace(*interval, num=len(times))
    return [[increase_resolution(np.array([np.array(p["lats"])[idx], np.array(p["longs"])[idx], np.array(p["alts"])[idx]]), factor=factor) for p in paths if np.sum((idx := ((np.array(p["time"]) <= t) & ((t - expire) <= np.array(p["time"]))))) > 0] for t in times], crafts

def get_mask(oob):
    m = []
    cur = True
    for i in oob:
        if cur is None:
            m.append(False)
            continue
        if cur:
            if i:
                m.append(False)
            else:
                cur = False
                m.append(True)
        else:
            if i:
                cur = None
                m.append(False)
            else:
                m.append(True)
    return m

def freeze_oob(flights, box=(15, 70, -135, -65), timed=False):
    res = []
    for f in flights:
        if len(f) == 0:
            continue
        if timed:
            _f = f[-1]
        else:
            _f = f
        latoob = (_f[0] < box[0]) | (box[1] < _f[0])
        longoob = (_f[1] < box[2]) | (box[3] < _f[1])
        oob = [False, False]#latoob | longoob
        m = get_mask(oob)
        if len(_f[0]) < 3:
            continue
        if np.all(m) and np.max(np.abs(_f[0, 1:] - _f[0, :-1])) < 2.5 and np.max(np.abs(_f[1, 1:] - _f[1, :-1])) < 5:
            res.append(f)
    return res