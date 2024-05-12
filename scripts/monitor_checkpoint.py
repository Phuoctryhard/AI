import datetime

timestamps = [
    1715372497.9711401,
    1715372584.239055,
    1715372677.2560706,
    1715372764.69967,
    1715372852.7316086,
    1715372941.2191112,
    1715373029.1574686,
    1715356033.0388734
]

checkpoints = [
    "ckpt-7",
    "ckpt-8",
    "ckpt-9",
    "ckpt-10",
    "ckpt-11",
    "ckpt-12",
    "ckpt-13",
    "last_preserved"
]

for cp, ts in zip(checkpoints, timestamps):
    dt_object = datetime.datetime.fromtimestamp(ts)
    print(f"{cp}: {dt_object}")
