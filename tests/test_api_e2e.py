"""End-to-end API test. Uses urllib multipart, NOT curl -F: a comma in a
path (PlantVillage's 'Pepper,_bell') makes curl -F silently fail to send."""
import glob, json, mimetypes, random, ssl, urllib.request, uuid, collections, sys, csv

B = "http://127.0.0.1:8000/api/scan"

def post(path):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as fh:
        data = fh.read()
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="upload.jpg"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        B, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

def sample(pat, k, seed=11):
    fs = sorted(glob.glob(pat, recursive=True))
    random.seed(seed)
    return random.sample(fs, min(k, len(fs))) if fs else []

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

print("=" * 76)
print(f"SUPPORTED LEAVES (test split) - should be ACCEPTED   n={N}/crop")
print("=" * 76)
rows = [r for r in csv.DictReader(open('datasets/prepared/manifest.csv'))
        if r['split'] == 'test']
random.seed(11)
for crop in ('tomato', 'potato', 'grape'):
    sub = [r for r in rows if r['crop'] == crop]
    pick = random.sample(sub, min(N, len(sub)))
    acc = cor = 0
    errs = collections.Counter()
    for r in pick:
        code, b = post('datasets/prepared/images/' + r['path'])
        if code != 200:
            errs[f"http{code}"] += 1
            continue
        if b.get('success'):
            acc += 1
            if b['plant']['id'] == crop and b['prediction']['id'] == r['condition']:
                cor += 1
        else:
            errs[b.get('reason', '?')] += 1
    print(f"  {crop:7s} accepted {acc:3d}/{len(pick)} ({acc/len(pick):.1%})   "
          f"correct {cor:3d}/{len(pick)} ({cor/len(pick):.1%})   rejections {dict(errs)}")

print()
print("=" * 76)
print(f"UNSUPPORTED - should be REJECTED   n={N}/category")
print("=" * 76)
for name, pat in [
    ("held-out species (unseen outliers)", 'datasets/ood/unsupported_lab_heldout/**/*.jpg'),
    ("unsupported field (PlantDoc)",       'datasets/ood/unsupported_field/**/*.jpg'),
    ("people (faces)",                     'datasets/ood/non_plant_objects/people/*.jpg'),
    ("animals",                            'datasets/ood/non_plant_objects/animals/*.jpg'),
    ("buildings",                          'datasets/ood/non_plant_objects/buildings/*.jpg'),
    ("food",                               'datasets/ood/non_plant_objects/food/*.jpg'),
    ("random objects",                     'datasets/ood/non_plant_objects/objects/*.jpg'),
    ("vehicles",                           'datasets/ood/non_plant_objects/vehicles/*.jpg'),
]:
    fs = sample(pat, N)
    if not fs:
        print(f"  {name:38s} (none)")
        continue
    rej = 0
    http_err = 0
    reasons = collections.Counter()
    for f in fs:
        code, b = post(f)
        if code != 200:
            http_err += 1
            continue
        if not b.get('success'):
            rej += 1
            reasons[b['reason']] += 1
    extra = f"  HTTP-errors {http_err}" if http_err else ""
    print(f"  {name:38s} rejected {rej:3d}/{len(fs)} ({rej/len(fs):.1%}){extra}")
    print(f"      {dict(reasons)}")
