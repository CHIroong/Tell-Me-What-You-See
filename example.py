import json

from patch_manager import PatchManager
from tagged_data_fetcher import TaggedDataFetcher

pm = PatchManager()
print("init complete")
lll = [59, 2, 3, 5, 6, 8, 10, 11, 12, 17, 18, 21, 22, 23, 26, 27, 28, 31, 33, 37, 38, 39, 42, 44, 45, 49, 50, 53, 54, 56, 58, 60, 64, 65, 67, 69, 70, 72, 74, 75, 79, 81, 83, 85, 86, 89, 91, 93]
for ind, i in enumerate(lll):
#for ind, i in enumerate([2, 3, 5, 6, 8, 10, 11, 12, 17, 18, 21, 22, 23, 26, 27, 28, 31, 33, 37, 38, 39, 42, 44, 45, 49, 50, 53, 54, 56, 58, 59, 60, 64, 65, 67, 69, 70, 72, 74, 75, 79, 81, 83, 85, 86, 89, 91, 93]):
    pm.feed(len(lll) - 1 - ind, 'final_stimuli/screencapture-%d.png' % i, 'final_stimuli/%d.txt' % i)
    print("fed %d" % i)
print("feeding complete")
pm.save_patches_at('final_stimuli_patches/')
with open('final_stimuli_spec_summary_dhodkseho.json', 'w') as f:
    f.write(pm.generate_spec(verbose=True))



"""
IND_MIN = 0 #inclusive
IND_MAX = 93 #inclusive

pm = PatchManager()
to_exclude = []
for i in range(IND_MIN, IND_MAX + 1):
    try:
        if not pm.is_valid('final_stimuli/screencapture-%d.png' % i, 'final_stimuli/%d.txt' % i, verbose=True):
            to_exclude.append(i)
    except Exception:
        print("file not exists:", i)
        to_exclude.append(i)

print("To Exclude: ", to_exclude)
"""


"""

for i in range(IND_MIN, IND_MAX + 1):
    if i not in to_exclude:
        pm.feed(i, 'final_stimuli/screencapture-%d.png' % i, 'final_stimuli/%d.txt' % i)
pm.save_patches_at('final_stimuli_patches/')
with open('final_stimuli_spec_%d_%d.json' % (IND_MIN, IND_MAX), 'w') as f:
    f.write(pm.generate_spec(verbose=True))
"""

"""
tag = TaggedDataFetcher("http://52.79.189.93:8005/screenshots/1/export", to_exclude)
with open("tagged.txt", "w") as f:
    f.write(json.dumps(tag.id_and_rects()))

print("fetched tagged info")
"""
