import os
import json
from patch_manager.patch_manager import PatchManager
from sonifier.sonifier import Sonifier

if __name__ == "__main__":

    RESULT_NAME = "final_result"

    # 1. Generate Patches and Spec
    print("1. Generate Patches and Spec")
    pm = PatchManager()
    stimuli = [2, 3, 5, 6, 8, 10, 11, 12, 17, 18, 21, 22, 23, 26, 27, 28, 31, 33, 37, 38, 39, 42, 44, 45, 49, 50, 53, 54, 56, 58, 60, 61, 64, 65, 67, 69, 70, 72, 74, 75, 79, 81, 83, 85, 86, 89, 91, 93]
    """
    for ind, i in enumerate(stimuli):
        pm.feed(ind, f'data/screencapture-{i}.png', f'data/{i}.txt')
    pm.save_patches_at(f'{RESULT_NAME}_patches/')
    pm.save_spec_at(f'{RESULT_NAME}_spec.json')
    pm.generate_spec(verbose=True)

    """
    # 2. Sonify the patches
    print("2. Sonify the patches")
    sonifier = Sonifier(f'{RESULT_NAME}_spec.json', use_gpu=False, verbose=True)
    folder = f"{RESULT_NAME}_sound/"
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    for i in range(len(stimuli)):
        print(i)
        default = folder + f"default_{i}.mp3"
        summary = folder + f"summary_{i}.mp3"
        keyword = folder + f"keyword_{i}.mp3"
        glance = folder + f"glance_{i}.mp3"
        sg = folder + f"SG_{i}.mp3"
        ss = folder + f"SS_{i}.mp3"

        sonifier.generate_glance_mp3(i, glance)
        sonifier.generate_defaults_mp3(i, default)
        sonifier.generate_summary_mp3(i, summary)
        sonifier.generate_keywords_mp3(i, keyword)
        sonifier.generate_merged_sg_mp3(glance, keyword, default, sg)
        sonifier.generate_merged_ss_mp3(summary, default, ss)