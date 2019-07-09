# coding:utf-8
# filename:ensemble.py
# function:模型识别结果融合程序, 融合4个最好的结果, 投票融合.

import csv

name = "submission_ensemble_zh"
sub_files = [
            './submit/baseline_nasnet.csv',
            './submit/baseline_ResNet50.csv',
            './submit/baseline_Xception.csv',
            './submit/baseline_InceptionResNetV2.csv',
            './submit/baseline_Densenet201.csv',
            './submit/subfiles_2_ensemble_0.9754.csv',
            './submit/submission_tta_32.csv'
            ]

# sub_files = [
#             './submit/submission_Local_Boot_Epochs350_512size_tta0.csv',
#             './submit/submission_Local_Boot_Epochs350_512size_tta1.csv',
#             './submit/submission_Local_Boot_Epochs350_512size_tta2.csv',
#             './submit/submission_Local_Boot_Epochs350_512size_tta3.csv',
#             './submit/submission_Local_Boot_Epochs350_512size_original_0.903.csv']
print(len(sub_files))

# Weights of the individual subs
sub_weight = [
            0.9709 ** 2,
            0.9638 ** 2,
            0.9660 ** 2,
            0.9562 ** 2,
            0.9710 ** 2,
            0.9754 ** 2,
            0.9742 ** 2]
# sub_weight = [
#             0.896 ** 2,
#             0.894 ** 2,
#             0.898 ** 2,
#             0.856 ** 2,
#             0.903 ** 2]

Hlabel = 'id'
Htarget = 'label'
npt = 1  # number of places in target

place_weights = {}
for i in range(npt):
    place_weights[i] = (1 / (i + 1))

print(place_weights)


lg = len(sub_files)
sub = [None] * lg
for i, file in enumerate(sub_files):
    ## input files ##
    print("Reading {}: w={} - {}".format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file, "r"))    # 将csv文件数据读入到字典中
    sub[i] = sorted(reader, key=lambda d: str(d[Hlabel]))


## output file ##
out = open("./submit/%s.csv" % name, "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel, Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt, 0) + (place_weights[ind] * sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()
