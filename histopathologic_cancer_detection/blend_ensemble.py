# coding: utf-8
# filename: ensemble.py
# function: 将4个模型测试结果文件集合在一起

import os
import pandas as pd

os.listdir('./')

# sub1 = pd.read_csv('./submit/subfiles_2_ensemble_0.9754.csv')
# sub2 = pd.read_csv('./submit/submission_tta_32_0.9742.csv')
# # sub3 = pd.read_csv('./submit/DenseNet201_Epoch60_tta_162.csv')
# sub4 = pd.read_csv('./submit/NASNetMobile_Epoch10_tta_64_0.9730.csv')
# sub5 = pd.read_csv('./submit/sub_tta_0.9738.csv')
#
# sub1['label'] = sub1['label'] * 0.4 + sub2['label'] * 0.2 + sub4['label'] * 0.2 + sub5['label'] * 0.2
# sub1.to_csv('./submit/subfiles_4_ensemble_0.4_0.2_0.2_0.2.csv', index=False)

base_name = 'DenseNet201_Epoch60'
tta_folder = base_name.split('_')[0].lower() + '_tta_result'
num_tta = 128

src_dir = os.getcwd()
src_dir = src_dir.replace('\\', '/')
print(os.listdir(src_dir))
src_dir = './'

SUBMIT = os.path.join(src_dir, 'submit')
TTA_OUTPUT = os.path.join(src_dir, 'submit/' + tta_folder)

df = pd.read_csv('./dataset/sample_submission.csv')
for i in range(num_tta):
    file_tta = 'DenseNet201_Epoch60_tta' + str(i) + '.csv'

    file_path = os.path.join(TTA_OUTPUT, file_tta)
    print(file_path)
    sub1 = pd.read_csv(file_path)
    df['label'] += sub1['label']
    print(i)
    print(file_tta)
df['label'] /= 128
# df.to_csv('./submit/ensemble_tta_32.csv', index=False)
df.to_csv(os.path.join(SUBMIT, base_name + '_tta_' + str(num_tta) + '.csv'), index=False)

