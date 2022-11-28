import os
import shutil

path_in = '/media/neptun/Data/Dataset/celeba_a'
path_out = '/home/neptun/PycharmProjects/datasets/celeba/train'

analitik = {}
file_class = os.path.join(path_in, 'list_attr_celeba.txt')
out_0 = []
out_1 = []
with open(file_class) as f:
    f_row = f.readline()
    head = f.readline()
    head = head.replace('  ', ' ').strip()
    head_l = head.split(' ')
    for h in head_l:
        analitik[h] = 0
    for rows in f.readlines():
        a = rows.replace('  ', ' ').strip()
        rows_l = a.split(' ')

        for i, h in enumerate(head_l):
            if rows_l[i+1] == '1':
                analitik[head_l[i]] = analitik[head_l[i]] + 1
            if rows_l[i+1] == '-1':
                analitik[head_l[i]] = analitik[head_l[i]] - 1
        if rows_l[21] == '-1':
            out_0.append(rows_l[0])
        if rows_l[21] == '1':
            out_1.append(rows_l[0])


print(analitik)

for f in out_0:
    src = os.path.join(path_in, 'img_align_celeba', f)
    dst = os.path.join(path_out, '0', f)
    shutil.copyfile(src, dst)
for f in out_1:
    src = os.path.join(path_in, 'img_align_celeba', f)
    dst = os.path.join(path_out, '1', f)
    shutil.copyfile(src, dst)