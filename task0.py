import _init_paths
from datasets.factory import get_imdb

import cv2

imdb = get_imdb('voc_2007_trainval')

print("The classes are:")
im_class_names = []
for i in  imdb.classes(imdb._load_pascal_annotation(imdb.image_index[2019])['gt_classes']-1).tolist():
    im_class_names.append(imdb.classes[i])
    print(imdb.classes[i])
image_path = imdb.image_path_at(2019)

print("Image path is :"+image_path)

rects = imdb._load_pascal_annotation(imdb.image_index[2019])['boxes']
img = cv2.imread(image_path)
for r in rects:
    cv2.rectangle(img,(r[0],r[1]),(r[2],r[3]),(0,255,0))
cv2.imwrite('task0_4',img)

img = cv2.imread(image_path)
dbs = imdb.selective_search_roidb()
max_score_indices = np.argsort(dbs[2019]['boxscores'].squeeze())[::-1][:10]
for b in dbs[2019]['boxes'][:10]:
    cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),(0,255,0))
cv2.imwrite('task0_3',img)