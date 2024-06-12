import json
fname = '/data/sriharsha/datasets/MSCOCO_2017/annotations/captions_train2017.json'
f = open(fname)

data = json.load(f)  ###  this data is of type dict with keys: ['info', 'licenses', 'images', 'annotations']
img_info = data['images']
id2img = {}

for img in img_info:
    id = img['id']
    fname = img['file_name']
    id2img[str(id)] = fname

cap_info = data['annotations']  ### caption information

id2cap = {}


for cap in cap_info:
    id  = str(cap['image_id'])
    caption = cap['caption']
    # fname = id2img[id]
    if id in id2cap.keys():
        id2cap[id].append(caption)
    else:
        id2cap[id] = []
        id2cap[id].append(caption)

coco_det = []
for id in id2cap.keys():
    d = {}
    d['caption'] = id2cap[id]
    d['image'] = id2img[id]
    coco_det.append(d)


###################  Write data to a json file  ##
with open('/data/sriharsha/datasets/MSCOCO_2017/annotations/train2017_req.jsonl', 'w') as f:
    for img in coco_det:
        row = json.dumps(img)
        f.write(row+'\n')