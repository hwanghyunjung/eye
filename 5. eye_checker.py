#!/usr/bin/env python
# coding: utf-8

# In[3]:
import eye_keras as eye
import os
from PIL import Image
import numpy as np

image_size = 50
categories = ["Bulging_Eyes", "Cataracts", "Crossed_Eyes", "Normal", "Uveitis"]
# 입력 이미지를 Numpy로 변환하기 --- (※2)
X = []
filepath= "./img/test/"
files = [filepath + f for f in os.listdir(filepath)]
for fname in files[0:]:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    X.append(in_data)
    files.append(fname)
X = np.array(X)
# CNN 모델 구축하기 --- (※3)
model = eye.build_model(X.shape[1:])
model.load_weights("./data/eye_diseases.hdf5")
# 데이터 예측하기 --- (※4)
html = ""
pre = model.predict(X)

for i, p in enumerate(pre):
    y = p.argmax()
    result = pre[i][y]*100
    print("+입력:", files[i])
    print("|input image prediction : ", categories[y])
    print("|input image prediction(%) : ", result)
    html += """
        <h3>Input Image : {0}</h3>
        <div>
          <p><img src="{1}" width=300></p>
          <p>Input Image Prediction : {2}</p>
          <p>Input Image Prediction(%) : {3}</p>
        </div>
    """.format(os.path.basename(files[i]),
        files[i],
        categories[y],
        result
        )
# 리포트 저장하기 --- (※5)
html = "<html><body style='text-align:center;'>" +     "<style> p { margin:0; padding:0; } </style>" +     html + "</body></html>"
with open("eye-result.html", "w", encoding="UTF-8") as f:
    f.write(html)

