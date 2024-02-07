import base64
import pandas as pd
import re
import os
from google.cloud import vision


def readImages():
    img_lst = [f for f in os.listdir('images/') if os.path.isfile(os.path.join('images/', f))]
    data = pd.DataFrame()
    for img_nm in img_lst:
        print(img_nm)
        get_num = re.findall(r'\d+', img_nm)
        points = imageToList(img_nm)
        df = pointsToDataFrame(pd.DataFrame(), get_num, points)
        data = pd.concat([data, df])
    return data.sort_values(by=['ID'])


def imageToList(img_nm):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'cred.json'
    base64_encoded = base64.b64encode(open(f"images/{img_nm}", 'rb').read()).decode('utf-8')
    client = vision.ImageAnnotatorClient()
    response = client.text_detection(image={"content": base64_encoded})
    points = str(response.text_annotations[0].description).replace('O', '0').replace('(', '').split('\n')
    return [int(point) for point in points]


def pointsToDataFrame(df, get_num, points):
    df['ID'] = [int(get_num[0])]
    df['Thrift'] = sum(points[:3])
    df['Parenting'] = sum(points[3:6])
    df['Self-realization'] = sum(points[6:])
    df['Total points'] = sum(points)
    df['Season'] = [get_num[1]]
    df['Episode'] = [get_num[2]]
    df['Type'] = [get_num[3]]
    df['City'] = [get_num[4]]
    return df
