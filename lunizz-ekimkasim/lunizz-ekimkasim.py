import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('C:/Users/Tolga/Desktop/tc6oq6g.jpg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
template = cv2.imread('C:/Users/Tolga/Desktop/reiskafasi.jpg', cv2.IMREAD_COLOR)

h, w = template.shape[:2]

threshold = 0.55
res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


prev_min_val, prev_max_val, prev_min_loc, prev_max_loc = None, None, None, None
while threshold < 1:
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if prev_min_val == min_val and prev_max_val == max_val and prev_min_loc == min_loc and prev_max_loc == max_loc:
        break
    else:
        prev_min_val, prev_max_val, prev_min_loc, prev_max_loc = min_val, max_val, min_loc, max_loc
    
    if max_val > threshold:
        start_row = max_loc[1] - h // 2 if max_loc[1] - h // 2 >= 0 else 0
        end_row = max_loc[1] + h // 2 + 1 if max_loc[1] + h // 2 + 1 <= res.shape[0] else res.shape[0]
        start_col = max_loc[0] - w // 2 if max_loc[0] - w // 2 >= 0 else 0
        end_col = max_loc[0] + w // 2 + 1 if max_loc[0] + w // 2 + 1 <= res.shape[1] else res.shape[0]

        res[start_row: end_row, start_col: end_col] = 0
        image = cv2.rectangle(image,(max_loc[0]-10,max_loc[1]-10), (max_loc[0]+w+10, max_loc[1]+h+10), (0,0,0) ,3)
        cv2.putText(image, 'Reis', (max_loc[0]-10,max_loc[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3)



template1 = cv2.imread('C:/Users/Tolga/Desktop/template1.jpg', cv2.IMREAD_COLOR)
template2 = cv2.imread('C:/Users/Tolga/Desktop/template2.jpg', cv2.IMREAD_COLOR)
template3 = cv2.imread('C:/Users/Tolga/Desktop/template3.jpg', cv2.IMREAD_COLOR)
template4 = cv2.imread('C:/Users/Tolga/Desktop/template4.jpg', cv2.IMREAD_COLOR)

res1 = cv2.matchTemplate(image, template1, cv2.TM_CCOEFF_NORMED)
min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
image1 = cv2.rectangle(image,(max_loc1[0],max_loc1[1]), (max_loc1[0]+100, max_loc1[1]+90), (62,104,66) ,5)
cv2.putText(image1, 'Daire', (max_loc1[0],max_loc1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (62,104,66), 3)


res2 = cv2.matchTemplate(image1, template2, cv2.TM_CCOEFF_NORMED)
min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
image2 = cv2.rectangle(image1,(max_loc2[0],max_loc2[1]), (max_loc2[0]+100, max_loc2[1]+90), (179,55,43) ,5)
cv2.putText(image2, 'Ucgen', (max_loc2[0],max_loc2[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (179,55,43), 3)

res3 = cv2.matchTemplate(image2, template3, cv2.TM_CCOEFF_NORMED)
min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
image3 = cv2.rectangle(image2,(max_loc3[0],max_loc3[1]), (max_loc3[0]+100, max_loc3[1]+90), (240,162,2) ,5)
cv2.putText(image3, 'Yildiz', (max_loc3[0],max_loc3[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240,162,2), 3)


res4 = cv2.matchTemplate(image3, template4, cv2.TM_CCOEFF_NORMED)
min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(res4)
image4 = cv2.rectangle(image3,(max_loc4[0],max_loc4[1]), (max_loc4[0]+100, max_loc4[1]+90), (28,113,196) ,5)
cv2.putText(image4, 'Semsiye', (max_loc4[0],max_loc4[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (28,113,196), 3)

    
plt.subplot(2, 2, 1)
plt.title("Daire")
plt.imshow(image1)

plt.subplot(2, 2, 2)
plt.title("Ucgen")
plt.imshow(image2)

plt.subplot(2, 2, 3)
plt.title("Yildiz")
plt.imshow(image3)

plt.subplot(2, 2, 4)
plt.title("Semsiye")
plt.imshow(image4)

plt.show()
