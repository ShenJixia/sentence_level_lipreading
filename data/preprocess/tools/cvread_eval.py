import cv2

image = cv2.imread(r"E:\dataset\GRID\frames\s1\bbaf2n\bbaf2n_frame.jpg")
print(image)


for i in range(9):
    if i==5:
        print('skip')
        continue
    print(i)
