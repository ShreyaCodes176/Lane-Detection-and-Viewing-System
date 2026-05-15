import cv2
import numpy as np

#finding region of interest
def region_of_interest(img):
    height,width=img.shape
    mask=np.zeros_like(img)

    polygon=np.array([[
        (0,height),
        (width,height),
        (width//2,int(height*0.6))
    ]])

    cv2.fillPoly(mask,polygon,255)
    return cv2.bitwise_and(img,mask)

#function to draw lane lines
def draw_lines(img,lines):
    line_img=np.zeros_like(img)

    if lines is None:
        return img

    left_lines=[]
    right_lines=[]

    for line in lines:
        x1,y1,x2,y2=line[0]

        if x2-x1==0:
            continue

        slope=(y2-y1) / (x2-x1)

        if abs(slope) < 0.5:
            continue

        if slope < 0:
            left_lines.append(line[0])
        else:
            right_lines.append(line[0])

    left_avg=average_line(left_lines,img)
    right_avg=average_line(right_lines,img)

    for line in [left_avg,right_avg]:
        if line is not None:
            x1,y1,x2,y2=line
            cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),10)

    return cv2.addWeighted(img,0.8,line_img,1,1)

def average_line(lines,img):
        if len(lines)==0:
            return None

        x1,y1,x2,y2=np.mean(lines,axis=0)

        slope=(y2-y1) / (x2-x1)
        intercept=y1-slope * x1

        y1_new=img.shape[0]
        y2_new=int(img.shape[0] * 0.6)

        x1_new=int((y1_new-intercept) / slope)
        x2_new=int((y2_new-intercept) / slope)

        return [x1_new,y1_new,x2_new,y2_new]

#Main Pipeline
cap=cv2.VideoCapture(r"D:\internship\pinnacle labs\roadView.mp4")

#Output video
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
out=cv2.VideoWriter("output.mp4",fourcc,20.0,
                      (int(cap.get(3)),int(cap.get(4))))

while cap.isOpened():
    ret,frame=cap.read()

    if not ret:
        break

    #Step 1: Grayscale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Step 2: Blur
    blur=cv2.GaussianBlur(gray,(5,5),0)

    #Step 3: Edge Detection
    edges=cv2.Canny(blur,50,150)

    #Step 4: ROI
    roi=region_of_interest(edges)

    #Step 5: Hough Transform
    lines=cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=100
    )

    #Step 6: Draw lanes
    final=draw_lines(frame,lines)

    #Show
    cv2.imshow("Lane Detection",final)

    #Save output
    out.write(final)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()