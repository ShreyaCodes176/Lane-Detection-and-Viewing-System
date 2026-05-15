import cv2
import numpy as np

def find_lane_pixels(binary):
    histogram=np.sum(binary[binary.shape[0]//2:, :], axis=0)
    #sum histogram technique to find peaks corresponding to lane lines
    histogram = np.convolve(histogram, np.ones(50)/50, mode='same')
    #find lane pixels
    midpoint=int(histogram.shape[0] // 2)
    leftx_base=np.argmax(histogram[:midpoint])
    rightx_base=np.argmax(histogram[midpoint:]) + midpoint

    return leftx_base, rightx_base

#sliding window technique to find lane pixels
def sliding_window(binary, leftx_base, rightx_base):
    n_windows=9
    window_height=int(binary.shape[0] / n_windows)

    nonzero=binary.nonzero()
    nonzeroy=np.array(nonzero[0])
    nonzerox=np.array(nonzero[1])

    margin=30
    minpix=20

    leftx_curr=leftx_base
    rightx_curr=rightx_base

    left_lane_inds=[]
    right_lane_inds=[]

    for window in range(n_windows):
        win_y_low=binary.shape[0]-(window + 1) * window_height
        win_y_high=binary.shape[0]-window * window_height

        win_xleft_low=leftx_curr-margin
        win_xleft_high=leftx_curr + margin

        win_xright_low=rightx_curr-margin
        win_xright_high=rightx_curr + margin

        good_left_inds=((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds=((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds)>minpix:
            leftx_curr=int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds)>minpix:
            rightx_curr=int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds=np.concatenate(left_lane_inds)
    right_lane_inds=np.concatenate(right_lane_inds)

    return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], \
           nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

def fit_polynomial(leftx, lefty, rightx, righty):
    left_fit=np.polyfit(lefty, leftx, 2)
    right_fit=np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

#function to draw lane lines
def draw_lane(org_img, binary, left_fit, right_fit, inv_matrix):
    ploty=np.linspace(0, binary.shape[0]-1, binary.shape[0])

    left_fitx=left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx=right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    lane_img=np.zeros_like(org_img)

    pts_left=np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right=np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    pts=np.hstack((pts_left, pts_right))

    cv2.polylines(
    lane_img,
    [np.int32(pts_left)],
    isClosed=False,
    color=(255,0,0),
    thickness=15)

    cv2.polylines(
    lane_img,
    [np.int32(pts_right)],
    isClosed=False,
    color=(0,0,255),
    thickness=15)

    cv2.fillPoly(lane_img, np.int_([pts]), (0, 255, 0))
    unwarp=cv2.warpPerspective(lane_img, inv_matrix, (org_img.shape[1], org_img.shape[0]))
    result=cv2.addWeighted(org_img, 1, unwarp, 0.3, 0)
    
    return result

def smooth_fit(fit_history, current_fit, history_size=10):

    fit_history.append(current_fit)

    if len(fit_history) > history_size:
        fit_history.pop(0)

    smoothed_fit = np.mean(fit_history, axis=0)

    return smoothed_fit

# Main pipeline
cap=cv2.VideoCapture(r"D:\internship\pinnacle labs\roadView.mp4")

mode="lane" #default mode
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #width
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
output=cv2.VideoWriter("curved_output.mp4", fourcc, 20.0, (w, h))

src=np.float32([[w*0.54, h*0.62],[w*0.66, h*0.6],[w*1,h],[w*0.3,h]])
dst=np.float32([[w*0.2,0],[w*0.8,0],[w*0.8,h],[w*0.2,h]])
# Store previous lane fits
left_fit_history = []
right_fit_history = []
history_size = 10

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break
    
    #step 1: perspective transform
    matrix = cv2.getPerspectiveTransform(src, dst)
    inv_matrix = cv2.getPerspectiveTransform(dst, src)
    warped=cv2.warpPerspective(frame, matrix, (w, h),flags=cv2.INTER_LINEAR)    
    #inverse perspective transform matrix(src becomes dst)

    #Step 2: grayscale and blur and adaptive thresholding
    hls=cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lower_white=np.array([0, 200, 0])
    upper_white=np.array([255, 255, 255])
    mask=cv2.inRange(hls, lower_white, upper_white)
    #blur = cv2.GaussianBlur(mask, (5,5), 0)
    binary=cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    #Step 3: Lane Detection using histogram and sliding window
    leftx_base, rightx_base=find_lane_pixels(binary)
    leftx, lefty, rightx, righty=sliding_window(binary, leftx_base, rightx_base)

    #Step 5: Fit polynomial to lane lines
    if len(leftx)>0 and len(rightx)>0:
        left_fit, right_fit=fit_polynomial(leftx, lefty, rightx, righty)
        left_fit = smooth_fit(left_fit_history, left_fit, history_size)
        right_fit = smooth_fit(right_fit_history, right_fit, history_size)
        final=draw_lane(frame, binary, left_fit, right_fit, inv_matrix)
    else:
        final=frame


    #display output according to mode
    if mode == "bird":
        cv2.imshow("Output", warped)

    elif mode == "lane":
        cv2.imshow("Output", final)
    #Saving output
    output.write(final)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('b'):
        mode = "bird"
    elif key == ord('l'):
        mode = "lane"
    elif key == ord('q'):
        break

#releasing resources
cap.release()
output.release()
cv2.destroyAllWindows()