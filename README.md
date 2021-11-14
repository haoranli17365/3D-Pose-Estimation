# OpenPose2D + 3DPoseEstimation


### Requirements 
```
pip install opencv-python matplotlib
```
### Sample Run

Under 3DPoseEstimation directory, run below:
```bash
python run_combine.py \
    --model2D ./model/OpenPose_for_TensorFlow_BatchSize_1.om \
    --model3D ./model/video_pose_3d.om \
    --input ./data/[INPUT_VIDEO] \
    --output_dir='./outputs'
```

### Demonstration outputs
Demonstration outputs are in the `outputs` folder.

### Implementation Detail
To overcome the multiple objects being detected in one frame, I implemented a voting system which will find the accumulated max score of all body parts in order to vote for the best `Human` object to iterate in `draw()` function in run_openpose_tf.py. Below is the implementeation detail:
```python
if len(humans) >= 2: # if appears to have detected multiple human in the current frame.
        # take the longest keypoint list
        max_score = 0 # global max
        store = [None]
        for human in humans:
            score = 0 # local max
            for key, body_part in human.body_parts.items():
                score += body_part.score # add up all score of current human
            if max_score <= score:
                max_score = max(max_score, score) # update human score
                store[0] = human # update human object

        humans = store

keypoints = draw(img, humans) # call the draw function
```

Another challenge is the missing keypoints in single frame. To accomplish this task, I used polynomial fitting function from numpy module after I collected all the keypoints from every frames, then gathered 60 corresponding keypoints from adjacent frames of the missing point frame and called `np.polyfit()` afterwards. See below for implementation details:

```python
MISSING_RANGE = 30 # the collecting range.
for key, val in missing_dict.items():
    cnt += 1; print(end=f"\rFitting missing points with Polynomial Fitting: {cnt} / {total_missing}")
    for pt in val:
        m_x, m_y = [], [] # collect all x and y values for y-input for polynomial fitting
        idx_list_x, idx_list_y = [], [] # collect all frame numbers as x-input for polynomial fitting
        for idx in range(1, MISSING_RANGE+1): 
            # check key still in-bound
            if (key - idx) >= 0 and keypoints[key-idx][pt][0] != 0.0:
                m_x.append(keypoints[key-idx][pt][0])
                idx_list_x.append(key-idx)

            if (key - idx) >= 0 and keypoints[key-idx][pt][1] != 0.0:
                m_y.append(keypoints[key-idx][pt][1])
                idx_list_y.append(key-idx)
                
            if (key + idx) < total_frames and keypoints[key+idx][pt][0] != 0.0:
                m_x.append(keypoints[key+idx][pt][0])
                idx_list_x.append(key+idx)

            if (key + idx) < total_frames and keypoints[key+idx][pt][1] != 0.0:
                m_y.append(keypoints[key+idx][pt][1])
                idx_list_y.append(key+idx)
        
        # fill missing points with polynomial fitting
        poly_x = np.polyfit(idx_list_x, m_x, 2)
        poly_y = np.polyfit(idx_list_y, m_y, 2)
        # forming the polynomial function and given the "key" as the x-input
        fit_eq_x = poly_x[0] * np.square(key) + poly_x[1] * key + poly_x[2]
        fit_eq_y = poly_y[0] * np.square(key) + poly_y[1] * key + poly_y[2]
        # assign to the missing points
        keypoints[key][pt][0] = fit_eq_x
        keypoints[key][pt][1] = fit_eq_y

```

For detecting the traffic signals, I basically detect the depth(z-index of the left hip and right hip), compare the average difference of z-index between left hip and right hip will tell which signal is performed:


```python
def detect_pose_signal(output):
    # H36M Order
    # PELVIS = 0, R_HIP = 1, R_KNEE = 2, R_FOOT = 3, L_HIP = 4, L_KNEE = 5, 
    # L_FOOT = 6, SPINE = 7, THORAX = 8, NOSE = 9, HEAD = 10, L_SHOULDER = 11, 
    # L_ELBOW = 12, L_WRIST = 13, R_SHOULDER = 14, R_ELBOW = 15, R_WRIST = 16
    l_hip = 0.0
    r_hip = 0.0
    # compare z-index depth for the left hip and right hip
    for idx in range(len(output)):
        l_hip += output[idx][1][2]
        r_hip += output[idx][4][2]

    l_hip_avg = np.mean(l_hip)
    r_hip_avg = np.mean(r_hip)
    depth_diff = abs(l_hip_avg - r_hip_avg)
    # left: 12.580614662221631
    # stop: 2.391942911250253
    # right: 8.282248636889676
    if depth_diff >= 10:
        print('######################################')
        print("left-turn traffic signal is performed.")
        print('######################################')
    elif depth_diff >= 8:
        # arm raising causing the y to decrease as normal.
        print('######################################')
        print("Right-turn traffic signal is performed.")
        print('######################################')
    else:
        print('######################################')
        print("Stop traffic signal is performed.")
        print('######################################')


```