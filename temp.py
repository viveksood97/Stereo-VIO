import cv2
import pykitti
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt

basedir = './'

date = '2011_09_30'
drive = '0018'
dataset = pykitti.raw(basedir, date, drive, imformat='cv2')

class Debugger:

    def __init__(self,size):
        self.stack = []
        self.type = []
        self.nrows = size[0]
        self.ncols = size[1]
    
    def collect(self,frame,frame_type):
        self.stack.append(frame)
        self.type.append(frame_type)

    def display(self,plot=False):
        if plot == True:
            for index in range(0,len(self.stack)):
                if self.type[index][0] == "image":
                    ax = plt.subplot(self.nrows,self.ncols,index+1)
                    ax.imshow(cv2.cvtColor(self.stack[index], cv2.COLOR_BGR2RGB), cmap='gray')
                elif self.type[index][0] == "plot":
                    ax = plt.subplot(self.nrows,self.ncols,index+1)
                    ax.plot(*zip(*self.stack[index]))
                elif self.type[index][0] == "scatter":
                    ax = plt.subplot(self.nrows,self.ncols,index+1)
                    ax.scatter(*zip(*self.stack[index]),linewidths=0.1)

                
            plt.tight_layout()
            plt.show()
        else:
            stack = list()
            temp = list()
            for index,image in enumerate(self.stack):
                if(self.type[index][1] == "binary"):
                    temp.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
                else:
                    temp.append(image)
                if len(temp) == 3:
                    stack.append(np.hstack((temp[0],temp[1],temp[2])))
                    temp = list()
                
            stacked = np.vstack(tuple(stack))
            cv2.imshow('Debugger',cv2.resize(stacked,None,fx=0.4,fy=0.4))


def compute_left_disparity_map(img_left, img_right, matcher='bm', rgb=False, verbose=False):
    '''
    Takes a left and right stereo pair of images and computes the disparity map for the left
    image. Pass rgb=True if the images are RGB.
    
    Arguments:
    img_left -- image from left camera
    img_right -- image from right camera
    
    Optional Arguments:
    matcher -- (str) can be 'bm' for StereoBM or 'sgbm' for StereoSGBM matching
    rgb -- (bool) set to True if passing RGB images as input
    verbose -- (bool) set to True to report matching type and time to compute
    
    Returns:
    disp_left -- disparity map for the left camera image
    
    '''
    # Feel free to read OpenCV documentation and tweak these values. These work well
    sad_window = 6
    num_disparities = sad_window*16
    block_size = 11
    matcher_name = matcher
    
    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size
                                     )
        
    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 3 * sad_window ** 2,
                                        P2 = 32 * 3 * sad_window ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                       )
    if rgb:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
    # plt.imshow(disp_left)
    # plt.show()
    return disp_left

def decompose_projection_matrix(p):
    '''
    Shortcut to use cv2.decomposeProjectionMatrix(), which only returns k, r, t, and divides
    t by the scale, then returns it as a vector with shape (3,) (non-homogeneous)
    
    Arguments:
    p -- projection matrix to be decomposed
    
    Returns:
    k, r, t -- intrinsic matrix, rotation matrix, and 3D translation vector
    
    '''
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    
    return k, r, t

def calc_depth_map(disp_left, k_left, t_left, t_right, rectified=True):
    '''
    Calculate depth map using a disparity map, intrinsic camera matrix, and translation vectors
    from camera extrinsic matrices (to calculate baseline). Note that default behavior is for
    rectified projection matrix for right camera. If using a regular projection matrix, pass
    rectified=False to avoid issues.
    
    Arguments:
    disp_left -- disparity map of left camera
    k_left -- intrinsic matrix for left camera
    t_left -- translation vector for left camera
    t_right -- translation vector for right camera
    
    Optional Arguments:
    rectified -- (bool) set to False if t_right is not from rectified projection matrix
    
    Returns:
    depth_map -- calculated depth map for left camera
    
    '''
    # Get focal length of x axis for left camera
    f = k_left[0][0]
    
    # Calculate baseline of stereo pair
    if rectified:
        b = t_right[0] - t_left[0] 
    else:
        b = t_left[0] - t_right[0]
        
    # Avoid instability and division by zero
    disp_left[disp_left == 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1
    
    # Make empty depth map then fill with depth
    depth_map = np.ones(disp_left.shape)
    depth_map = f * b / disp_left

    # plt.imshow(depth_map)
    # plt.show()
    
    return depth_map

def extract_features(image, detector='sift', mask=None):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()
    elif detector == 'surf':
        det = cv2.xfeatures2d.SURF_create()
        
    kp, des = det.detectAndCompute(image, mask)
    
    return kp, des

def match_features(des1, des2, matching='BF', detector='sift', sort=True, k=2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image
    matching -- (str) can be 'BF' for Brute Force or 'FLANN'
    detector -- (str) can be 'sift or 'orb'. Default is 'sift'
    sort -- (bool) whether to sort matches by distance. Default is True
    k -- (int) number of neighbors to match to each feature.

    Returns:
    matches -- list of matched features from two images. Each match[i] is k or less matches for 
               the same query descriptor
    """
    if matching == 'BF':
        if detector == 'sift':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        elif detector == 'orb':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=k)
    elif matching == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=k)
    
    if sort:
        matches = sorted(matches, key = lambda x:x[0].distance)

    return matches

def filter_matches_distance(matches, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    for m, n in matches:
        if m.distance <= dist_threshold*n.distance:
            filtered_match.append(m)

    return filtered_match

def estimate_motion(match, kp1, kp2, k, depth1=None, max_depth=3000):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera intrinsic calibration matrix 
    
    Optional arguments:
    depth1 -- Depth map of the first frame. Set to None to use Essential Matrix decomposition
    max_depth -- Threshold of depth to ignore matched features. 3000 is default

    Returns:
    rmat -- estimated 3x3 rotation matrix
    tvec -- estimated 3x1 translation vector
    image1_points -- matched feature pixel coordinates in the first image. 
                     image1_points[i] = [u, v] -> pixel coordinates of i-th match
    image2_points -- matched feature pixel coordinates in the second image. 
                     image2_points[i] = [u, v] -> pixel coordinates of i-th match
               
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))

    image1_points = []
    image2_points = []

    # for i,(m,n) in enumerate(match):
    #     image1_points.append(kp1[m.queryIdx].pt)
    #     image2_points.append(kp2[m.trainIdx].pt)

    # image1_points = np.float32(image1_points)
    # image2_points = np.float32(image2_points)

    image1_points = np.float32([kp1[m.queryIdx].pt for m in match])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in match])

    if depth1 is not None:
        cx = k[0, 2]
        cy = k[1, 2]
        fx = k[0, 0]
        fy = k[1, 1]
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for i, (u, v) in enumerate(image1_points):
            z = depth1[int(v), int(u)]
            # If the depth at the position of our matched feature is above 3000, then we
            # ignore this feature because we don't actually know the depth and it will throw
            # our calculations off. We add its index to a list of coordinates to delete from our
            # keypoint lists, and continue the loop. After the loop, we remove these indices
            if z > max_depth:
                delete.append(i)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC for robustness to outliers
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)
        #print('Number of inliers: {}/{} matched features'.format(len(inliers), len(match)))
        
        # Above function returns axis angle rotation representation rvec, use Rodrigues formula
        # to convert this to our desired format of a 3x3 rotation matrix
        rmat = cv2.Rodrigues(rvec)[0]
    
    else:
        # With no depth provided, use essential matrix decomposition instead. This is not really
        # very useful, since you will get a 3D motion tracking but the scale will be ambiguous
        image1_points_hom = np.hstack([image1_points, np.ones(len(image1_points)).reshape(-1,1)])
        image2_points_hom = np.hstack([image2_points, np.ones(len(image2_points)).reshape(-1,1)])
        E = cv2.findEssentialMat(image1_points, image2_points, k)[0]
        _, rmat, tvec, mask = cv2.recoverPose(E, image1_points, image2_points, k)
    
    return rmat, tvec, image1_points, image2_points

def cartesian(longitude,latitude, elevation):
    R = 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(longitude * math.pi / 180) * math.sin(latitude * math.pi / 180)
    Y = R * math.sin(longitude * math.pi / 180) * math.sin(latitude * math.pi / 180)
    Z = R * math.cos(latitude * math.pi / 180)
    return X,Y,Z

def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)
    # plt.show()

nframes = len(dataset.cam0_files)
# nframes = 1000

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot()
# ax = fig.add_subplot(projection='3d')
# ax.view_init(elev=-20, azim=270)
trajectory = np.zeros((nframes+2, 3, 4))
f = np.array(dataset.get_gray(0)[0])
mask = np.zeros(f.shape[:2], dtype=np.uint8)
# mask = cv2.rectangle(mask, (96, 0), (xmax, ymax), (255), thickness=-1)

k_left, r_left, t_left = decompose_projection_matrix(dataset.calib.P_rect_00)
k_right, r_right, t_right = decompose_projection_matrix(dataset.calib.P_rect_10)

for i in range(nframes-1):
    # print(i)
    # first_gray = dataset.get_gray(i)
    # second_gray = dataset.get_gray(i+1)

    # disp = compute_left_disparity_map(np.array(first_gray[0]),np.array(first_gray[1]), 
    #                                 matcher='bm',
    #                                 verbose=False)

    # # print(dataset.calib.P_rect_00)

    # depth = calc_depth_map(disp, k_left, t_left, t_right)
    
    # for j, pixel in enumerate(depth[0]):
    #     if pixel < depth.max():
    #         print('First non-max value at index', j)
    #         break

    # mask = cv2.rectangle(mask, (96, 0), (depth.shape[0], depth.shape[1]), (255), thickness=-1)

    # kp0, des0 = extract_features(np.array(first_gray[0]), 'sift', mask)
    # kp1, des1 = extract_features(np.array(second_gray[0]), 'sift', mask)

    # matches_unfilt = match_features(des0, des1, matching='BF', detector='sift', sort=True)

    # matches_unfilt = filter_matches_distance(matches_unfilt, 0.5)

    # # visualize_matches(np.array(first_gray[0]),kp0, np.array(second_gray[0]), kp1, matches_unfilt)

    # rmat, tvec, img1_points, img2_points = estimate_motion(matches_unfilt, kp0, kp1, k_left, depth)

    # Tmat = np.eye(4)
    # Tmat[:3, :3] = rmat
    # Tmat[:3, 3] = tvec.T

    # T_tot = np.eye(4)

    # T_tot = T_tot.dot(np.linalg.inv(Tmat))
    
    # trajectory[i+1, :, :] = T_tot[:3, :]

    # xs = trajectory[:i+2, 0, 3]
    # ys = trajectory[:i+2, 1, 3]
    # zs = trajectory[:i+2, 2, 3]

    lat1 = dataset.oxts[i][0][0]
    # lat2 = dataset.oxts[1][0][0]

    long1 = dataset.oxts[i][0][1]
    # long2 = dataset.oxts[1][0][1]

    ele1 = dataset.oxts[i][0][2]
    # ele2 = dataset.oxts[1][0][2]

    x1,y1,z1 = cartesian(lat1,long1,ele1)
    # x2,y2,z2 = cartesian(lat2,long2,ele2)
    
    # ax.scatter(x1, y1, z1, c='chartreuse')
    ax.scatter(x1,y1,c='chartreuse')
    plt.pause(1e-320)

    # cv2.imshow("gray1",np.array(first_gray[0]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(xs,x2-x1)
    # print(ys,y2-y1)
    # print(zs,z2-z1)
    # dataset.oxts[0].packet.lat 
# ax.plot(trajectory[:, :, 3][:, 0], 
#         trajectory[:, :, 3][:, 1], 
#         trajectory[:, :, 3][:, 2], label='estimated', color='orange')

# ax.view_init(elev=-20, azim=270)
plt.show()