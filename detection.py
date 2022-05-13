from scipy.spatial import distance
from imutils import face_utils 
from fuzzy_logic import FatigueSys
import dlib
import cv2
import matplotlib.pyplot as plt


def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

def calculate_MAR(mouth):
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    D = distance.euclidean(mouth[12], mouth[16])

    mouth_aspect_ratio = (A + B + C) / (2.0 * D)
    return mouth_aspect_ratio

def drive_process():
    process(alert=True)

def test_process():
    process(alert=False)

def process(alert): 
    # all eye and mouth aspect ratio
    ear_list=[]
    mar_list=[]

    # Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink 
    EAR_THRESHOLD = 0.20
    # declare another costant to hold the consecutive number of frames to consider for a driver state (3 sec) (20?)
    CONSECUTIVE_FRAMES = 120 
    # Another constant which will work as a threshold for MAR value
    MAR_THRESHOLD = 0.9

    # initialize camera
    cam = cv2.VideoCapture(0)

    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # initialize fuzzy control system
    fatigue_sys = FatigueSys()
    # driver states computed from EAR and MAR by fuzzy logic system
    calc_states = []

    while True:
        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces 
        faces = hog_face_detector(gray)

        for face in faces:
            face_landmarks = dlib_facelandmark(gray, face)

            # convert it to a (68, 2) size numpy array 
            shape = face_utils.shape_to_np(face_landmarks)

            # draw a rectangle over the detected face 
            (x, y, w, h) = face_utils.rect_to_bb(face) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

            # put a description 
            cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # left eye detection
            leftEye = shape[42:48]
            leftEyeHull = cv2.convexHull(leftEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)

            # right eye detection
            rightEye = shape[36:42]
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear + right_ear) / 2
            EAR = round(EAR, 6)
            ear_list.append(EAR)

            # mouth detection
            mouth = shape[48:68]
            cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1) 
            MAR = calculate_MAR(mouth)
            MAR = round(MAR, 6)
            mar_list.append(MAR)

            cv2.rectangle(frame, (0, 430), (640, 480), (195, 217, 189), -1)
            cv2.rectangle(frame, (0, 420), (120, 440), (82, 64, 46), -1)
            cv2.putText(frame, "Messages", (5, 435), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
            cv2.putText(frame, "Press any key to exit", (5, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # compute numerical driver state
            calculated_state = fatigue_sys.compute_inference(EAR, MAR)
            calc_states.append(calculated_state)
            # interpreted numerical driver states ('fatigued', 'sluggish' or 'wakeful') 
            interpreted_states = []

            # get results only for some last frames
            if len(calc_states) > CONSECUTIVE_FRAMES:
                # interpret states for some last frames
                for state in calc_states[-CONSECUTIVE_FRAMES:]:
                    interpreted_state = fatigue_sys.interpret_membership(state)
                    interpreted_states.append(interpreted_state)

                # get the most common state for some last frames
                common_state = max(set(interpreted_states), key=interpreted_states.count)


                if common_state == 'fatigued':
                    msg = "Danger. You may have an accident. Please stop car immediately"
                    # alert and say 'Danger. You may have an accident. Please stop car immediately'
                    if (alert):
                        

                if common_state == 'sluggish':
                    msg = "There is loss of attention. Please have a rest"
                    # say 'There is loss of attention. Please have a rest'
                
                cv2.rectangle(frame, (0, 430), (640, 480), (195, 217, 189), -1)
                cv2.rectangle(frame, (0, 420), (120, 440), (82, 64, 46), -1)
                cv2.putText(frame, "Messages", (5, 435), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
                cv2.putText(frame, msg, (5, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # check Eye Aspect Ratio for closing the eye
            if EAR < EAR_THRESHOLD:
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            # check Mouth Aspect Ratio for closing the mouth
            if MAR > MAR_THRESHOLD:
                cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 

        cv2.imshow("Fatigue Detection System", frame)

        key = cv2.waitKey(1)
        if key != -1:  
            cam.release()
            cv2.destroyAllWindows()
            break

    plt.figure()
    plt.plot(ear_list)
    # plt.subplots_adjust(bottom=0.30)
    plt.title("EAR calculation")
    plt.ylabel('EAR')
    #plt.gca().axes.get_xaxis().set_visible(False)

    plt.figure()
    plt.plot(mar_list)
    plt.title("MAR calculation")
    plt.ylabel("MAR")

    plt.figure()
    plt.plot(calc_states)
    plt.title("Driver State calculation")
    plt.ylabel("Driver State")

    plt.show()
