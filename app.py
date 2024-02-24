import streamlit as st
import cv2
import mediapipe as mp
import pickle
from utilities import load_model, calculate_angle
import numpy as np
import pandas as pd


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def main():
    
    st.set_page_config(page_title="Body Language Detection", page_icon=':running:', layout='wide')
    
    st.title("Body Language Detector")
    
    
    
    
    
    
    
    
    with st.sidebar:
        st.title("Body Language Detection")
        st.image("https://i0.wp.com/sefiks.com/wp-content/uploads/2022/01/facial-landmarks-mediapipe-scaled.jpg?ssl=1")
        st.info("This app is designed to detect your body language using Mediapipe. It is not a typical object detection algorithm.")
        options = st.selectbox("Choose one of the following models to detect Various things",
                    ['Drowsiness Detection', 'Sign Language Detection', 'Gym Instructor'])
        
        
        
    if options == "Drowsiness Detection":
        st.subheader("This will detect the Drowsiness in the real time using your webcam feed.")
        st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRjNX0W_B6qPYOotN7xa5IZE9_4z3O4wogJ1A&usqp=CAU")

        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("While making detections on your face, these pointers will be detected by the Mediapipe model.")
            st.image("facemeshhero.gif")
        
        with col2:
            st.write("This is your Webcam Feed.")
            
            
            st.write("Click on the Button below to launch your Webcam:")
            submit = st.button("Click here to Start your Webcam Feed")
            
            if submit:
            
                model = load_model("Drowsiness_Detection.pkl")
                
                stframe = st.empty()
            
                cap = cv2.VideoCapture(0)

                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                
                    while cap.isOpened():
                        ret, frame = cap.read()
                        

                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        image.flags.writeable = False        
                        

                        results = holistic.process(image)
                        
                        
                        image.flags.writeable = True   
                        
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        
                        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                                #mp_drawing.DrawingSpec(color=(220,20,60), thickness=1, circle_radius=1),
                                                #mp_drawing.DrawingSpec(color=(240,128,128), thickness=1, circle_radius=1)
                                                #)
                        
                        
                        #mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                #mp_drawing.DrawingSpec(color=(46,139,87), thickness=2, circle_radius=1),
                                                #mp_drawing.DrawingSpec(color=(152,251,152), thickness=2, circle_radius=2)
                                                #)

                        
                        #mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                #mp_drawing.DrawingSpec(color=(46,139,87), thickness=2, circle_radius=1),
                                                #mp_drawing.DrawingSpec(color=(152,251,152), thickness=2, circle_radius=2)
                                                #)

                    
                        #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                                #mp_drawing.DrawingSpec(color=(47,79,79), thickness=2, circle_radius=1),
                                                #mp_drawing.DrawingSpec(color=(224,255,255), thickness=2, circle_radius=2)
                                                #)
                                                
                        # Export coordinates
                        try:
                            # Extract Pose landmarks
                            pose = results.pose_landmarks.landmark
                            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                            
                            # Extract Face landmarks
                            face = results.face_landmarks.landmark
                            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                            
                            # Concate rows
                            row = pose_row+face_row 

                            # Make Detections
                            X = pd.DataFrame([row])
                            body_language_class = model.predict(X)[0]
                            body_language_prob = model.predict_proba(X)[0]
                            print(body_language_class, body_language_prob)
                            
                            # Grabbing the ear coordinates
                            coords = tuple(np.multiply(
                                            np.array(
                                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                        , [640,480]).astype(int))
                            
                            cv2.rectangle(image, 
                                        (coords[0], coords[1]+5), 
                                        (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                        (245, 117, 16), -1)
                            cv2.putText(image, body_language_class, coords, 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            # Getting the box
                            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                            
                            # Display the class name
                            cv2.putText(image, 'CLASS'
                                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(image, body_language_class.split(' ')[0]
                                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            # Display the probability
                            cv2.putText(image, 'PROB'
                                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                        except:
                            pass
                        
                        
                        frame = cv2.resize(image, (0, 0), fx = 0.8, fy = 0.8)
                        stframe.image(frame, channels = "BGR", use_column_width=True)
            
            
            
            

            st.info("This model is predicting the class based on the Coordinates of your facial features.")
            st.info("It is repeatedly extracting those Coordinates and predicting upon them.")
            
        
        
        
        
        
    if options == 'Sign Language Detection':
        
        st.subheader("This is trained to Detect your hand movements and predict based on that.")
        st.sidebar.image("hand_detections.jpg")
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("These are the pointers that Mediapipe model will detect in your hands.")
            st.image("https://3.bp.blogspot.com/-CWTYSEEB3mA/XmfimK9wP1I/AAAAAAAAC0E/wIvHQktx8IEbeB_vbtIEZt3VFNayIFzRACLcBGAsYHQ/s1600/hand_trimmed.gif")
            
        
        
        with col2:
            st.write("This is your Webcam Feed.")            
            
            st.write("Click on the Button below to launch your Webcam:")
            submit = st.button("Click here to Start your Webcam Feed")
                        
            if submit:
                
                stframe1 = st.empty()

                sign_language = load_model("body_language.pkl")

                
                FaceVideo = cv2.VideoCapture(0)

                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                
                    while FaceVideo.isOpened():
                        ret, frame = FaceVideo.read()
                        

                        image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        image1.flags.writeable = False        
                        

                        results = holistic.process(image1)
                        
                        
                        image1.flags.writeable = True   
                        
                        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
                        
                        
                        mp_drawing.draw_landmarks(image1, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                                mp_drawing.DrawingSpec(color=(220,20,60), thickness=1, circle_radius=1),
                                                mp_drawing.DrawingSpec(color=(240,128,128), thickness=1, circle_radius=1)
                                                )
                        
                        
                        #mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                #mp_drawing.DrawingSpec(color=(46,139,87), thickness=2, circle_radius=1),
                                                #mp_drawing.DrawingSpec(color=(152,251,152), thickness=2, circle_radius=2)
                                                #)

                        
                        #mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                #mp_drawing.DrawingSpec(color=(46,139,87), thickness=2, circle_radius=1),
                                                #mp_drawing.DrawingSpec(color=(152,251,152), thickness=2, circle_radius=2)
                                                #)

                    
                        #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                                #mp_drawing.DrawingSpec(color=(47,79,79), thickness=2, circle_radius=1),
                                                #mp_drawing.DrawingSpec(color=(224,255,255), thickness=2, circle_radius=2)
                                                #)
                                                
                        # Export coordinates
                        try:
                            # Extract Pose landmarks
                            pose1 = results.pose_landmarks.landmark
                            pose_row1 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose1]).flatten())
                            
                            # Extract Face landmarks
                            face1 = results.face_landmarks.landmark
                            face_row1 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face1]).flatten())
                            
                            # Concate rows
                            row1 = pose_row1+face_row1

                            # Make Detections
                            X1 = pd.DataFrame([row1])
                            body_language_class1 = sign_language.predict(X1)[0]
                            body_language_prob1 = sign_language.predict_proba(X1)[0]
                            print(body_language_class1, body_language_prob1)
                            
                            # Grabbing the ear coordinates
                            coords1 = tuple(np.multiply(
                                            np.array(
                                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                        , [640,480]).astype(int))
                            
                            cv2.rectangle(image1, 
                                        (coords1[0], coords1[1]+5), 
                                        (coords1[0]+len(body_language_class1)*20, coords1[1]-30), 
                                        (245, 117, 16), -1)
                            cv2.putText(image1, body_language_class1, coords1, 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            # Getting the box
                            cv2.rectangle(image1, (0,0), (250, 60), (245, 117, 16), -1)
                            
                            # Display the class name
                            cv2.putText(image1, 'CLASS'
                                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(image1, body_language_class1.split(' ')[0]
                                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            # Display the probability
                            cv2.putText(image1, 'PROB'
                                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(image1, str(round(body_language_prob1[np.argmax(body_language_prob1)],2))
                                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                        except:
                            pass
                        
                        
                        frame1 = cv2.resize(image1, (0, 0), fx = 0.8, fy = 0.8)
                        stframe1.image(image1, channels = "BGR", use_column_width=True)
        

        
        
        
    
    
    if options == 'Gym Instructor':
        
        st.subheader("This will try to count the number of REPS you are doing.")
        st.sidebar.info("This is made to count the number of the REPS you are doing based on the angle calculated b/w your shoulder, elbow and wrist of your left hand.")
        

        col1, col2 = st.columns(2)
        
        with col1:
            st.info("This will detect the number of reps based on the calculated angle.")
            st.image("https://1.bp.blogspot.com/-nsLiFUVt6S4/XzVpLWay6VI/AAAAAAAAGXI/oPyuvuQEFcASODqPdT9dqptyUvUuGlTvACLcBGAsYHQ/s427/image3.gif")
           
        
        with col2:
            
            submit = st.button("Click here to start your Webcam feed")
            
            if submit:
            
                stframe2 = st.empty()
                cap = cv2.VideoCapture(0)
        
                counter = 0
                shape = None
                
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                
                    while cap.isOpened():
                        ret, frame = cap.read()
                        
                        # Recolor image to RGB
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                    
                        # Make detection
                        results = pose.process(image)
                    
                        # Recolor back to BGR
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        # Extract landmarks
                        try:
                            landmarks = results.pose_landmarks.landmark
                            
                            # Get coordinates
                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                            
                            # Calculate angle
                            angle = calculate_angle(shoulder, elbow, wrist)
                            
                            # Visualize angle
                            cv2.putText(image, str(angle), 
                                        tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )
                            
                            #showing the box
                            if angle > 160:
                                shape = 'Down'
                            
                            if angle < 30 and shape == 'Down':
                                counter += 1
                                shape = 'Up'
                                print(counter)
                                
                                    
                        except:
                            pass
                        
                        
                    # Render curl counter
                        # Setup status box
                        cv2.rectangle(image, (0,0), (255,73), (245,117,16), -1)
                    
                        # Rep data
                        cv2.putText(image, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                        # Stage data
                        cv2.putText(image, 'STAGE', (95,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, shape, 
                                (95,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    
                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )
                        
                        
                        
                        frame = cv2.resize(image, (0, 0), fx = 0.8, fy = 0.8)
                        stframe2.image(image, channels = "BGR", use_column_width=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
