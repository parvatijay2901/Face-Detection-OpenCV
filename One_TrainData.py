#TRAIN DATA

import cv2


#Load HAAR face classifier
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#load funtions
def face_extractor(img):
    #function detects faces and return the cropped face
    #If no face detcted, it returns the input image
        
   # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame)
        
    if faces is ():
        return None
    
    #Crop all faces found
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h,x:x+w]
            
    return cropped_face

num = int(input("Enter the number of users you want to add"))
user = 1;

#Initialize webcam
cap = cv2.VideoCapture(0)
count = 0
print("Start capturing the input data set..")
#Collect 10 samples of your face from webcam input

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        
        #Sacve file in specified directory with unique name
        file_name_path = './faces/user/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path,face)
        
        #Put count on images and dispay live count
        cv2.putText(face,str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Face Cropper", face)
        
        
    else:
        print("Face not found")
        pass

    if (count%25)==0 and count!=num*25 and count!=0:
            print("Place the new user signatur")
            cv2.waitKey()
    key = cv2.waitKey(1)
    if cv2.waitKey(1) == 13 or count == num*25: #13 is the enter key
        break
    
cap.release()
cv2.destroyAllWindows()
print("Collecting samples complete")






























        