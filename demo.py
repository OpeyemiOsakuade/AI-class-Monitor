import streamlit as st
from fastai.vision.all import * 
from fastai.metrics import error_rate, accuracy
import pathlib
import torch
import io
import tempfile 
import cv2
import os
import ffmpeg
import subprocess
# Instantiate the PosixPath. 
# N/B: PosixPath to Linux as WindowsPath to Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# load AI Class Monitor model
model = load_learner("export.pkl")


def scores(value):
    attentiveness_score = 0    
    for i in value[0]: 
        if i == 'bending':
            attentiveness_score +=0.3
        elif i == 'chatting':
            attentiveness_score+=0.3
        elif i == 'raising hand':
            attentiveness_score+=0.9
        elif i == 'sitting':
            attentiveness_score+=0.5
        elif i == 'standing':
            attentiveness_score+=0.5
        else:
            attentiveness_score+=0.8
    avgScore = round(attentiveness_score/len(value[0]), 1)
    return avgScore

def main():
    st.title("AI Class Monitor")
    
    st.text("Check the attentiveness score of students in a classroom by uploading a class video/image")
    text_option = st.radio("Select an option", ("Upload Video (.mp4)", "Upload Image (.jpg/png)"))
    
    if text_option == "Upload Video (.mp4)":
        file = st.file_uploader(label="Upload a classroom video", type="mp4")
        
        if file is not None:
            if ".mp4" in file.name.lower():
        
                file_details = {"File name": file.name, "File type": file.type}
                st.write("File Uploaded", file_details)
                tFile = tempfile.NamedTemporaryFile(delete=False) 
                tFile.write(file.read())
                videoFile = cv2.VideoCapture(tFile.name)

            else:
                st.write("Cannot read file. Accepted formats are .mp4, .jpg and .png")
        
            st.write("Cutting video into frames. Please wait...")
            success,image = videoFile.read()
            count = 0
            while success:
                cv2.imwrite("frame%d.png" % count, image)         
                success,image = videoFile.read()
                count += 1
            st.write("Calculating attentiveness scores...") 
        else:
            pass
        dataframe = []    
        for videoFrame in glob.iglob("./*.png"):
            vidValue = model.predict(videoFrame)    
            avgSc = scores(vidValue)
            dataframe.append([videoFrame, vidValue[0], avgSc])
        attentivenessDf = pd.DataFrame(dataframe, \
            columns=["frame", "predicted_classes", "attentiveness_score"])

        st.write(attentivenessDf)

        try:
            overall = attentivenessDf["attentiveness_score"].sum()/len(attentivenessDf)
            result = round(overall, 3)
            st.write("The overall attentiveness score of this class is: ", result)
        except ZeroDivisionError:
            pass
            #st.exception(ZeroDivisionError("This is an error"))
        

    elif text_option == "Upload Image (.jpg/png)":
        file = st.file_uploader(label="Upload a classroom image", type=["jpg", "png"])
        
        if file is not None:
            if ".jpg" in file.name.lower() or ".png" in file.name.lower():
                file_details = {"File name": file.name, "File type": file.type}
                st.write("File Uploaded", file_details)
            
            else:
                st.write("Cannot read file. Accepted formats are .mp4, .jpg and .png") 
            imValue = model.predict(file.name)
            avg = scores(imValue)
            st.write("The attententiveness score of this image is: ", avg)
    else:
        pass


if __name__ == '__main__':    
    main()