🚀 Real-Time Facial Emotion Detection: 

📄 Overview:

This project focuses on real-time facial emotion recognition using a webcam feed. The model is trained on the RAF-DB dataset and leverages transfer learning (ResNet18 backbone) to classify facial expressions into 7 different emotions with high accuracy.

😄 Emotions Detected: 

Surprise

Fear

Disgust

Happy

Sad

Angry

Neutral

💡 Features: 
Real-time emotion detection from webcam

High accuracy (~92% on test set)

Transfer learning-based model (ResNet18)

Easily extendable to other datasets and emotions



⚙️ Installation: 

bash
Copy
Edit
git clone https://github.com/Rajverma30/real-time-facial-emotion-detection.git
cd real-time-facial-emotion-detection
pip install -r requirements/requirements.txt
🏃‍♂️ Running Real-Time Detection
bash
Copy
Edit
python src/main.py
👀 Your webcam will open, and emotion predictions will appear live on the video feed.



🧑‍💻 Training:

You can retrain or fine-tune the model using the notebook:

bash
Copy
Edit
jupyter notebook Notebook/Model_Training.ipynb


🎯 Future Work:

Add more emotion classes (e.g., contempt, excitement)

Support for multi-face detection in the same frame

Integration with audio (paralanguage) emotion cues

🙏 Credits:

Dataset: RAF-DB (Real-world Affective Faces Database)

Libraries: PyTorch, OpenCV, torchvision, PIL

🌟 Let's Connect: 
Feel free to connect on www.linkedin.com/in/raj-verma-485900255 or open an issue to discuss improvements or collaborations.
