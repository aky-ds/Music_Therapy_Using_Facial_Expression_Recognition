# Musical Therapy Using Facial Expression Recognition  

This project aims to utilize facial expression recognition to recommend music that can help alleviate stress or improve mood. By analyzing facial expressions, the system determines the emotional state of the user and plays music accordingly.  

## Features  
- **Facial Expression Detection**: Uses a CNN model trained on the FER dataset to identify emotions.  
- **Music Recommendation**: Matches the detected emotions to a curated playlist of stress-relieving or mood-enhancing tracks.  
- **Interactive Interface**: A user-friendly interface developed using Streamlit for real-time interaction.  
- **Deployment Ready**: Easily deployable on local or cloud environments.  

## Dataset  
The model is trained on the **FER2013 dataset**, which contains grayscale images of facial expressions labeled with emotions like anger, happiness, sadness, and more.  

### Dataset Link  
[FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)  

## Requirements  
Install the necessary Python libraries using the `requirements.txt` file:  
```bash  
pip install -r requirements.txt  
## Run
python app.py
