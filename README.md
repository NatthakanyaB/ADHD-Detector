# AiBxDarun Image Classification: ADHD-Detector
The Image classification model that detects ADHD in the fMRI picture of the brain.
# Feature of this model
This model can predict the difference between the ADHD brain and the normal brain.
# Dataset that used to train this model
- ADHD brain fMRI data:https://fcon_1000.projects.nitrc.org/indi/adhd200/
- Normal brain fMRI data:https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network 
- Data from each website were collected for the dataset, then separate data into 50% to the training dataset, 25% to the testing set, and 25% to the validation set.
# About the model
- resnet50 model
- Using the FastAI library to train
- Using accuracy for the metrics to check the performance of the model
# Links
- For further information about the model (medium):
- Try the model (Streamlit): https://adhd-detector-enhjts2povdcwcmtk6txtu.streamlit.app/
