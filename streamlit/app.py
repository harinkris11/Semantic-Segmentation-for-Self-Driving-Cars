import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import matplotlib.pyplot as plt
from models.unet import unet_model
import streamlit as st
# import cv2

st. set_page_config(layout="wide")
st.title('Semantic Segmentation for Self Driving Cars')
uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
c1, c2 = st.columns(2)
if uploaded_file is not None:
    c1.image(uploaded_file)


if 'unet_model' not in st.session_state:
	st.session_state.unet_model = unet_model()
	st.session_state.unet_model.load_state_dict(torch.load('unet_model_scratch.pth',map_location=torch.device('cpu')))


if uploaded_file:

	img = np.array(Image.open(uploaded_file))
	t1 = A.Compose([
	    A.Resize(160,240),
	    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
	    ToTensorV2()
	])

	aug = t1(image=img)
	x = aug['image']
	x = x.unsqueeze(0)


	softmax = nn.Softmax(dim=1)
	preds = torch.argmax(softmax(st.session_state.unet_model(x)),axis=1)
	img1 = np.transpose(np.array(x[0,:,:,:]),(1,2,0))
	preds1 = np.array(preds[0,:,:])
	print(preds1.shape)
	fig, ax = plt.subplots()
	ax.imshow(preds1)
	c2.pyplot(fig)
