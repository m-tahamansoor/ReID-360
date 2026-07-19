import torch
import cv2
import faiss
import networkx
import torchreid
import ultralytics
import fastapi

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("FAISS OK")
print("NetworkX OK")
print("Torchreid OK")
print("Ultralytics OK")
print("FastAPI OK")