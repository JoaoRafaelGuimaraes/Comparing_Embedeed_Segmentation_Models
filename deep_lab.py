# deeplab_adapter.py
import torch
import numpy as np
import cv2
import torch
import torchvision.models.segmentation as models
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import time

class DeepLabResult:
    def __init__(self, prediction, inf_time):
        self.prediction = prediction
        self.speed = {"inference": inf_time}


class DeepLab:
    def __init__(self,num_classes=2, model_path="models/landslide1000_200_adam.pth"):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=self.device)

        
                
        for k in ("state_dict", "model_state_dict"):
            if isinstance(state_dict, dict) and k in state_dict:
                state_dict = state_dict[k]

       
        if len(state_dict) and next(iter(state_dict)).startswith("module."):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

       
        num_classes_sd = state_dict["classifier.4.weight"].shape[0]  # == 2

        try:
            model = models.deeplabv3_resnet101(weights=None, weights_backbone=None,
                                            aux_loss=False, num_classes=num_classes_sd)
        except TypeError:
        
            model = models.deeplabv3_resnet101(weights=None, weights_backbone=None, aux_loss=False)
            model.classifier[4] = nn.Conv2d(256, num_classes_sd, kernel_size=1, bias=True)

        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        self.model = model.to(self.device).eval()
        print("Modelo pronto para inferência ✅")


        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),     # ou mantenha o tamanho nativo e depois faça interpolate nos logits
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])


        # model.predict(
        #         source=os.path.join(test_paths, img_path),
        #         imgsz=imgsz,
        #         device=device,
        #         verbose=False
        #     )
    def predict(self, source, imgsz=512, device=None, verbose=False):
        img = cv2.imread(source)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.last_img = img_rgb.copy()
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                torch.cuda.synchronize()  
            start = time.time()
            out = self.model(input_tensor)["out"][0] 
            if self.device.type == "cuda":
                torch.cuda.synchronize()            
            end = time.time() 
            prediction = out.argmax(0).cpu().numpy()  

        inf_time = (end - start) * 1000 
        self.last_prediction = prediction
        return [DeepLabResult(prediction, inf_time)]

    def plot(self):
        img_512 = cv2.resize(self.last_img, (512, 512), interpolation=cv2.INTER_LINEAR)

        # 3) cores e overlay
        colors = np.array([[0,0,0], [255,0,0]], dtype=np.uint8)
        mask = colors[self.last_prediction]                            # (512,512,3)
        overlay = (0.6 * img_512 + 0.4 * mask).astype(np.uint8)

        # 4) plot
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].imshow(img_512); ax[0].set_title("Imagem (512×512)"); ax[0].axis("off")
        ax[1].imshow(mask);    ax[1].set_title("Máscara predita");   ax[1].axis("off")
        ax[2].imshow(overlay); ax[2].set_title("Overlay");           ax[2].axis("off")
        plt.tight_layout(); plt.show()

