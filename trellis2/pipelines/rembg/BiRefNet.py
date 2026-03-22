from typing import Optional, Any
from transformers import AutoModelForImageSegmentation, AutoConfig, PretrainedConfig
import torch
from torchvision import transforms
from PIL import Image

if not hasattr(PretrainedConfig, "is_encoder_decoder"):
    PretrainedConfig.is_encoder_decoder = False

class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet", cache_dir: Optional[str] = None):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        for target in [config, config.__class__]:
            if not hasattr(target, "is_encoder_decoder"):
                try:
                    setattr(target, "is_encoder_decoder", False)
                except:
                    pass
            
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, 
            config=config,
            trust_remote_code=True, 
            low_cpu_mem_usage=False, 
            device_map=None,
            cache_dir=cache_dir
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)
    
    def eval(self):
        self.model.eval()
        return self

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to(next(self.model.parameters()).device)
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    