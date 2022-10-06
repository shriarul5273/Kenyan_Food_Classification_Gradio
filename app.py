import onnxruntime
from torchvision import transforms
import torch
import torch.nn.functional as F
import gradio as gr

orst_run = onnxruntime.InferenceSession("model.onnx")

idx_to_class = {0: 'chapati', 
                1: 'mukimo', 
                2: 'kukuchoma',
                3: 'kachumbari',
                4: 'ugali',
                5: 'githeri',
                6: 'matoke',
                7: 'pilau',
                8: 'nyamachoma',
                9: 'sukumawiki',
                10: 'bhaji',
                11: 'mandazi',
                12: 'masalachips'}



def predict(image):
    preprocess = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    output = orst_run.run(None, {'input': input_batch.numpy()})
    output = torch.from_numpy(output[0])
      
    prediction=F.softmax(output,dim=1)
    predProab,predIndexs = torch.topk(prediction, 3)

    predProab = predProab.numpy()[0]
    predIndexs = predIndexs.numpy()[0]
    confidences = {idx_to_class[predIndexs[i]]: float(predProab[i]) for i in range(3)}

    return confidences

def inference(img):
    return predict(img)




title = 'Kenyan Food Classification'
description = "Kenyan Food Classification"

examples = ['1.jpg','2.jpg','3.jpg','4.jpg']

gr.Interface(inference, gr.Image(type="pil",source="webcam"), "label", server_name="0.0.0.0",title=title,
                                description=description, examples=examples).launch()