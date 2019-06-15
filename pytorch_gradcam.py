import cv2
import numpy as np
import torch


class GradCam:
    def __init__(self, model, target_layer):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.target_layer = target_layer
        self.gradient = None

    def __call__(self, input_, class_index=None, cam_size=None):
        cam = self.unprocessed_cam(input_, class_index)
        cam = self.process(cam, cam_size)
        return cam

    def save_gradient(self, gradient):
        self.gradient = gradient[0].cpu().data.numpy()

    def unprocessed_cam(self, input_, class_index=None):

        x = input_.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        ''' if VGG '''
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                feature_maps = x[0].cpu().data.numpy()
        x = self.model.classifier(x.view(1, -1))
        ''''''

        ''' if ResNet '''
        # for name, module in self.model._modules.items():
        #     if name == 'fc':
        #         x = x.view(1, -1)
        #     x = module(x)
        #     if name == self.target_layer:
        #         x.register_hook(self.save_gradient)
        #         feature_maps = x[0].cpu().data.numpy()
        ''''''

        if class_index == None:
            class_index = np.argmax(x.cpu().data.numpy())

        one_hot = np.zeros((1, x.shape[-1]), dtype=np.float32)
        one_hot[0, class_index] = 1
        one_hot = torch.from_numpy(one_hot)
        objective_value = torch.sum(one_hot.to(self.device) * x)

        self.model.zero_grad()
        objective_value.backward()

        weights = np.mean(self.gradient, axis=(1, 2))

        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i, :, :]

        return cam

    def process(self, cam, cam_size=None):
        cam = np.maximum(cam, 0)
        if cam_size:
            cam = cv2.resize(cam, cam_size)
        cam = cam - np.min(cam)
        cam = cam / (cam.max() if cam.max() > 0 else 1)
        return cam


if __name__ == '__main__':
    ''' Sample '''
    import torchvision.models as models

    model = models.vgg16(pretrained=True)
    gradcam = GradCam(model, target_layer='29')
    # model = models.vgg19(pretrained=True)
    # gradcam = GradCam(model, target_layer='35')
    # model = models.resnet50(pretrained=True)
    # gradcam = GradCam(model, target_layer='layer4')

    img = cv2.imread('both.png')
    img = cv2.resize(img, (224, 224))
    img = img / 255
    input_ = img.transpose(2, 0, 1)
    input_ = torch.from_numpy(input_).to(dtype=torch.float)

    cam = gradcam(input_, None, (224, 224))
    # cam = gradcam(input_, 243, (224, 224))  # 243:'bull mastiff'
    # cam = gradcam(input_, 254, (224, 224))  # 254:'pug, pug-dog'
    # cam = gradcam(input_, 282, (224, 224))  # 282:'tiger cat'

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    cv2.imwrite('cam.jpg', np.uint8(cam * 255))
