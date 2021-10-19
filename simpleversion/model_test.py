
from simpleversion.modeling.detector.ssd_detector import SSDDetector
from simpleversion.config import cfg
import torch

def main():
    model = SSDDetector(cfg)
    model.train()
    model.cuda()
    images = torch.rand((1, 3, 300, 300), dtype=torch.float32).cuda()
    boxes = torch.rand((1, 8732, 4), dtype=torch.float32).cuda()
    labels = torch.randint(1, 21, (1, 8732), dtype=torch.int64).cuda()
    targets = {'boxes': boxes, 'labels': labels}
    loss_dict = model(images, targets=targets)
    print('done')



if __name__ == '__main__':
    main()

