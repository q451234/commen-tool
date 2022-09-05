from torchvision import transforms, datasets
from torch.utils.data import DataLoader,  RandomSampler
from PIL import Image

from matplotlib import pyplot as plt

class ResizePadding(object):
    """Reszie and padding
    Args:
        img_size: (h,w)
        mode (Boolean): True for normal; False for center
    """
 
    def __init__(self, img_size, padding = (0, 0, 0), mode = True):
        self.img_size = img_size
        self.mode = mode
        self.padding =  padding
 
    def __call__(self, image):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        w = self.img_size[0]
        h = self.img_size[1]

        iw, ih = image.size

        scale   = min(w / iw, h / ih)
        nw      = int(iw * scale)
        nh      = int(ih * scale)

        image       = image.resize((nw, nh), Image.BICUBIC)
        new_image   = Image.new('RGB', [w, h], self.padding)
        if(self.mode):
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image.paste(image, (0, 0))

        # label       = label.resize((nw,nh), Image.NEAREST)
        # new_label   = Image.new('L', [w, h], (0))
        # new_label.paste(label, ((w-nw)//2, (h-nh)//2))
        
        return new_image 


transform_test = transforms.Compose([
    # transforms.Resize((args.img_size, args.img_size)),
    ResizePadding((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])      

testset = datasets.ImageFolder(root="./data/train/",
                            transform=transform_test)

test_sampler = RandomSampler(testset)

test_loader = DataLoader(testset,
                        sampler=test_sampler,
                        batch_size=4,
                        pin_memory=True)


for i, data in enumerate(test_loader):

    inputs, labels = data   # B C H W

    img_tensor = inputs[0, ...]     # C H W
    print(img_tensor.size())
    img = transforms.ToPILImage()(img_tensor)
    
    # img = transform_invert(img_tensor, train_transform)
    plt.imshow(img)
    plt.show()
    plt.pause(0.5)
    plt.close()
    break
