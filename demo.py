import torch
import sys
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
import glob


#img_path = 'demo.jpg'
#img_path = 'nyu_rgb_0001.png' # use sample from nyuv2

model = create_model(opt)

input_height = 480
input_width  = 640


#(654, 480, 640)

def test_simple(model):
    #imgfolder = ''
    

    test_img_names = sorted(glob.glob("/work/NYUv2/nyu_test_rgb/*.png"))
    train_img_names = sorted(glob.glob("/work/NYUv2/nyu_train_rgb/*.png"))

    total_loss = 0 
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()
    print(len(test_img_names))
    print(len(train_img_names))
    test_depth = np.zeros((len(test_img_names),480,640))
    train_depth = np.zeros((len(train_img_names),480,640))

    for i in range(len(test_img_names)):
        #print("first_iteration")
        img_path = test_img_names[i]
        img = np.float32(io.imread(img_path))/255.0
        img = resize(img, (input_height, input_width), order = 1)
        input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img.cuda())
        pred_log_depth = model.netG.forward(input_images) 
        pred_log_depth = torch.squeeze(pred_log_depth)
        
        pred_depth = torch.exp(pred_log_depth)


        pred_depth = pred_depth.data.cpu().numpy()
        test_depth[i,:,:] = pred_depth

        #print(pred_depth.shape)
        
    for i in range(len(train_img_names)):
    #print("first_iteration")
        img_path = train_img_names[i]
        img = np.float32(io.imread(img_path))/255.0
        img = resize(img, (input_height, input_width), order = 1)
        input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img.cuda())
        pred_log_depth = model.netG.forward(input_images) 
        pred_log_depth = torch.squeeze(pred_log_depth)
        
        pred_depth = torch.exp(pred_log_depth)


        pred_depth = pred_depth.data.cpu().numpy()
        train_depth[i,:,:] = pred_depth
    #print(pred_depth.shape)

    savefolder = '/work/NYUv2_DE'
    np.savez("%s/%s_depth_estimation_megadepth.npz" % (savefolder, "train"), train_depth)
    np.savez("%s/%s_depth_estimation_megadepth.npz" % (savefolder, "test"), test_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    #pred_inv_depth = 1/pred_depth
    #pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    #pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

    #io.imsave('demo.png', pred_inv_depth)
    #io.imsave('test.png', pred_inv_depth)

    # print(pred_inv_depth.shape)
    sys.exit()



test_simple(model)
print("We are done")
