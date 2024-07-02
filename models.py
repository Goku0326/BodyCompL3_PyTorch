import torch
import torch.nn as nn
import torch.nn.init as init


#[b,h,w,c] -> [b,c,h,w]

def double_conv(in_c,out_c,dropout_rate):
    conv = nn.Sequential(
        nn.Conv2d(in_c,out_c,kernel_size = 3,padding='same'),
        nn.PReLU(),
        nn.BatchNorm2d(out_c),
        nn.Dropout(dropout_rate),
        nn.Conv2d(out_c,out_c,kernel_size = 3,padding='same'),
        nn.PReLU(),
        nn.BatchNorm2d(out_c)
    )
    return conv
        
class UNet(nn.Module):
    def __init__(self,params,num_classes):
        super(UNet, self).__init__()
        #num_classes = params.dict['num_classes_bc']
        dropout_rate = params.dict['dropout_rate']
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1,32,dropout_rate)
        self.down_conv_2 = double_conv(32,64,dropout_rate)
        self.down_conv_3 = double_conv(64,128,dropout_rate)
        self.down_conv_4 = double_conv(128,256,dropout_rate)
        self.down_conv_5 = double_conv(256,512,dropout_rate)
        
        
        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2)
        
        self.up_conv_1 = double_conv(512,256,dropout_rate)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2)
        
        self.up_conv_2 = double_conv(256,128,dropout_rate)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2)
        
        self.up_conv_3 = double_conv(128,64,dropout_rate)   

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2)
        
        self.up_conv_4 = double_conv(64,32,dropout_rate)     

        self.out = nn.Conv2d(
            in_channels = 32,
            out_channels = num_classes, 
            kernel_size = 1
        )
        
    def forward(self,image):
        
        x1 = self.down_conv_1(image) 
        #after each convolution apply max_pooling
        p1 = self.max_pool_2x2(x1)

        x2 = self.down_conv_2(p1) 
        p2 = self.max_pool_2x2(x2)

        x3 = self.down_conv_3(p2)    
        p3 = self.max_pool_2x2(x3)

        x4 = self.down_conv_4(p3) 
        p4 = self.max_pool_2x2(x4)  

        x5 = self.down_conv_5(p4) 

        #decoder
        u6 = self.up_trans_1(x5)
        x6 = self.up_conv_1(torch.cat([u6,x4],1))
        
        u7 = self.up_trans_2(x6)
        x7 = self.up_conv_2(torch.cat([u7,x3],1))

        u8 = self.up_trans_3(x7)
        x8 = self.up_conv_3(torch.cat([u8,x2],1))

        u9 = self.up_trans_4(x8)
        x9 = self.up_conv_4(torch.cat([u9,x1],1)) 

        x = self.out(x9)
        return x
        
        

if __name__ == "__main__":
    params = load_params('params.json')
    image = torch.rand((1, 1, 512, 512))
    model = UNet(params)
    outputs = model(image)
    print(outputs.size())  
     

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
