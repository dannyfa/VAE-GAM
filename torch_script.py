import torch
import torch.nn as nn
import numpy as np


def get_fc_shape(input_shape, conv_layers, z_dim):
    x = torch.ones((1,1)+input_shape)
    for layer in conv_layers:
        x = layer(x)
    return x.shape[1:]


def get_layers(input_shape, num_conv_layers=5, num_encoder_shared_fc=3, \
    num_encoder_separate_fc=2, nf=8, z_dim=32):
    """
    Construct a 3d convolutional network.

    Returns
    -------
    layers : dict
        Maps keys to layers.
    """
    layers = {
        'encoder_conv':[],
        'encoder_shared_fc':[],
        'encoder_mu_fc':[],
        'encoder_u_fc':[],
        'encoder_d_fc':[],
        'post_conv_len': None,
        'post_conv_shape': None,
        'decoder_fc': [],
        'decoder_convt': [],
    }
    # Make the conv and convt layers.
    x = torch.ones((1,1)+input_shape)
    print(x.shape)
    for i in range(num_conv_layers):
        nf_1, nf_2 = max(1,((i+1)//2)*nf), ((i+2)//2)*nf
        stride = 1 + (i % 2)
        temp_layer = nn.Conv3d(nf_1,nf_2,3,stride)
        layers['encoder_conv'].append(temp_layer)
        x = temp_layer(x)
        print(x.shape)
        if stride == 1:
            temp_layer = nn.ConvTranspose3d(nf_2,nf_1,3,stride)
        else:
            # NOTE: Something is wrong here with the output padding.
            out_pad = tuple(int(j%2==0) for j in x.shape[-3:])
            temp_layer = nn.ConvTranspose3d(nf_2, nf_1,3,stride,output_padding=out_pad)
        layers['decoder_convt'] = [temp_layer] + layers['decoder_convt']
    # Make the shared fc layers.
    layers['post_conv_shape'] = x.shape[1:]
    s1 = np.prod(x.shape)
    layers['post_conv_len'] = s1
    x = x.view(1,-1)
    print(x.shape)
    num_fc = num_encoder_shared_fc+num_encoder_separate_fc
    seq = [int(round(np.exp(i))) for i in np.linspace(np.log(s1),np.log(z_dim),num_fc+1)]
    for i in range(num_encoder_shared_fc):
        layers['encoder_shared_fc'].append(nn.Linear(seq[i],seq[i+1]))
        x = layers['encoder_shared_fc'][-1](x)
        print(x.shape, "encoder shared")
    # Make the separate fc layers
    keys = ['encoder_mu_fc', 'encoder_u_fc', 'encoder_d_fc']
    for i in range(num_encoder_separate_fc):
        index = num_encoder_shared_fc + i
        for j, key in enumerate(keys):
            layers[key].append(nn.Linear(seq[index],seq[index+1]))
            if j == 0:
                x = layers[key][-1](x)
                print(x.shape, "encoder separate")
    # Make the decoder fc layers.
    seq = seq[::-1]
    for i in range(num_fc):
        layers['decoder_fc'].append(nn.Linear(seq[i], seq[i+1]))
        x = layers['decoder_fc'][-1](x)
        print(x.shape, "decoder fc")
    print(x.shape)
    x = x.reshape((1,)+layers['post_conv_shape'])
    # Run through the convt layers.
    for i in range(num_conv_layers):
        x = layers['decoder_convt'][i](x)
        print(x.shape, "convt")
    return layers


def test_layers(layers, input_shape, z_dim):
    pass


if __name__ == '__main__':
    input_shape = (112,112,64)
    get_layers(input_shape, num_conv_layers=7)



###
