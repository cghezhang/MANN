"""
Data loaderv and processer
"""
import numpy as np
import tensorflow as tf
import Utils as utils

"""global parameters"""
offset_Begin  = 3 #skip first 3 dims
offset_End    = 3 #ranslational offset and rotational offset
num_trajPoints       = 12 #trajectory points
num_trajUnit_noSpeed = 6  #trajectory units: Position X,Z; Direction X,Z; Velocity X,Z;           
num_trajUnit_speed   = 7  #trajectory units: Position X,Z; Direction X,Z; Velocity X,Z; Speed
num_jointUnit        = 12 #joint units: PositionXYZ Rotation VelocityXYZ

def Load_Speed(filename, savefile, num_joint, num_style):
    data = np.float32(np.loadtxt(filename))
    frameCount = data.shape[0]
    
    A = []
    B = []
    C = []
    for i in range(frameCount-2):
        if(data[i,0] == data[i+1,0] and data[i,0] == data[i+2,0]):
            A.append(data[i])
            B.append(data[i+1])
            C.append(data[i+2])
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)    
    
    jointNeurons      = num_jointUnit* num_joint     
    trajectoryUnit    = num_trajUnit_speed+ num_style
    trajectoryNeurons = trajectoryUnit* num_trajPoints 

    #input 
    X = np.concatenate(
            (
                    B[:,offset_Begin+jointNeurons:offset_Begin+jointNeurons+trajectoryNeurons], 
                    A[:,offset_Begin:offset_Begin+jointNeurons]
            ),axis = 1) 

    Traj_out = np.float32(np.zeros((A.shape[0],np.int(num_trajPoints/2* num_trajUnit_noSpeed))))
    Traj_out_start = np.int(offset_Begin+ jointNeurons+ num_trajPoints/2*trajectoryUnit)
    for i in range(np.int(num_trajPoints/2)):
        Traj_out[:,i*num_trajUnit_noSpeed:(i+1)*num_trajUnit_noSpeed] = C[:,Traj_out_start:Traj_out_start+num_trajUnit_noSpeed]
        Traj_out_start += trajectoryUnit    
        
    Y = np.concatenate(
            (
                    Traj_out, 
                    B[:,offset_Begin:offset_Begin+jointNeurons], 
                    C[:,offset_Begin+jointNeurons+trajectoryNeurons:offset_Begin+jointNeurons+trajectoryNeurons+offset_End]
            ),axis = 1)
    
    
    
    #normalization
    Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
    Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)
    for i in range(Xstd.size):
        if (Xstd[i]==0):
            Xstd[i]=1
    for i in range(Ystd.size):
        if (Ystd[i]==0):
            Ystd[i]=1
    X = (X - Xmean) / Xstd
    Y = (Y - Ymean) / Ystd
    
    
    utils.build_path([savefile])
    Xmean.tofile(savefile+'/Xmean.bin')
    Ymean.tofile(savefile+'/Ymean.bin')
    Xstd.tofile(savefile+'/Xstd.bin')
    Ystd.tofile(savefile+'/Ystd.bin')
    input_x = X
    input_y = Y
    
    input_size  = input_x.shape[1]
    output_size = input_y.shape[1]
    
    number_example =input_x.shape[0]
    
    print("Data is processed")
    print("input dim: ", input_size)
    print("output dim: ", output_size)
    
    return input_x, input_y, input_size, output_size, number_example




#get the velocity of joints, desired velocity and style
def getDogFeetsVSS(data, index_joint, num_styles):    
    trajectoryUnit    = num_trajUnit_speed + num_styles
    trajectoryNeurons = trajectoryUnit* num_trajPoints 
    
    bone0 = index_joint[0] * num_jointUnit + trajectoryNeurons
    bone1 = index_joint[1] * num_jointUnit + trajectoryNeurons
    bone2 = index_joint[2] * num_jointUnit + trajectoryNeurons
    bone3 = index_joint[3] * num_jointUnit + trajectoryNeurons
    
    style_start  = trajectoryUnit * 6      
    
    gating_input  = tf.concat([data[...,bone0+9:bone0+12], 
                              data[...,bone1+9:bone1+12], 
                              data[...,bone2+9:bone2+12],
                              data[...,bone3+9:bone3+12],
                              data[...,style_start+num_trajUnit_noSpeed:style_start+trajectoryUnit]],
                             axis = -1)
    input_size_gt =  4*3+1+num_styles
    return gating_input, input_size_gt 

