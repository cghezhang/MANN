"""
Data loaderv and processer
"""
import tensorflow as tf

"""global parameters"""
num_trajPoints       = 12 #number of trajectory points
num_trajUnit_noSpeed = 6  #number of trajectory units: Position X,Z; Direction X,Z; Velocity X,Z;           
num_trajUnit_speed   = 7  #number of trajectory units: Position X,Z; Direction X,Z; Velocity X,Z; Speed
num_jointUnit        = 12 #number of joint units: PositionXYZ Rotation VelocityXYZ


#get the velocity of joints, desired velocity and style
def getDogFVSS(data, index_joint, num_styles):    
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

