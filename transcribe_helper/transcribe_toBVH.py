from bvh_skeleton import h36m_skeleton
import numpy as np

if __name__ == "__main__":
    # remove them if you don't want to use.
    customPose = np.load("./input.npy")
    output = './output.bvh'

def preprocessingForNeck(customPose):
    # generate neck node
    neck = np.add(customPose[:,1],customPose[:,2])/2
    neck = np.expand_dims(neck, 1)
    customPose = np.append(customPose, neck, axis = 1)
    return customPose

def makeBvhWithNpyFile(customPose, output = "./output.bvh"):
    customPose = preprocessingForNeck(customPose)
    h36m_skel = h36m_skeleton.H36mSkeleton()
    h36m_skel.poses2bvh(customPose, None, output)