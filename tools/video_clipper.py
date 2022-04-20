# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:24:12 2021

@author: jankos
"""
import pandas as pd
import numpy as np
from skimage import io, transform
import cv2
import collections

class VideoFrame:
    """
    Video frame definition, user can define their own. The frame class should
    have a frame parameter and __eq__ method
    """
    def __init__(self, action, carrying, frame):
        self.action = action
        self.carrying = carrying
        self.frame = frame
        
    def __eq__(self, frame2):
        return self.action == frame2.action and \
                self.carrying == frame2.carrying and \
                self.frame.shape == frame2.frame.shape
                
class VideoFrameDrill:
    """
    Video frame definition, user can define their own. The frame class should
    have a frame parameter and __eq__ method
    """
    def __init__(self, drilling, frame):
        self.drilling = drilling
        self.frame = frame
        
    def __eq__(self, frame2):
        return self.drilling == frame2.drilling and \
                self.frame.shape == frame2.frame.shape
                
class VideoClip:
    #to be done, class for individual clips
    pass
        
class Event(list):
    def __call__(self, *args, **kwargs):
        for item in self:
            item(*args, **kwargs)
            
class VideoClipWatcher:
    """
    Class that watches video clips and checks if the clip has changed or if 
    the clip has been approved
    """
    def __init__(self):
        self.VideoClipChanged = Event() 
        self.VideoClipApproved = Event()

class VideoClipReader(VideoClipWatcher):
    """
    Reads frames from video and adds them into a collections.deque that
    represents the video clip. Sends out a singnal every time clip changes.
    """
    def __init__(self, clip_length):
        super().__init__()
        self.clip_length = clip_length
        self.clip = collections.deque([])

    def append(self,frame):
        self.clip.append(frame)
        if len(self.clip)>self.clip_length:
            self.clip.popleft()
            self.VideoClipChanged(self.clip)
            
class VideoClipApprover(VideoClipWatcher):
    """
    If video clip changes, checks if the clip fulfills some
    'condition' given by the user and sends a signal with the approved clip
    """
    def __init__(self, clipreaders, condition):
        super().__init__()
        self.condition = condition
        for clipreader in clipreaders:
            clipreader.VideoClipChanged.append(self.can_create)
    def can_create(self, clip):
        if self.condition(clip):
            self.VideoClipApproved(clip)
        else:
            self.VideoClipApproved(None)
        
class VideoClipCreator:
    """
    Takes a stack of frames (video clip) and rehsapes them into a single image
    so that the individual frames are stacked vertically.
    """
    def __init__(self, approver=None, name=''):
        self.name = name
        self.approver = approver
        if approver is not None:
            approver.VideoClipApproved.append(self.create_clip)
        self.clip = None
    
    def create_clip(self, clip):
        if clip is not None:
            h,w,ch = clip[0].frame.shape
            array = np.array([x.frame for x in clip]).reshape(-1, w, ch)
            self.clip = array
        else:
            self.clip = None