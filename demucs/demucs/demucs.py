import rclpy
from rclpy.node import Node

from jack_msgs.msg import JackAudio

import math
import time
import torch
import numpy as np

import inspect
import denoiser

class DemucsPhaseROSAudio(Node):
  def __init__(self, hop_secs=0.5):
    super().__init__('demucs')
    
    self.device = "cuda"
    self.subscription = self.create_subscription(JackAudio, '/jackaudio', self.jackaudio_callback,10)
    self.subscription  # prevent unused variable warning
    self.publisher = self.create_publisher(JackAudio, '/jackaudio_filtered', 10)
    
    this_share_directory = get_package_share_directory('demucs')
    this_base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(this_share_directory))))
    this_src_directory = os.path.join(this_base_directory,"src","demucs")
    self.demucs_modelpath = os.path.join(this_src_directory,"pretrained_model","best.th")
    print("Using the following pretrained model: "+self.demucs_modelpath)
    
    self.demucs = self.load_demucsmodel(self.demucs_modelpath, device=self.device)
    self.past_start = time.time()
    
  def jackaudio_callback(self, msg):
    capt_time = time.time() - self.past_start
    
    start_time = time.time()
    noisy_win = torch.tensor(msg.data,device=self.device).unsqueeze(0).unsqueeze(0)
    interf_signal = noisy_win.clone() #this is ignored by the current version of demucs, but is required
    noisy_win = self.combine_interf (noisy_win,interf_signal) #this is ignored by the current version of demucs, but is required
    filt_win = self.demucs(noisy_win)[0][0].tolist()
    exec_time = time.time() - start_time
    self.get_logger().info('capture time: %f, response time: %f' % (capt_time, exec_time))
    
    msg_filt = JackAudio()
    msg_filt.size = len(filt_win)
    msg_filt.header.stamp = self.get_clock().now().to_msg()
    msg_filt.data = filt_win
    self.publisher.publish(msg_filt)
    
    self.past_start = time.time()

  def combine_interf (self, signal,interf):
    return torch.cat((signal,interf),2)

  def deserialize_model(self, package, strict=False):
    return model
  
  def load_demucsmodel(self, model_path, device="cuda"):
    package = torch.load(model_path, map_location=device)
    
    klass = package['class']
    kwargs = package['kwargs']
    
    sig = inspect.signature(klass)
    kw = package['kwargs']
    for key in list(kw):
      if key not in sig.parameters:
        del kw[key]
    model = klass(*package['args'], **kw)
    model.load_state_dict(package['state'])
    
    model.to(device)
    return model


def main(args=None):
  rclpy.init(args=args)
  demucsphaserosaudio = DemucsPhaseROSAudio()
  rclpy.spin(demucsphaserosaudio)

  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  demucsphaserosaudio.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()
