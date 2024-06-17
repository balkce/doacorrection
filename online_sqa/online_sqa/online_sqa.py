import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from jack_msgs.msg import JackAudio

import math
import time
import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE
import numpy as np
from pykalman import KalmanFilter

from threading import Thread

class OptScale(torch.nn.Module):
  def __init__(self, device):
    super(OptScale, self).__init__()
    init_scaling_factor = torch.zeros(1, device=device)
    self.scaling_factor = torch.nn.Parameter(init_scaling_factor, requires_grad=True)

  def forward(self, x):
    output = self.scaling_factor * x
    return output

class OnlineSQA(Node):
  def __init__(self, hop_secs=0.1, win_len_secs=3.0):
    super().__init__('onlinesqa')
    
    self.device = "cuda:0"
    self.subscription = self.create_subscription(JackAudio,'/jackaudio_filtered',self.jackaudio_callback,10)
    #self.subscription = self.create_subscription(JackAudio,'/jackaudio',self.jackaudio_callback,10)
    self.subscription  # prevent unused variable warning
    self.publisher = self.create_publisher(Float32, '/SDR', 10)
    
    self.objective_model = SQUIM_OBJECTIVE.get_model().to(self.device)
    
    self.samplerate = 16000
    self.hop_secs = hop_secs
    self.hop_len = int(round(self.hop_secs*self.samplerate))
    self.hop_i = 0
    self.hop_i_ref = 0
    self.win_len_secs = win_len_secs
    self.win_len = int(round(self.win_len_secs*self.samplerate))
    self.win = torch.zeros(1,self.win_len)
    
    self.num_hops = int(self.win_len/self.hop_len)
    self.num_hops_towait = 2
    self.win_sdr_start = 0
    self.win_sdr_end = 0
    
    self.get_logger().info('sample rate (ass.): %d' % self.samplerate)
    self.get_logger().info('window length     : %f (%d samples)' % (self.win_len_secs, self.win_len))
    self.get_logger().info('hop length        : %f (%d samples, %d hops in window)' % (self.hop_secs, self.hop_len, self.num_hops))
    
    self.sqa_ready = False
    self.sqa_ready_ref = False
    self.sqa_thread = Thread(target=self.do_sqa)
    self.sqa_thread.start()
    
    self.kf_firststep = True
    self.kf_last_filtered_state_mean = 0.0
    self.kf_last_filtered_state_covariance = 0.0
    self.kf_this_filtered_state_mean = 0.0
    self.kf_this_filtered_state_covariance = 0.0
    self.kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    
    #self.kf = KalmanFilter(
    #  initial_state_mean=0,
    #  transition_matrices=[1],
    #  observation_matrices=[1],
    #  observation_covariance=np.array([0.1]),
    #  transition_covariance=np.array([0.1])
    #)
    
    self.smooth_weight = 0.5 # to use with kalman filter 0.5
    self.smooth_val = None
  
    self.vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
    self.vad_threshold = 0.15
    self.vad_confs = torch.zeros(self.num_hops)
    
  def jackaudio_callback(self, msg):
    self.win = torch.roll(self.win,len(msg.data))
    self.win[0,0:len(msg.data)] = torch.Tensor(msg.data)
    self.hop_i += len(msg.data)
    
    if self.hop_i >= self.hop_len:
      #start_time = time.time()
      new_confidence = self.vad(self.win[0,0:self.hop_i], self.samplerate).item()
      #exec_time = time.time() - start_time
      #self.get_logger().info('VAD confidence: %0.2f, in %f secs' % (new_confidence, exec_time))
      
      self.vad_confs = torch.roll(self.vad_confs,1)
      self.vad_confs[0] = 1 if new_confidence > self.vad_threshold else 0
      
      if torch.sum(self.vad_confs) == self.num_hops:
        #the user has not shut up for the whole window
        self.win_sdr_start = 0
        self.win_sdr_end = self.win_len
        while self.sqa_ready:
          time.sleep(0.01)
        self.sqa_ready = True
      elif self.vad_confs[0] == 1:
        #the user is talking
        vad_where = torch.where(self.vad_confs[:-1] != self.vad_confs[1:])[0]
        if vad_where.shape[0] > 0:
          hop_num_stop = vad_where[0].item()+1
          if hop_num_stop > self.num_hops_towait:
            self.win_sdr_start = 0
            win_i = hop_num_stop*self.hop_i
            self.win_sdr_end = win_i if win_i < self.win_len else self.win_len
              
            while self.sqa_ready:
              time.sleep(0.01)
            self.sqa_ready = True
      
      self.hop_i = 0
  
  def smooth(self,data_point):
    if self.smooth_val == None:
      this_smooth_val = data_point
    else:
      this_smooth_val = self.smooth_val * self.smooth_weight + (1 - self.smooth_weight) * data_point
    
    self.smooth_val = this_smooth_val
    return this_smooth_val

  def do_sqa(self):
    while True:
      while not self.sqa_ready:
        time.sleep(0.001)
      
      #start_time = time.time()
      win_clone = self.win.to(self.device)
      stoi_hyp, pesq_hyp, si_sdr_hyp = self.objective_model(win_clone[0,self.win_sdr_start:self.win_sdr_end].unsqueeze(0))
      unfiltered_observation = si_sdr_hyp.item()
      if math.isnan(unfiltered_observation):
        unfiltered_observation = 0.0
      #exec_time = time.time() - start_time
      
      if self.kf_firststep:
        self.kf_last_filtered_state_mean = unfiltered_observation
        self.kf_firststep = False
      else:
        self.kf_last_filtered_state_mean = self.kf_this_filtered_state_mean
        self.kf_last_filtered_state_covariance = self.kf_this_filtered_state_covariance
      
      self.kf_this_filtered_state_mean, self.kf_this_filtered_state_covariance = (
          self.kf.filter_update(
              self.kf_last_filtered_state_mean,
              self.kf_last_filtered_state_covariance,
              unfiltered_observation
          )
      )
      filtered_observation = self.kf_this_filtered_state_mean[0].item()
      filtered_observation_smooth = self.smooth(filtered_observation)
      self.get_logger().info('SDR: %0.4f, filtered: %0.4f, smooth: %0.4f' % (unfiltered_observation,filtered_observation,filtered_observation_smooth))
      
      msg = Float32()
      msg.data = filtered_observation_smooth
      self.publisher.publish(msg)
      self.sqa_ready = False


def main(args=None):
  rclpy.init(args=args)
  onlinesqa = OnlineSQA()
  rclpy.spin(onlinesqa)

  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  onlinesqa.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()
