#Compute analytics such as confidence intervals/standard errors

from scipy.stats import norm, t
import numpy as np
import math

#Compute width of confidence interval
#or just standard error
#Input: sample standard deviation [float]
#number of items in sample [int]
#confidence level (if want a CI width not just standard error) [float 0-1]
#Output: width of confidence interval (on 1 side of the mean) [float]
#  or standard error
def error_width(sample_std, num_items, confidence_level=None):
  std_err = sample_std / math.sqrt(num_items)
  if confidence_level is None:
    return std_err

  stat = t.ppf(confidence_level, num_items)

  ci_width = stat*std_err
  return ci_width

#Compute sample standard deviation
#Input: list to compute standard deviation of [list]
#   OR the mean of a binary RV [float 0-1]
#   (e.g. ML classification results)
#Output: sample standard deviation [float]
def sample_std(sample):
  variance = None
  if type(sample) is float:
    if sample < 0 or sample > 1:
      raise ValueError("not a valid mean of binary variable")
    #can easily derive this as the variance of a binary RV
    variance = sample*(1 - sample)
  elif type(sample) is list:
    variance = np.var(np.asarray(sample))
  return math.sqrt(variance)






