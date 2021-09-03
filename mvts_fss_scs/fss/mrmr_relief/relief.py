from mvts_fss_scs.fss.base_fss import BaseFSS


class RELIEF(BaseFSS):

  def __init__(self,data = None,**kwargs):
  
    if data:
      self.data = data
    else:
      super().__init__(data)
    
    """
    1. This is where use initialize the keyword arguments you need for your algorithm.
    Refer pie.py or csfs.py to get a rough idea of the implementation.

    2. You need to implement a method called rank which returns your scores (dataframe) in desceneding order
      in the following format.

      Features | Score

    3. Create an instance of the above class in the main __init__file
    """
  def rank(self):
    pass
