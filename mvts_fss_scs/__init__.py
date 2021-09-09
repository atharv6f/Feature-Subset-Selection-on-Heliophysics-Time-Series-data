
from CONSTANTS import RESULTS

# from mvts_fss_scs.fss.clever.clever import CLEVER
#from mvts_fss_scs.fss.corona.corona import CORONA
#from mvts_fss_scs.fss.csfs.csfs import CSFS
#from mvts_fss_scs.fss.fcbf.fcbf import FCBF
#from mvts_fss_scs.fss.mrmr_relief.mrmr import MRMR
#from mvts_fss_scs.fss.mrmr_relief.relief import RELIEF
#from mvts_fss_scs.fss.pie.pie import PIE
#from mvts_fss_scs.fss.rfe.rfe import RFE
from mvts_fss_scs.fss.utils import save_table
import os

#from mvts_fss_scs.fss.clever.clever import CLEVER


if __name__ == '__main__':
  save_path = RESULTS
  # pie = PIE(n_neighbors = 5,mode='connectivity' )
  # pie_rank = pie.rank()
  #csfs = CSFS()
  #csfs_rank = csfs.rank()
  # corona = CORONA()
  # corona_ranks = corona.rank()
  # fcbf = FCBF()
  # fcbf_rank = fcbf.rank()
  # clever = CLEVER()
  # clever_rank = clever.rank()
  #save_table(save_path, pie_rank, "pie_rank")
  #save_table(save_path, csfs_rank, "csfs_rank")
  # save_table(save_path, clever_rank, "clever_rank")
  # save_table(save_path, fcbf_rank, "fcbf_rank")
  # save_table(save_path, corona_ranks,"corona_ranks")

