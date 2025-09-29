#BASICS
import numpy as np
import pandas as pd
from datetime import datetime
import itertools
#CARTOPY
import cartopy.geodesic
geod = cartopy.geodesic.Geodesic()
import cartopy.crs as ccrs
import cartopy.feature as cfeature
land_feature=cfeature.NaturalEarthFeature(category='physical',name='land',scale='50m',facecolor=[0.4,0.6,0.7])
#MATPLOTLIB
import matplotlib.pyplot as plt
#GRID
from scipy.interpolate import griddata
# NEARNEIGHBOR SCIKIT-LEARN
from sklearn.neighbors import NearestNeighbors

# Distance calculation function with cartopy geodesic
def dist(p1,p2):
    d,a1,a2 = np.array(geod.inverse(p1,p2).T)
    return d[0]/1000.0

#LOAD DATA
DAT=pd.read_csv('ar_index_global_prof.txt',
                sep=',',
                index_col=None,
                header=0, 
                skiprows=8,
                parse_dates=[1, 7])

#TIME DATA SUBSET
YY=datetime.today().year
MM=datetime.today().month
if MM==1:
    PM=12
    PY=YY-1
else:
    PM=MM-1
    PY=YY
    
tquery='date >= "'+str(PY)+'-'+str(PM)+'" & date < "'+str(YY)+'-'+str(MM)+'"'
MDAT=DAT.query(tquery)    
MDAT=MDAT[~MDAT.latitude.isnull() & ~MDAT.longitude.isnull()]
MDAT=MDAT.query('longitude >= -180 & longitude <= 180 & latitude >= -90 & latitude <= 90')

#NEARNEIGHBOR OVER A GRID
sx = np.arange(-180,180,1./4)
sy = np.arange(-70,80,1./4)
#pour que scikit nearneighbors les ingère, on les passe en serie de points
grid_points = np.array(list(itertools.product(sx, sy)))
fit_points = list(zip(MDAT['longitude'].values,MDAT['latitude'].values))
nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(fit_points)
inds = nbrs.kneighbors(grid_points,return_distance=False)

# RECALCUL DES DISTANCES GEODESIQUES:
mdist=np.empty(len(grid_points))
for i in range(inds.shape[0]):
    di=0
    for j in [1,2,3,4]:
        d_temp=dist(np.array([grid_points[i,0],grid_points[i,1]]),
                    np.array([MDAT['longitude'].values[inds[i,j]],MDAT['latitude'].values[inds[i,j]]]))
        di+=d_temp
    di=di/4.0
    mdist[i]=di
# REGRIDDING    
grid_sx, grid_sy = np.meshgrid(sx, sy)
grid_sz=griddata((grid_points[:,0],grid_points[:,1]),mdist,(grid_sx,grid_sy),method='cubic')    

fig=plt.figure(figsize=(15,10),dpi=200)
pr1=ccrs.PlateCarree()
pr2=ccrs.Robinson()
ax = fig.add_subplot(1, 1, 1, projection=pr2)
hh=ax.contourf(grid_sx,grid_sy,grid_sz,np.arange(0,800,100),cmap='coolwarm',extend='neither',transform=pr1)
ax.add_feature(land_feature, edgecolor='black')
cbar=plt.colorbar(hh,orientation= 'vertical', shrink=0.3, pad=0.02)
cbar.set_label('km')
ax.set_extent([-180, 180, -70, 80], crs=pr1)
plt.title('Argo sparseness (km) for '+str(PY)+'-'+str(PM))
fig.savefig('PNG/'+str(PY)+'-'+str(PM)+'.png',dpi=200,facecolor='w',edgecolor='none',bbox_inches='tight', pad_inches=0.1)
fig.savefig('PNG/Last.png',dpi=200,facecolor='w',edgecolor='none',bbox_inches='tight', pad_inches=0.1)



