#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: JonnyFLDN
"""
import shapefile
import requests
import zipfile
import io
import os
import glob
import numpy as np
from collections import OrderedDict
from rtree import index
from itertools import chain
from shapely.geometry import shape,Point


class Zipcode_Search(object):
    zip_url = 'http://www2.census.gov/geo/tiger/GENZ2017/shp/cb_2017_us_zcta510_500k.zip'
    zip_name = os.path.join('./',os.path.basename(os.path.normpath(zip_url))).replace('.zip','')
    #used to find column position of attribute_label in shapefile
    attribute_label = 'ZCTA5CE10'

    def __init__(self,shapefile_location=None):
        
        if shapefile_location == None:
            #download files and save in path
            self.download_files()
        self.load_shapes()
    
    def load_shapes(self):

        myshp = open(glob.glob(Zipcode_Search.zip_name+"/*.shp")[0],"rb")
        mydbf = open(glob.glob(Zipcode_Search.zip_name+"/*.dbf")[0],"rb")
        self.shapefile_reader = shapefile.Reader(shp =myshp,dbf =mydbf)

        
    def download_files(self):
        ''' Download and store Cartographic files from census.gov '''
        
        print('Downloading Cartographic Boundary Shapefiles...')

        r = requests.get(Zipcode_Search.zip_url,stream=True)
        z = zipfile.Zipfile(io.BytesIO(r.content))
        z.extractall(Zipcode_Search.zip_name)
        
 
    @staticmethod
    def haversine_dist(lat1, lon1, lat2, lon2,r =3959.87433):
        ''' Haversine distance calculation
        
        Parameters
        ----------
        lat1,lon1,lat2,lon2: float
            latitude/longiude in degrees
        
        r: float
            Radius of earth in miles 
        
        Returns
        ----------
        m: Haversine distance matrix in miles
        
        '''
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
        dlon = lon2 - lon1
        dlat = lat2 - lat1
    
        a = np.sin(dlat*0.5)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon*0.5)**2
        m =2 * np.arcsin(np.sqrt(a)) *r
        return m

    def get_shapefile(self):
        ''' Retrieve raw shapefile '''
        return self.shapefile_reader
    
    def get_shapes(self,zip_find=None):
        ''' Retrieve shapes from shapefile
        
        Parameters
        ----------
        zip_find: int
            5-digit zip code to find

        Returns
        ----------
        Ordered dict: dictionary with zip code as key and shapes as value
        
        '''
        #Find position of field numbers
        field_num = self.shapefile_reader.fields[1:][0].index(Zipcode_Search.attribute_label)
        all_zip_codes = [record[field_num] for record in self.shapefile_reader.iterRecords()]
        shape_recs = np.asarray(self.shapefile_reader.shapeRecords())
        
        if zip_find == None:
            return OrderedDict((s.record[0],s.shape) for s in shape_recs[:])
        else:
            _,_,zidx = np.intersect1d(zip_find,all_zip_codes,False,True)
            
            if zidx.size >0:
                return OrderedDict((s.record[0],s.shape) for s in shape_recs[zidx])
            else:
                return "zip codes not found"
                
    def coord_neighbour(self,lat_lon,n_num =5,zip_restrict =None):
        ''' Find the closest zip codes to given coordinates
        
        Parameters
        ----------
        lat_lon: array-like
            latitude and longitude in degrees to find
        
        n_num: int
            max number of closest zip-codes to show
        
        zip_restrict: array-like
            contains 5-digit zip codes to restrict in search
            
        Returns
        ----------
        Ordered dict: contains coordinate number as key 
                      and tuples of (zip-code, haversince distance)
                      
        '''
        
        lat_lon = np.asarray(lat_lon) 
        lat_lon = np.fliplr(lat_lon.reshape(1,2)) if lat_lon.ndim == 1 else np.fliplr(lat_lon)
        
        shape_dic = self.get_shapes(zip_find=zip_restrict)
        
        #Load r-tree 
        index_tree = index.Index(list((i,tuple(obj.bbox),None) for i,(_,obj) in enumerate(shape_dic.items())))
        
        #Grab a minimum of 5 nearest neighbours
        srch_num = max(5,n_num)
        all_zip = np.asarray(list(shape_dic.keys()))

        coord_near= [all_zip[list(index_tree.nearest((n.tolist()),srch_num))] for n in lat_lon]
        unique_near = set(list(chain(*coord_near)))

        d_shapes = {z:shape(shape_dic[z]) for z in unique_near} 
        dist_list = OrderedDict()
        
        for idx,c in enumerate(lat_lon):
            cd_list = []
            p= Point(c)
            for z in coord_near[idx]:
                if p.within(d_shapes[z]):
                    cd_list.append((z,0))
                else:
                    z_coord = np.asarray(shape_dic[z].points)
                    min_dist = self.haversine_dist(c[1],c[0],z_coord[:,1],z_coord[:,0]).min()
                    cd_list.append((z,min_dist))
            
            dist_list[idx] = sorted(cd_list,key=lambda x:x[1])[:n_num]
        
        return dist_list
